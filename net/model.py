import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import (
    DegradationClassificationHead,
    DegradationGuidedTransformerStage,
    DegradationRepresentationModule,
    Downsample,
    OverlapPatchEmbed,
    TransformerBlock,
    Upsample,
)
from .STFR import (
    ConditionalRestorationRefinement,
    CRR_POSITIONS,
    TaskRelevanceFusion,
    TRF_POSITIONS,
)


def _transformer_stage(dim, depth, heads, expansion, bias, norm_type):
    return nn.Sequential(*[
        TransformerBlock(dim, heads, expansion, bias, norm_type)
        for _ in range(depth)
    ])


def _degradation_guided_stage(
    dim, depth, degradation_dim, heads, expansion, bias, norm_type,
):
    return DegradationGuidedTransformerStage(
        dim, depth, degradation_dim, heads, expansion, bias, norm_type,
    )


def _dataset_num_degradations():
    try:
        from utils.task_dataset_utils import NUM_DEGRADATIONS
    except ImportError:
        from ..utils.task_dataset_utils import NUM_DEGRADATIONS
    return int(NUM_DEGRADATIONS)


class TaskIRNet(nn.Module):
    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=(4, 6, 6, 8),
        num_refinement_blocks=4,
        heads=(1, 2, 4, 8),
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type="WithBias",
        feedback_dim=64,
        degradation_dim=128,
        num_degradations=None,
    ):
        super().__init__()
        num_blocks, heads = tuple(num_blocks), tuple(heads)
        dims = (dim, dim * 2, dim * 4, dim * 8)

        self.inp_channels = int(inp_channels)
        self.feedback_dim = int(feedback_dim)
        self.degradation_dim = int(degradation_dim)
        self.num_degradations = (
            _dataset_num_degradations()
            if num_degradations is None else int(num_degradations)
        )
        d1, d2, d3, d4 = dims

        self.degradation_representation = DegradationRepresentationModule(
            inp_channels, d1, self.degradation_dim, bias=bias,
        )
        self.degradation_classifier = DegradationClassificationHead(
            self.degradation_dim, self.num_degradations,
        )

        self.patch_embed = OverlapPatchEmbed(inp_channels, d1, bias)
        self.encoder_level1 = _transformer_stage(
            d1, num_blocks[0], heads[0], ffn_expansion_factor, bias, LayerNorm_type,
        )
        self.down1_2 = Downsample(d1)
        self.encoder_level2 = _transformer_stage(
            d2, num_blocks[1], heads[1], ffn_expansion_factor, bias, LayerNorm_type,
        )
        self.down2_3 = Downsample(d2)
        self.encoder_level3 = _transformer_stage(
            d3, num_blocks[2], heads[2], ffn_expansion_factor, bias, LayerNorm_type,
        )
        self.down3_4 = Downsample(d3)
        self.latent = _degradation_guided_stage(
            d4, num_blocks[3], self.degradation_dim, heads[3],
            ffn_expansion_factor, bias, LayerNorm_type,
        )

        self.up4_3 = Upsample(d4)
        self.reduce_chan_level3 = nn.Conv2d(d3 * 2, d3, 1, bias=bias)
        self.decoder_level3 = _degradation_guided_stage(
            d3, num_blocks[2] + 1, self.degradation_dim, heads[2],
            ffn_expansion_factor, bias, LayerNorm_type,
        )

        self.up3_2 = Upsample(d3)
        self.reduce_chan_level2 = nn.Conv2d(d3, d2, 1, bias=bias)
        self.decoder_level2 = _degradation_guided_stage(
            d2, num_blocks[1] + 1, self.degradation_dim, heads[1],
            ffn_expansion_factor, bias, LayerNorm_type,
        )

        self.up2_1 = Upsample(d2)
        self.reduce_chan_level1 = nn.Conv2d(d2, d1, 1, bias=bias)
        self.decoder_level1 = _degradation_guided_stage(
            d1, num_blocks[0] + num_refinement_blocks, self.degradation_dim, heads[0],
            ffn_expansion_factor, bias, LayerNorm_type,
        )
        self.output = nn.Conv2d(d1, out_channels, 3, padding=1, bias=bias)
        self._trf_position_channels = (d4, d3, d2)
        self.trf = None
        self.crr = None

    def attach_trf(
        self, prototype_dim=64, num_prototypes=4, prototype_temperature=0.1,
        sinkhorn_epsilon=0.05, sinkhorn_iterations=30,
        route_temperature=1.0,
    ):
        self.trf = TaskRelevanceFusion(
            restoration_channels=self._trf_position_channels,
            feedback_dim=self.feedback_dim,
            prototype_dim=prototype_dim,
            num_prototypes=num_prototypes,
            prototype_temperature=prototype_temperature,
            sinkhorn_epsilon=sinkhorn_epsilon,
            sinkhorn_iterations=sinkhorn_iterations,
            route_temperature=route_temperature,
        )
        reference = next(self.parameters())
        self.trf.to(device=reference.device, dtype=reference.dtype)
        return self

    def attach_crr(
        self, internal_dim=64, num_heads=4, correction_scale_initial=0.1,
    ):
        if self.trf is None:
            raise RuntimeError("Attach TRF before attaching CRR.")
        self.crr = ConditionalRestorationRefinement(
            feature_dims=self._trf_position_channels,
            internal_dim=internal_dim,
            num_heads=num_heads,
            correction_scale_initial=correction_scale_initial,
        )
        reference = next(self.parameters())
        self.crr.to(device=reference.device, dtype=reference.dtype)
        return self

    def get_trf_config(self):
        return None if self.trf is None else self.trf.get_config()

    def get_crr_config(self):
        return None if self.crr is None else self.crr.get_config()

    @staticmethod
    def _pad_to_multiple(image, multiple=8):
        height, width = image.shape[-2:]
        pad_h, pad_w = (-height) % multiple, (-width) % multiple
        if pad_h == 0 and pad_w == 0:
            return image
        mode = "reflect" if pad_h < height and pad_w < width else "replicate"
        return F.pad(image, (0, pad_w, 0, pad_h), mode=mode)

    def forward(self, inp_img, feedback_feat=None, return_aux=False):
        original_size = inp_img.shape[-2:]
        degradation, spatial_degradation = self.degradation_representation(
            inp_img, return_spatial=True,
        )
        inp_img = self._pad_to_multiple(inp_img)
        if self.trf is not None and self.crr is None:
            raise RuntimeError("Stage-II TaskIRNet has TRF attached but no CRR.")
        trf_context = None
        trf_auxiliary = None
        crr_auxiliary = None
        if self.trf is not None and feedback_feat is not None:
            trf_context = self.trf.prepare_task_feedback(
                feedback_feat, return_aux=return_aux,
            )
            if return_aux:
                trf_auxiliary = {
                    "task_assignments": trf_context["task_assignments"],
                    "positions": {},
                }
                crr_auxiliary = {"positions": {}}

        enc1 = self.encoder_level1(self.patch_embed(inp_img))
        enc2 = self.encoder_level2(self.down1_2(enc1))
        enc3 = self.encoder_level3(self.down2_3(enc2))
        latent = self.latent(
            self.down3_4(enc3), degradation, spatial_degradation,
        )
        if trf_context is not None:
            trf_result = self.trf.forward_position(
                0, latent, trf_context, return_aux=return_aux,
            )
            if return_aux:
                latent_feedback, position_aux = trf_result
                trf_auxiliary["positions"][TRF_POSITIONS[0]] = position_aux
            else:
                latent_feedback = trf_result
            crr_result = self.crr.forward_position(
                0, latent, latent_feedback, return_aux=return_aux,
            )
            if return_aux:
                latent, position_aux = crr_result
                crr_auxiliary["positions"][CRR_POSITIONS[0]] = position_aux
            else:
                latent = crr_result

        dec3 = self.reduce_chan_level3(torch.cat((self.up4_3(latent), enc3), dim=1))
        dec3 = self.decoder_level3(dec3, degradation, spatial_degradation)
        if trf_context is not None:
            trf_result = self.trf.forward_position(
                1, dec3, trf_context, return_aux=return_aux,
            )
            if return_aux:
                dec3_feedback, position_aux = trf_result
                trf_auxiliary["positions"][TRF_POSITIONS[1]] = position_aux
            else:
                dec3_feedback = trf_result
            crr_result = self.crr.forward_position(
                1, dec3, dec3_feedback, return_aux=return_aux,
            )
            if return_aux:
                dec3, position_aux = crr_result
                crr_auxiliary["positions"][CRR_POSITIONS[1]] = position_aux
            else:
                dec3 = crr_result

        dec2 = self.reduce_chan_level2(torch.cat((self.up3_2(dec3), enc2), dim=1))
        dec2 = self.decoder_level2(dec2, degradation, spatial_degradation)
        if trf_context is not None:
            trf_result = self.trf.forward_position(
                2, dec2, trf_context, return_aux=return_aux,
            )
            if return_aux:
                dec2_feedback, position_aux = trf_result
                trf_auxiliary["positions"][TRF_POSITIONS[2]] = position_aux
            else:
                dec2_feedback = trf_result
            crr_result = self.crr.forward_position(
                2, dec2, dec2_feedback, return_aux=return_aux,
            )
            if return_aux:
                dec2, position_aux = crr_result
                crr_auxiliary["positions"][CRR_POSITIONS[2]] = position_aux
            else:
                dec2 = crr_result

        dec1 = self.reduce_chan_level1(torch.cat((self.up2_1(dec2), enc1), dim=1))
        dec1 = self.decoder_level1(dec1, degradation, spatial_degradation)
        restored = self.output(dec1) + inp_img
        height, width = original_size
        restored = restored[..., :height, :width]
        if not return_aux:
            return restored
        auxiliary = {
            "z_d": degradation,
            "degradation_logits": self.degradation_classifier(degradation),
        }
        if trf_auxiliary is not None:
            auxiliary["trf"] = trf_auxiliary
        if crr_auxiliary is not None:
            auxiliary["crr"] = crr_auxiliary
        return restored, auxiliary
