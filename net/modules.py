import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def to_3d(x):
    return rearrange(x, "b c h w -> b (h w) c")


def to_4d(x, height, width):
    return rearrange(x, "b (h w) c -> b c h w", h=height, w=width)


class BiasFreeLayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(torch.Size(normalized_shape)))

    def forward(self, x):
        variance = x.var(-1, keepdim=True, unbiased=False)
        return x * torch.rsqrt(variance + 1e-5) * self.weight


class WithBiasLayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        variance = x.var(-1, keepdim=True, unbiased=False)
        return (x - mean) * torch.rsqrt(variance + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, layer_norm_type):
        super().__init__()
        if layer_norm_type == "BiasFree":
            self.body = BiasFreeLayerNorm(dim)
        elif layer_norm_type == "WithBias":
            self.body = WithBiasLayerNorm(dim)
        else:
            raise ValueError("layer_norm_type must be 'BiasFree' or 'WithBias'.")

    def forward(self, x):
        height, width = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), height, width)


class FeedForward(nn.Module):
    def __init__(self, dim, expansion_factor, bias):
        super().__init__()
        hidden_dim = int(dim * expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_dim * 2, 1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_dim * 2, hidden_dim * 2, 3, padding=1,
            groups=hidden_dim * 2, bias=bias,
        )
        self.project_out = nn.Conv2d(hidden_dim, dim, 1, bias=bias)

    def forward(self, x):
        x1, x2 = self.dwconv(self.project_in(x)).chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3, 3, padding=1, groups=dim * 3, bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, 1, bias=bias)

    def forward(self, x):
        _, _, height, width = x.shape
        q, k, v = self.qkv_dwconv(self.qkv(x)).chunk(3, dim=1)
        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        attention = ((q @ k.transpose(-2, -1)) * self.temperature).softmax(dim=-1)
        out = attention @ v
        out = rearrange(
            out, "b head c (h w) -> b (head c) h w",
            head=self.num_heads, h=height, w=width,
        )
        return self.project_out(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, expansion_factor, bias, layer_norm_type):
        super().__init__()
        self.norm1 = LayerNorm(dim, layer_norm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, layer_norm_type)
        self.ffn = FeedForward(dim, expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class MultiScaleDegradationExtraction(nn.Module):
    def __init__(self, dim, bias=False):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    dim, dim, 3, padding=dilation, dilation=dilation,
                    groups=dim, bias=bias,
                ),
                nn.GELU(),
            )
            for dilation in (1, 2, 3)
        ])
        self.fuse = nn.Conv2d(dim * len(self.branches), dim, 1, bias=bias)

    def forward(self, x):
        multi_scale = torch.cat([branch(x) for branch in self.branches], dim=1)
        return x + self.fuse(multi_scale)


class GlobalStandardDeviationPooling(nn.Module):
    def forward(self, x):
        return x.std(dim=(-2, -1), unbiased=False)


class DegradationRepresentationModule(nn.Module):
    def __init__(self, in_channels, feature_dim, degradation_dim, bias=False):
        super().__init__()
        self.degradation_dim = int(degradation_dim)
        self.stem = nn.Conv2d(in_channels, feature_dim, 3, padding=1, bias=bias)
        self.mde = MultiScaleDegradationExtraction(feature_dim, bias=bias)
        self.project = nn.Conv2d(feature_dim, feature_dim, 1, bias=bias)
        self.gsp = GlobalStandardDeviationPooling()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim * 2, self.degradation_dim),
            nn.GELU(),
            nn.Linear(self.degradation_dim, self.degradation_dim),
        )
        self.spatial_branch = nn.Sequential(
            nn.Conv2d(
                feature_dim, feature_dim, 3, padding=1,
                groups=feature_dim, bias=bias,
            ),
            nn.GELU(),
            nn.Conv2d(feature_dim, 1, 1, bias=True),
        )
        nn.init.normal_(self.spatial_branch[-1].weight, mean=0.0, std=1e-6)
        nn.init.zeros_(self.spatial_branch[-1].bias)

    def forward(self, image, return_spatial=False):
        feature = self.project(self.mde(self.stem(image)))
        gap = feature.mean(dim=(-2, -1))
        gsp = self.gsp(feature)
        degradation = self.mlp(torch.cat((gsp, gap), dim=1))
        if not return_spatial:
            return degradation
        spatial_degradation = self.spatial_branch(feature)
        return degradation, spatial_degradation


class DegradationModulation(nn.Module):
    def __init__(self, degradation_dim, feature_dim):
        super().__init__()
        self.degradation_dim = int(degradation_dim)
        self.feature_dim = int(feature_dim)
        self.norm = nn.LayerNorm(self.degradation_dim)
        self.expand = nn.Linear(self.degradation_dim, self.feature_dim * 2)
        self.activation = nn.GELU()
        self.project = nn.Linear(self.feature_dim * 2, self.feature_dim * 6)
        nn.init.normal_(self.project.weight, mean=0.0, std=1e-6)
        nn.init.zeros_(self.project.bias)

    def forward(self, degradation):
        parameters = self.project(self.activation(self.expand(self.norm(degradation))))
        return parameters.chunk(6, dim=1)


class DegradationGuidedTransformerBlock(nn.Module):
    def __init__(
        self, dim, num_heads, expansion_factor, bias, layer_norm_type,
    ):
        super().__init__()
        self.dim = int(dim)
        self.norm1 = LayerNorm(dim, layer_norm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, layer_norm_type)
        self.ffn = FeedForward(dim, expansion_factor, bias)

    def forward(self, x, modulation_parameters, spatial_modulation):
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = modulation_parameters

        normalized1 = self.norm1(x)
        modulated1 = (1.0 + torch.tanh(gamma1)) * normalized1 + beta1
        x1 = x + (
            (1.0 + torch.tanh(alpha1))
            * spatial_modulation
            * self.attn(modulated1)
        )

        normalized2 = self.norm2(x1)
        modulated2 = (1.0 + torch.tanh(gamma2)) * normalized2 + beta2
        return x1 + (
            (1.0 + torch.tanh(alpha2))
            * spatial_modulation
            * self.ffn(modulated2)
        )


class DegradationGuidedTransformerStage(nn.Module):
    def __init__(
        self, dim, depth, degradation_dim, num_heads,
        expansion_factor, bias, layer_norm_type,
    ):
        super().__init__()
        self.dim = int(dim)
        self.modulation = DegradationModulation(degradation_dim, dim)
        self.blocks = nn.ModuleList([
            DegradationGuidedTransformerBlock(
                dim, num_heads, expansion_factor, bias, layer_norm_type,
            )
            for _ in range(depth)
        ])

    def forward(self, x, degradation, spatial_degradation):
        modulation_parameters = tuple(
            parameter[:, :, None, None]
            for parameter in self.modulation(degradation)
        )
        if spatial_degradation.shape[-2:] != x.shape[-2:]:
            spatial_degradation = F.interpolate(
                spatial_degradation, size=x.shape[-2:],
                mode="bilinear", align_corners=False,
            )
        spatial_degradation = spatial_degradation.to(device=x.device, dtype=x.dtype)
        spatial_modulation = 2.0 * torch.sigmoid(spatial_degradation)
        for block in self.blocks:
            x = block(x, modulation_parameters, spatial_modulation)
        return x


class DegradationClassificationHead(nn.Module):
    def __init__(self, degradation_dim, num_degradations):
        super().__init__()
        self.classifier = nn.Linear(degradation_dim, num_degradations)

    def forward(self, degradation):
        return self.classifier(degradation)


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_channels, embed_dim, bias):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, 3, padding=1, bias=bias)

    def forward(self, x):
        return self.proj(x)


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 3, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.body(x)
