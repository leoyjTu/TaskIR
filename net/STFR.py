import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .modules import LayerNorm


TRF_POSITIONS = ("latent", "decoder_level3", "decoder_level2")
TRF_FEEDBACK_LEVELS = (1, 2, 3)
CRR_POSITIONS = ("latent", "decoder_level3", "decoder_level2")


def _projection(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(int(in_channels), int(out_channels), 1),
        LayerNorm(int(out_channels), "WithBias"),
    )


class SoftPrototypeExtractor(nn.Module):
    def __init__(self, prototype_dim=64, num_prototypes=4, temperature=0.1):
        super().__init__()
        self.prototype_dim = int(prototype_dim)
        self.num_prototypes = int(num_prototypes)
        self.temperature = float(temperature)

    def forward(self, projected_feature, prototype_anchors, return_assignment=False):
        tokens = projected_feature.flatten(2).transpose(1, 2)
        with torch.autocast(device_type=projected_feature.device.type, enabled=False):
            normalized_tokens = F.normalize(tokens.float(), dim=-1, eps=1e-6)
            normalized_anchors = F.normalize(
                prototype_anchors.float(), dim=-1, eps=1e-6,
            )
            assignment_logits = torch.einsum(
                "bnd,md->bnm", normalized_tokens, normalized_anchors,
            ) / self.temperature
            assignment = assignment_logits.softmax(dim=-1)
            token_values = tokens.float()
            denominator = assignment.sum(dim=1).unsqueeze(-1).clamp_min(1e-6)
            prototypes = (
                torch.einsum("bnm,bnd->bmd", assignment, token_values)
                / denominator
            )
        if return_assignment:
            return prototypes, assignment
        return prototypes


class SinkhornPrototypeMatcher(nn.Module):
    def __init__(self, epsilon=0.05, iterations=30):
        super().__init__()
        self.epsilon = float(epsilon)
        self.iterations = int(iterations)

    def _sinkhorn(self, similarity):
        count = similarity.shape[1]
        log_affinity = similarity.float() / self.epsilon
        log_affinity = log_affinity - log_affinity.amax(
            dim=(1, 2), keepdim=True,
        )
        log_marginal = -math.log(count)
        log_u = torch.zeros_like(log_affinity[:, :, 0])
        log_v = torch.zeros_like(log_affinity[:, 0, :])
        for _ in range(self.iterations):
            log_u = log_marginal - torch.logsumexp(
                log_affinity + log_v[:, None, :], dim=2,
            )
            log_v = log_marginal - torch.logsumexp(
                log_affinity + log_u[:, :, None], dim=1,
            )
        return torch.exp(log_affinity + log_u[:, :, None] + log_v[:, None, :])

    def forward(self, restoration_prototypes, task_prototypes, return_aux=False):
        with torch.autocast(
            device_type=restoration_prototypes.device.type, enabled=False,
        ):
            restoration = F.normalize(
                restoration_prototypes.float(), dim=-1, eps=1e-6,
            )
            task = F.normalize(task_prototypes.float(), dim=-1, eps=1e-6)
            similarity = torch.matmul(restoration, task.transpose(1, 2))
            transport = self._sinkhorn(similarity)
            relevance = (transport * similarity).sum(dim=(1, 2))
        if return_aux:
            return relevance, similarity, transport
        return relevance


class TaskRelevanceFusionStage(nn.Module):
    def __init__(
        self, restoration_channels, feedback_dim=64, prototype_dim=64,
        route_temperature=1.0,
    ):
        super().__init__()
        self.restoration_channels = int(restoration_channels)
        self.feedback_dim = int(feedback_dim)
        self.route_temperature = float(route_temperature)
        self.restoration_projection = _projection(
            self.restoration_channels, prototype_dim,
        )
        self.alignment_projections = nn.ModuleList([
            nn.Conv2d(self.feedback_dim, self.restoration_channels, 1)
            for _ in TRF_FEEDBACK_LEVELS
        ])

    def forward(
        self, restoration_feature, feedback_features, task_prototypes,
        prototype_anchors, prototype_extractor, prototype_matcher,
        return_aux=False,
    ):
        projected_restoration = self.restoration_projection(restoration_feature)
        if return_aux:
            restoration_prototypes, restoration_assignment = prototype_extractor(
                projected_restoration, prototype_anchors, return_assignment=True,
            )
        else:
            restoration_prototypes = prototype_extractor(
                projected_restoration, prototype_anchors,
            )
            restoration_assignment = None

        route_scores, similarities, transports = [], [], []
        for task_prototype in task_prototypes:
            if return_aux:
                relevance, similarity, transport = prototype_matcher(
                    restoration_prototypes, task_prototype, return_aux=True,
                )
                similarities.append(similarity)
                transports.append(transport)
            else:
                relevance = prototype_matcher(
                    restoration_prototypes, task_prototype,
                )
            route_scores.append(relevance)
        route_scores = torch.stack(route_scores, dim=1)
        route_weights = (route_scores / self.route_temperature).softmax(dim=1)

        target_size = restoration_feature.shape[-2:]
        aligned_feedback = []
        for feedback, projection in zip(feedback_features, self.alignment_projections):
            aligned = projection(feedback)
            if aligned.shape[-2:] != target_size:
                aligned = F.interpolate(
                    aligned, size=target_size, mode="bilinear", align_corners=False,
                )
            aligned_feedback.append(aligned)
        aligned_stack = torch.stack(aligned_feedback, dim=1)
        fused_feedback = (
            aligned_stack
            * route_weights.to(aligned_stack)[:, :, None, None, None]
        ).sum(dim=1).to(restoration_feature.dtype)

        if not return_aux:
            return fused_feedback
        return fused_feedback, {
            "restoration_assignment": restoration_assignment,
            "restoration_prototypes": restoration_prototypes,
            "route_scores": route_scores,
            "route_weights": route_weights,
            "similarities": tuple(similarities),
            "transports": tuple(transports),
        }


class TaskRelevanceFusion(nn.Module):
    def __init__(
        self, restoration_channels=(384, 192, 96), feedback_dim=64,
        prototype_dim=64, num_prototypes=4, prototype_temperature=0.1,
        sinkhorn_epsilon=0.05, sinkhorn_iterations=30,
        route_temperature=1.0,
    ):
        super().__init__()
        self.restoration_channels = tuple(int(value) for value in restoration_channels)
        self.feedback_dim = int(feedback_dim)
        self.prototype_dim = int(prototype_dim)
        self.num_prototypes = int(num_prototypes)
        self.prototype_temperature = float(prototype_temperature)
        self.sinkhorn_epsilon = float(sinkhorn_epsilon)
        self.sinkhorn_iterations = int(sinkhorn_iterations)
        self.route_temperature = float(route_temperature)

        self.prototype_anchors = nn.Parameter(
            torch.empty(self.num_prototypes, self.prototype_dim),
        )
        nn.init.normal_(self.prototype_anchors, mean=0.0, std=0.02)
        self.prototype_extractor = SoftPrototypeExtractor(
            self.prototype_dim, self.num_prototypes, self.prototype_temperature,
        )
        self.prototype_matcher = SinkhornPrototypeMatcher(
            self.sinkhorn_epsilon, self.sinkhorn_iterations,
        )
        self.task_prototype_projections = nn.ModuleList([
            _projection(self.feedback_dim, self.prototype_dim)
            for _ in TRF_FEEDBACK_LEVELS
        ])
        self.stages = nn.ModuleList([
            TaskRelevanceFusionStage(
                channels, self.feedback_dim, self.prototype_dim,
                self.route_temperature,
            )
            for channels in self.restoration_channels
        ])

    @staticmethod
    def _feedback_tuple(feedback):
        if isinstance(feedback, dict):
            features = tuple(
                feedback.get(level, feedback.get(str(level)))
                for level in TRF_FEEDBACK_LEVELS
            )
        elif isinstance(feedback, (list, tuple)) and len(feedback) == 3:
            features = tuple(feedback)
        else:
            raise TypeError("TRF feedback must be a three-level dictionary or sequence.")
        if any(feature is None for feature in features):
            raise ValueError("TRF requires feedback levels 1, 2 and 3.")
        return features

    def prepare_task_feedback(self, feedback, return_aux=False):
        feedback_features = self._feedback_tuple(feedback)
        task_prototypes, task_assignments = [], []
        for feature, projection in zip(
            feedback_features, self.task_prototype_projections,
        ):
            projected = projection(feature)
            if return_aux:
                prototypes, assignment = self.prototype_extractor(
                    projected, self.prototype_anchors, return_assignment=True,
                )
                task_assignments.append(assignment)
            else:
                prototypes = self.prototype_extractor(
                    projected, self.prototype_anchors,
                )
            task_prototypes.append(prototypes)
        return {
            "feedback_features": feedback_features,
            "task_prototypes": tuple(task_prototypes),
            "task_assignments": tuple(task_assignments) if return_aux else None,
        }

    def forward_position(
        self, position, restoration_feature, task_context, return_aux=False,
    ):
        if isinstance(position, str):
            position = TRF_POSITIONS.index(position)
        position = int(position)
        if not 0 <= position < len(self.stages):
            raise IndexError(f"TRF position must be in [0, 2], got {position}.")
        return self.stages[position](
            restoration_feature,
            task_context["feedback_features"],
            task_context["task_prototypes"],
            self.prototype_anchors,
            self.prototype_extractor,
            self.prototype_matcher,
            return_aux=return_aux,
        )

    def forward(self, restoration_features, feedback, return_aux=False):
        task_context = self.prepare_task_feedback(feedback, return_aux=return_aux)
        fused, position_aux = [], []
        for position, restoration_feature in enumerate(restoration_features):
            result = self.forward_position(
                position, restoration_feature, task_context, return_aux=return_aux,
            )
            if return_aux:
                current_fused, current_aux = result
                fused.append(current_fused)
                position_aux.append(current_aux)
            else:
                fused.append(result)
        if not return_aux:
            return tuple(fused)
        return tuple(fused), {
            "task_assignments": task_context["task_assignments"],
            "positions": {
                name: values for name, values in zip(TRF_POSITIONS, position_aux)
            },
        }

    def get_config(self):
        return {
            "prototype_dim": self.prototype_dim,
            "num_prototypes": self.num_prototypes,
            "prototype_temperature": self.prototype_temperature,
            "sinkhorn_epsilon": self.sinkhorn_epsilon,
            "sinkhorn_iterations": self.sinkhorn_iterations,
            "route_temperature": self.route_temperature,
        }


class EfficientTaskCrossAttention(nn.Module):
    def __init__(self, feature_dim, internal_dim=64, num_heads=4):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.internal_dim = int(internal_dim)
        self.num_heads = int(num_heads)

        self.temperature = nn.Parameter(torch.ones(self.num_heads, 1, 1))
        self.q_projection = nn.Conv2d(
            self.feature_dim, self.internal_dim, 1, bias=False,
        )
        self.q_local = nn.Conv2d(
            self.internal_dim, self.internal_dim, 3, padding=1,
            groups=self.internal_dim, bias=False,
        )
        self.kv_projection = nn.Conv2d(
            self.feature_dim, self.internal_dim * 2, 1, bias=False,
        )
        self.kv_local = nn.Conv2d(
            self.internal_dim * 2, self.internal_dim * 2, 3, padding=1,
            groups=self.internal_dim * 2, bias=False,
        )
        self.output_projection = nn.Conv2d(
            self.internal_dim, self.feature_dim, 1,
        )

    def forward(self, restoration_feature, task_feedback, return_aux=False):
        q = self.q_local(self.q_projection(restoration_feature))
        k, v = self.kv_local(self.kv_projection(task_feedback)).chunk(2, dim=1)
        height, width = restoration_feature.shape[-2:]
        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        normalized_q = F.normalize(q, dim=-1, eps=1e-6)
        normalized_k = F.normalize(k, dim=-1, eps=1e-6)
        attention = (
            torch.matmul(normalized_q, normalized_k.transpose(-2, -1))
            * self.temperature
        ).softmax(dim=-1)
        attended = torch.matmul(attention, v)
        attended = rearrange(
            attended, "b head c (h w) -> b (head c) h w",
            head=self.num_heads, h=height, w=width,
        )
        output = self.output_projection(attended)
        if not return_aux:
            return output
        return output, {
            "q": q,
            "k": k,
            "v": v,
            "attention": attention,
        }


class TaskConditionedResponseScorer(nn.Module):
    def __init__(self, feature_dim, internal_dim=64):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.internal_dim = int(internal_dim)
        self.state_projection = nn.Conv2d(
            self.feature_dim, self.internal_dim, 1,
        )
        self.task_projection = nn.Conv2d(
            self.feature_dim, self.internal_dim, 1,
        )
        self.response_mapping = nn.Sequential(
            nn.Conv2d(self.internal_dim * 4, self.internal_dim, 1),
            nn.Conv2d(
                self.internal_dim, self.internal_dim, 3, padding=1,
                groups=self.internal_dim,
            ),
            nn.GELU(),
            # A shared output bias would cancel exactly in R_update - R_keep.
            nn.Conv2d(self.internal_dim, self.feature_dim, 1, bias=False),
        )

    def project_task(self, task_feedback):
        return self.task_projection(task_feedback)

    def score_projected(self, candidate_state, projected_task):
        projected_state = self.state_projection(candidate_state)
        interaction = projected_state * projected_task
        difference = (projected_state - projected_task).abs()
        comparison = torch.cat(
            (projected_state, projected_task, interaction, difference), dim=1,
        )
        return self.response_mapping(comparison)

    def forward(self, candidate_state, task_feedback):
        return self.score_projected(
            candidate_state, self.project_task(task_feedback),
        )


class ConditionalRestorationRefinementStage(nn.Module):
    def __init__(
        self, feature_dim, internal_dim=64, num_heads=4,
        correction_scale_initial=0.1,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.internal_dim = int(internal_dim)
        self.num_heads = int(num_heads)
        self.correction_scale_initial = float(correction_scale_initial)
        self.cross_attention = EfficientTaskCrossAttention(
            self.feature_dim, self.internal_dim, self.num_heads,
        )
        self.correction_mapper = nn.Sequential(
            nn.Conv2d(
                self.feature_dim, self.feature_dim, 3, padding=1,
                groups=self.feature_dim, bias=False,
            ),
            nn.GELU(),
            nn.Conv2d(self.feature_dim, self.feature_dim, 1),
        )
        self.correction_scale = nn.Parameter(
            torch.tensor(self.correction_scale_initial),
        )
        self.response_scorer = TaskConditionedResponseScorer(
            self.feature_dim, self.internal_dim,
        )
        self.routing_head = nn.Conv2d(self.feature_dim, self.feature_dim, 1)
        nn.init.normal_(self.routing_head.weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.routing_head.bias, -2.0)

    def forward(self, restoration_feature, trf_feedback, return_aux=False):
        if return_aux:
            attended_feature, attention_aux = self.cross_attention(
                restoration_feature, trf_feedback, return_aux=True,
            )
        else:
            attended_feature = self.cross_attention(
                restoration_feature, trf_feedback,
            )
            attention_aux = None

        correction_logits = self.correction_mapper(attended_feature)
        bounded_correction = torch.tanh(correction_logits)
        delta = self.correction_scale * bounded_correction
        keep_state = restoration_feature
        update_state = restoration_feature + delta

        projected_task = self.response_scorer.project_task(trf_feedback)
        response_keep = self.response_scorer.score_projected(
            keep_state, projected_task,
        )
        response_update = self.response_scorer.score_projected(
            update_state, projected_task,
        )
        contribution = response_update - response_keep
        routing_mask = torch.sigmoid(self.routing_head(contribution))
        updated_feature = restoration_feature + routing_mask * delta

        if not return_aux:
            return updated_feature
        return updated_feature, {
            "attention": attention_aux,
            "attended_feature": attended_feature,
            "delta_hat": correction_logits,
            "correction_logits": correction_logits,
            "bounded_correction": bounded_correction,
            "delta": delta,
            "keep_state": keep_state,
            "update_state": update_state,
            "projected_task": projected_task,
            "response_keep": response_keep,
            "response_update": response_update,
            "contribution": contribution,
            "routing_mask": routing_mask,
        }


class ConditionalRestorationRefinement(nn.Module):
    def __init__(
        self, feature_dims=(384, 192, 96), internal_dim=64, num_heads=4,
        correction_scale_initial=0.1,
    ):
        super().__init__()
        self.feature_dims = tuple(int(value) for value in feature_dims)
        self.internal_dim = int(internal_dim)
        self.num_heads = int(num_heads)
        self.correction_scale_initial = float(correction_scale_initial)
        self.stages = nn.ModuleList([
            ConditionalRestorationRefinementStage(
                feature_dim, self.internal_dim, self.num_heads,
                self.correction_scale_initial,
            )
            for feature_dim in self.feature_dims
        ])

    def forward_position(
        self, position, restoration_feature, trf_feedback, return_aux=False,
    ):
        if isinstance(position, str):
            position = CRR_POSITIONS.index(position)
        position = int(position)
        if not 0 <= position < len(self.stages):
            raise IndexError(f"CRR position must be in [0, 2], got {position}.")
        return self.stages[position](
            restoration_feature, trf_feedback, return_aux=return_aux,
        )

    def forward(self, restoration_features, trf_feedback, return_aux=False):
        updated, diagnostics = [], []
        for position, (feature, feedback) in enumerate(
            zip(restoration_features, trf_feedback),
        ):
            result = self.forward_position(
                position, feature, feedback, return_aux=return_aux,
            )
            if return_aux:
                current_updated, current_diagnostics = result
                updated.append(current_updated)
                diagnostics.append(current_diagnostics)
            else:
                updated.append(result)
        if not return_aux:
            return tuple(updated)
        return tuple(updated), {
            name: values for name, values in zip(CRR_POSITIONS, diagnostics)
        }

    def get_config(self):
        return {
            "internal_dim": self.internal_dim,
            "num_heads": self.num_heads,
            "correction_scale_initial": self.correction_scale_initial,
        }
