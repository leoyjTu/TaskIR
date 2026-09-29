import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from net.modules import LayerNorm
from utils.downstream_utils import (
    HIERARCHICAL_FEATURE_LEVELS,
    extract_hierarchical_task_features,
)


LEVELS = (1, 2, 3)
TASK_NAMES = ("cls", "seg", "det")
TASK_TO_INDEX = {name: index for index, name in enumerate(TASK_NAMES)}


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _resize_feature(feature, size):
    if feature.shape[-2:] == size:
        return feature
    return F.interpolate(feature, size=size, mode="bilinear", align_corners=False)


def _clone_frozen(module):
    teacher = copy.deepcopy(module).eval()
    teacher.requires_grad_(False)
    return teacher


@torch.no_grad()
def _ema_update(student, teacher, momentum):
    momentum = float(momentum)
    if not 0.0 <= momentum <= 1.0:
        raise ValueError("EMA momentum must be in [0, 1].")
    for student_parameter, teacher_parameter in zip(student.parameters(), teacher.parameters()):
        teacher_parameter.mul_(momentum).add_(student_parameter, alpha=1.0 - momentum)
    for student_buffer, teacher_buffer in zip(student.buffers(), teacher.buffers()):
        teacher_buffer.copy_(student_buffer)


def _group_count(channels, max_groups=8):
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def _normalize_gap(gap, eps=1e-6):
    scale = gap.detach().abs().mean((1, 2, 3), keepdim=True).clamp_min(eps)
    return torch.tanh(gap / scale)


class ProjectionBank(nn.Module):
    def __init__(self, input_channels, feedback_dim=64):
        super().__init__()
        self.feedback_dim = int(feedback_dim)
        self.input_channels = {
            str(task): tuple(int(value) for value in channels)
            for task, channels in input_channels.items()
        }
        if not self.input_channels:
            raise ValueError("ProjectionBank requires at least one task.")

        self.projections = nn.ModuleDict({
            task: nn.ModuleDict({
                level_name: nn.Conv2d(channels[index], self.feedback_dim, 1)
                for index, level_name in enumerate(HIERARCHICAL_FEATURE_LEVELS)
            })
            for task, channels in self.input_channels.items()
        })

    def forward(self, task, features, target_sizes):
        task = str(task).lower()
        projected = {}
        for level, level_name in zip(LEVELS, HIERARCHICAL_FEATURE_LEVELS):
            feature = features[level_name]
            projected[level] = _resize_feature(
                self.projections[task][level_name](feature), target_sizes[level],
            )
        return projected


class TaskConditionedDynamicAdapter(nn.Module):
    def __init__(self, dim=64, condition_dim=64, layer_norm_type="WithBias"):
        super().__init__()
        self.dim = int(dim)
        self.condition_dim = int(condition_dim)
        self.norm = LayerNorm(self.dim, layer_norm_type)
        self.input_mapping = nn.Conv2d(self.dim, self.dim, 1, bias=False)
        self.dynamic_branches = nn.ModuleList([
            nn.Conv2d(
                self.dim, self.dim, 3, padding=dilation, dilation=dilation,
                groups=self.dim, bias=False,
            )
            for dilation in (1, 2, 3)
        ])
        self.weight_mlp = nn.Sequential(
            nn.Linear(self.dim + self.condition_dim, self.dim),
            nn.GELU(),
            nn.Linear(self.dim, 3 * self.dim),
        )
        self.output_mapping = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(self.dim, self.dim, 1),
        )
        self.lambda_scale = nn.Parameter(torch.tensor(1e-3))

    def forward(self, feature, condition, return_details=False):
        mapped = self.input_mapping(self.norm(feature))
        branch_features = tuple(branch(mapped) for branch in self.dynamic_branches)
        global_feature = feature.mean(dim=(-2, -1))
        weights = self.weight_mlp(torch.cat((global_feature, condition), dim=1))
        weights = weights.reshape(feature.shape[0], 3, self.dim).softmax(dim=1)
        aggregated = sum(
            weights[:, index, :, None, None] * branch_feature
            for index, branch_feature in enumerate(branch_features)
        )
        adaptation = self.output_mapping(aggregated)
        adapted = feature + self.lambda_scale * adaptation
        if not return_details:
            return adapted
        return adapted, {
            "branches": branch_features,
            "omega": weights,
            "aggregated": aggregated,
            "adaptation": adaptation,
        }


class TaskGapPredictor(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.norm = nn.GroupNorm(_group_count(dim), dim)
        self.body = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, bias=False), nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1, bias=False), nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1),
        )
        nn.init.normal_(self.body[-1].weight, std=1e-3)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, adapted_feature):
        return torch.tanh(self.body(self.norm(adapted_feature)))


class GapGate(nn.Module):
    def __init__(self, dim=64, init_bias=-2.0):
        super().__init__()
        self.norm = nn.GroupNorm(_group_count(dim), dim)
        self.body = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False), nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1),
        )
        nn.init.normal_(self.body[-1].weight, std=1e-3)
        nn.init.constant_(self.body[-1].bias, init_bias)

    def forward(self, adapted_feature):
        return torch.sigmoid(self.body(self.norm(adapted_feature)))


class _TRFGFrontEnd(nn.Module):
    def __init__(self, input_channels, feedback_dim=64):
        super().__init__()
        self.feedback_dim = int(feedback_dim)
        self.projection_bank = ProjectionBank(input_channels, self.feedback_dim)
        self.task_level_embedding = nn.Embedding(
            len(TASK_NAMES) * len(LEVELS), self.feedback_dim,
        )
        self.tcdas = nn.ModuleList([
            TaskConditionedDynamicAdapter(self.feedback_dim, self.feedback_dim)
            for _ in LEVELS
        ])

    def _condition(self, task, level_index, batch_size, device):
        task_index = TASK_TO_INDEX[task]
        embedding_index = task_index * len(LEVELS) + level_index
        indices = torch.full(
            (batch_size,), embedding_index, dtype=torch.long, device=device,
        )
        return self.task_level_embedding(indices)

    def forward(self, task, raw_features, target_sizes, return_details=False):
        task = str(task).lower()
        projected = self.projection_bank(task, raw_features, target_sizes)
        adapted, details = {}, {}
        for level_index, (level, tcda) in enumerate(zip(LEVELS, self.tcdas)):
            condition = self._condition(
                task, level_index, projected[level].shape[0], projected[level].device,
            )
            if return_details:
                adapted[level], tcda_details = tcda(
                    projected[level], condition, return_details=True,
                )
                details[level] = {
                    "condition": condition,
                    "projected": projected[level],
                    **tcda_details,
                }
            else:
                adapted[level] = tcda(projected[level], condition)
        return (adapted, details) if return_details else adapted


class TaskToRestorationFeedbackGenerator(nn.Module):
    def __init__(self, input_channels, feedback_dim=64):
        super().__init__()
        self.feedback_dim = int(feedback_dim)
        self.student_frontend = _TRFGFrontEnd(input_channels, self.feedback_dim)
        self.teacher_frontend = _clone_frozen(self.student_frontend)
        self.task_gap_predictors = nn.ModuleList([
            TaskGapPredictor(self.feedback_dim) for _ in LEVELS
        ])
        self.gap_gates = nn.ModuleList([
            GapGate(self.feedback_dim) for _ in LEVELS
        ])

    @property
    def projection_bank(self):
        return self.student_frontend.projection_bank

    @property
    def task_level_embedding(self):
        return self.student_frontend.task_level_embedding

    @property
    def tcdas(self):
        return self.student_frontend.tcdas

    def train(self, mode=True):
        super().train(mode)
        self.teacher_frontend.eval()
        return self

    @torch.no_grad()
    def update_teacher(self, momentum):
        _ema_update(self.student_frontend, self.teacher_frontend, momentum)

    @torch.no_grad()
    def forward_teacher(self, task, raw_features, target_sizes):
        self.teacher_frontend.eval()
        return self.teacher_frontend(task, raw_features, target_sizes)

    def forward(self, task, raw_features, target_sizes, return_details=False):
        if return_details:
            adapted, frontend_details = self.student_frontend(
                task, raw_features, target_sizes, return_details=True,
            )
        else:
            adapted = self.student_frontend(task, raw_features, target_sizes)
            frontend_details = None

        feedback, predicted_gap, gates = {}, {}, {}
        for level, predictor, gate_module in zip(
            LEVELS, self.task_gap_predictors, self.gap_gates,
        ):
            predicted_gap[level] = predictor(adapted[level])
            gates[level] = gate_module(adapted[level])
            feedback[level] = adapted[level] + gates[level] * predicted_gap[level]
        if not return_details:
            return feedback, predicted_gap, adapted
        return feedback, predicted_gap, adapted, {
            "frontend": frontend_details,
            "gates": gates,
        }


def build_trfg(opt, enabled_tasks, device, downstream=None):
    enabled_tasks = tuple(enabled_tasks)
    input_channels = {}
    if "cls" in enabled_tasks:
        input_channels["cls"] = (512, 1024, 2048)
    if "seg" in enabled_tasks:
        input_channels["seg"] = (128, 320, 512)
    if "det" in enabled_tasks:
        input_channels["det"] = tuple(
            int(value) for value in _unwrap_model(downstream["det"]).get_feature_channels()
        )
    unknown = set(enabled_tasks) - set(TASK_NAMES)
    if unknown:
        raise ValueError(f"Unknown TRFG tasks: {sorted(unknown)}")
    return TaskToRestorationFeedbackGenerator(
        input_channels, getattr(opt, "feedback_dim", 64),
    ).to(device)


def forward_pre_net(pre_net, lq):
    pre_net = _unwrap_model(pre_net)
    pre_net.eval()
    pre_net.requires_grad_(False)
    with torch.no_grad():
        return pre_net(lq)


def _feedback_sizes(height, width):
    padded_height = int(height) + (-int(height)) % 8
    padded_width = int(width) + (-int(width)) % 8
    return {
        1: (max(padded_height // 2, 1), max(padded_width // 2, 1)),
        2: (max(padded_height // 4, 1), max(padded_width // 4, 1)),
        3: (max(padded_height // 8, 1), max(padded_width // 8, 1)),
    }


def _empty_feedback(batch_size, channels, sizes, reference):
    return {
        level: reference.new_zeros(batch_size, channels, *sizes[level])
        for level in LEVELS
    }


def _insert_feedback(feedback, task_feedback, indices):
    for level in LEVELS:
        feature = task_feedback[level].to(feedback[level])
        feedback[level] = feedback[level].index_copy(0, indices, feature)


def compute_gap_distillation_loss(predicted_gap, target_gap, direction_weight=0.1):
    if target_gap is None:
        return next(iter(predicted_gap.values())).new_zeros(())
    regression_losses, direction_losses = [], []
    for level in LEVELS:
        prediction, target = predicted_gap[level], target_gap[level]
        regression_losses.append(F.smooth_l1_loss(prediction, target))
        prediction, target = prediction.flatten(1), target.flatten(1)
        valid = target.norm(dim=1) > 1e-6
        if valid.any():
            cosine = F.cosine_similarity(prediction[valid], target[valid], dim=1)
            direction_losses.append((1.0 - cosine).mean())
    regression = torch.stack(regression_losses).mean()
    direction = torch.stack(direction_losses).mean() if direction_losses else regression.new_zeros(())
    return regression + float(direction_weight) * direction


def build_feedback_feature(
    lq, task_id, pre_net, downstream, trfg, opt, TASK2ID, gt=None,
):
    restored_0 = forward_pre_net(pre_net, lq)
    clean = None if gt is None else gt.detach().clamp(0, 1)
    task_id = task_id.to(lq.device)

    sizes = _feedback_sizes(*lq.shape[-2:])
    feedback = _empty_feedback(
        lq.shape[0], getattr(opt, "feedback_dim", 64), sizes, lq,
    )
    teacher_momentum = getattr(opt, "gap_teacher_momentum", 0.999)
    direction_weight = getattr(opt, "gap_direction_weight", 0.1)
    gap_losses = []
    generator = _unwrap_model(trfg)
    if clean is not None:
        generator.update_teacher(teacher_momentum)

    for task_name in TASK_NAMES:
        valid = task_id == TASK2ID[task_name]
        if not valid.any():
            continue
        indices = valid.nonzero(as_tuple=False).flatten()
        with torch.no_grad():
            current_features = extract_hierarchical_task_features(
                task_name, downstream[task_name], restored_0[valid],
            )
            current_features = {
                name: feature.detach() for name, feature in current_features.items()
            }
            clean_features = None
            if clean is not None:
                clean_features = extract_hierarchical_task_features(
                    task_name, downstream[task_name], clean[valid],
                )
                clean_features = {
                    name: feature.detach() for name, feature in clean_features.items()
                }

        task_feedback, predicted_gap, _ = trfg(task_name, current_features, sizes)
        _insert_feedback(feedback, task_feedback, indices)
        if clean_features is not None:
            with torch.no_grad():
                current_teacher = generator.forward_teacher(
                    task_name, current_features, sizes,
                )
                clean_teacher = generator.forward_teacher(
                    task_name, clean_features, sizes,
                )
                target_gap = {
                    level: _normalize_gap(
                        clean_teacher[level] - current_teacher[level]
                    ).detach()
                    for level in LEVELS
                }
            gap_losses.append(
                compute_gap_distillation_loss(predicted_gap, target_gap, direction_weight)
            )

    loss_gap = torch.stack(gap_losses).mean() if gap_losses else lq.new_zeros(())
    return feedback, restored_0, loss_gap
