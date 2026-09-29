import math
import os
import random
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from net.model import TaskIRNet
from utils.task_dataset_utils import NUM_DEGRADATIONS, TASK2ID, TaskDrivenTrainDataset
from utils.task_feedback_utils import build_feedback_feature
from utils.task_loss_utils import compute_task_loss
from utils.val_utils import AverageMeter


FEEDBACK_SCHEME = "task_to_restoration_feedback_trf_crr"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def setup_distributed():
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if not distributed:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return False, 0, 0, 1, device
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl", init_method="env://", timeout=timedelta(hours=4),
    )
    return True, dist.get_rank(), local_rank, dist.get_world_size(), torch.device("cuda", local_rank)


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank):
    return rank == 0


def rank_zero_print(rank, message):
    if is_main_process(rank):
        print(message)


def seed_everything(seed, rank=0):
    process_seed = int(seed) + int(rank)
    random.seed(process_seed)
    np.random.seed(process_seed)
    torch.manual_seed(process_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(process_seed)


def unwrap_model(model):
    return model.module if isinstance(model, (DDP, nn.DataParallel)) else model


def wrap_trfg_ddp(trfg, local_rank, distributed):
    if trfg is None or not distributed:
        return trfg
    return DDP(
        trfg, device_ids=[local_rank], output_device=local_rank,
        broadcast_buffers=False, find_unused_parameters=True,
    )


def optimizer_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def to_device(batch, device):
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def reduce_average_meter(meter, device, distributed):
    values = torch.tensor([meter.sum, meter.count], dtype=torch.float64, device=device)
    if distributed:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return float(values[0].item() / max(values[1].item(), 1.0))


def collect_task_lists(opt, train=True):
    prefix = "train" if train else "val"
    task_lists = {}
    for task in ("cls", "seg", "det"):
        path = getattr(opt, f"{prefix}_{task}_list")
        if not path:
            continue
        task_lists[task] = path
    return task_lists


def build_train_loaders(opt, task_lists, distributed, rank, world_size):
    if opt.batch_size % world_size:
        raise ValueError("Global batch_size must be divisible by world_size.")
    batch_size = opt.batch_size // world_size
    workers = opt.num_workers // world_size if distributed else opt.num_workers
    loaders, samplers = {}, {}

    for task, list_file in task_lists.items():
        dataset = TaskDrivenTrainDataset(
            list_file=list_file, root="", default_task=task,
            resolution=opt.patch_size, is_train=True, online_corrupt=False,
            ignore_index=opt.ignore_index, max_det_boxes=opt.max_det_boxes,
        )

        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True,
            seed=opt.seed, drop_last=True,
        )
        generator = torch.Generator().manual_seed(opt.seed + TASK2ID[task] * 1009 + rank)
        loaders[task] = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, sampler=sampler,
            num_workers=workers, pin_memory=torch.cuda.is_available(), drop_last=True,
            persistent_workers=False, generator=generator,
        )
        if len(loaders[task]) == 0:
            raise RuntimeError(f"Training loader for task '{task}' contains no complete batch.")
        samplers[task] = sampler
    return loaders, samplers, batch_size, workers


def set_restore_stage_trainable(net):
    net.requires_grad_(True)
    for name, parameter in net.named_parameters():
        if (
            name.startswith("trf.")
            or name.startswith("crr.")
        ):
            parameter.requires_grad = False


def set_feedback_stage_trainable(net):
    net.requires_grad_(False)
    for name, parameter in net.named_parameters():
        if (
            name.startswith("trf.")
            or name.startswith("crr.")
        ):
            parameter.requires_grad = True


def attach_trf_from_options(net, opt):
    return net.attach_trf(
        prototype_dim=opt.trf_proto_dim,
        num_prototypes=opt.trf_num_prototypes,
        prototype_temperature=opt.trf_proto_temperature,
        sinkhorn_epsilon=opt.trf_sinkhorn_epsilon,
        sinkhorn_iterations=opt.trf_sinkhorn_iters,
        route_temperature=opt.trf_route_temperature,
    )


def attach_crr_from_options(net, opt):
    return net.attach_crr(
        internal_dim=opt.crr_dim,
        num_heads=opt.crr_heads,
        correction_scale_initial=opt.crr_correction_scale_init,
    )


def build_taskir_net(opt, device):
    return TaskIRNet(
        feedback_dim=opt.feedback_dim,
        degradation_dim=opt.degradation_dim,
        num_degradations=NUM_DEGRADATIONS,
    ).to(device)


def load_stage1_checkpoint(model, checkpoint_path):
    candidate = Path(os.path.expanduser(str(checkpoint_path)))
    if not candidate.is_absolute():
        project_candidate = PROJECT_ROOT / candidate
        if project_candidate.exists():
            candidate = project_candidate
    candidate = candidate.resolve()
    checkpoint = torch.load(candidate, map_location="cpu", weights_only=True)
    return model.load_state_dict(checkpoint["state_dict"], strict=True)


def build_restore_models(opt, device, distributed, local_rank):
    net = build_taskir_net(opt, device)
    pre_net = None
    if opt.stage == "restore":
        set_restore_stage_trainable(net)
    else:
        load_stage1_checkpoint(net, opt.restore_ckpt)
        attach_trf_from_options(net, opt)
        attach_crr_from_options(net, opt)
        set_feedback_stage_trainable(net)
        pre_net = build_taskir_net(opt, device)
        load_stage1_checkpoint(pre_net, opt.restore_ckpt)
        pre_net.eval().requires_grad_(False)

    if distributed:
        net = DDP(
            net, device_ids=[local_rank], output_device=local_rank,
            broadcast_buffers=False, find_unused_parameters=False,
        )
    return net, pre_net


def save_checkpoint(
    path, net, trfg, optimizer, scheduler, task_scheduler,
    batch_provider, iteration, opt, world_size,
):
    state = {
        "state_dict": unwrap_model(net).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "task_scheduler": task_scheduler.state_dict(),
        "batch_provider": batch_provider.state_dict(),
        "iteration": int(iteration),
        "max_iters": int(opt.max_iters),
        "warmup_iters": int(opt.warmup_iters),
        "stage": opt.stage,
        "world_size": int(world_size),
        "global_batch_size": int(opt.batch_size),
        "num_workers": int(opt.num_workers),
    }
    if opt.stage == "feedback":
        state["feedback_scheme"] = FEEDBACK_SCHEME
        state["trf_config"] = unwrap_model(net).get_trf_config()
        state["crr_config"] = unwrap_model(net).get_crr_config()
    if trfg is not None:
        state["trfg"] = unwrap_model(trfg).state_dict()
    torch.save(state, path)


def _reconstruct_data_progress(task_scheduler, batch_provider, completed_iterations):
    counts = {task: 0 for task in batch_provider.train_loaders}
    for _ in range(completed_iterations):
        counts[task_scheduler.next_task()] += 1
    batch_provider.load_state_dict({"batches_consumed": counts})


def load_training_checkpoint(
    path, net, trfg, optimizer, scheduler, task_scheduler,
    batch_provider, opt, world_size, device,
):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    unwrap_model(net).load_state_dict(checkpoint["state_dict"], strict=True)
    if trfg is not None:
        unwrap_model(trfg).load_state_dict(checkpoint["trfg"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    optimizer_to_device(optimizer, device)
    scheduler.load_state_dict(checkpoint["scheduler"])

    completed = int(checkpoint["iteration"])
    if "task_scheduler" in checkpoint and "batch_provider" in checkpoint:
        task_scheduler.load_state_dict(checkpoint["task_scheduler"])
        batch_provider.load_state_dict(checkpoint["batch_provider"])
    else:
        _reconstruct_data_progress(task_scheduler, batch_provider, completed)
    return completed


def build_restore_task_schedule(loaders, cycle_index, seed):
    schedule = [task for task, loader in loaders.items() for _ in range(len(loader))]
    random.Random(seed + cycle_index).shuffle(schedule)
    return schedule


def build_feedback_task_schedule(loaders, cycle_index, seed):
    schedule = list(loaders)
    random.Random(seed + cycle_index).shuffle(schedule)
    return schedule


class IterationTaskScheduler:
    def __init__(self, stage, train_loaders, seed):
        self.stage, self.train_loaders, self.seed = stage, train_loaders, int(seed)
        self.cycle_index, self.schedule, self.offset = 0, [], 0

    def _refresh(self):
        builder = build_restore_task_schedule if self.stage == "restore" else build_feedback_task_schedule
        self.schedule = builder(self.train_loaders, self.cycle_index, self.seed)
        self.cycle_index += 1
        self.offset = 0

    def next_task(self):
        if self.offset >= len(self.schedule):
            self._refresh()
        task = self.schedule[self.offset]
        self.offset += 1
        return task

    def state_dict(self):
        return {
            "stage": self.stage, "seed": self.seed,
            "cycle_index": self.cycle_index,
            "schedule": list(self.schedule), "offset": self.offset,
        }

    def load_state_dict(self, state):
        schedule = list(state.get("schedule", []))
        self.cycle_index = int(state.get("cycle_index", 0))
        self.schedule = schedule
        self.offset = int(state.get("offset", 0))


class TaskBatchProvider:
    def __init__(self, train_loaders, train_samplers):
        self.train_loaders, self.train_samplers = train_loaders, train_samplers
        self.batches_consumed = {task: 0 for task in train_loaders}
        self.base_worker_seeds = {
            task: loader.generator.initial_seed() for task, loader in train_loaders.items()
        }
        for sampler in train_samplers.values():
            if sampler is not None:
                sampler.set_epoch(0)
        self.task_iters = {task: None for task in train_loaders}

    def _make_iterator(self, task, cycle):
        sampler = self.train_samplers.get(task)
        if sampler is not None:
            sampler.set_epoch(cycle)
        loader = self.train_loaders[task]
        loader.generator.manual_seed(self.base_worker_seeds[task] + cycle * 1000003)
        return iter(loader)

    def next_batch(self, task):
        if self.task_iters[task] is None:
            cycle = self.batches_consumed[task] // len(self.train_loaders[task])
            self.task_iters[task] = self._make_iterator(task, cycle)
        try:
            batch = next(self.task_iters[task])
        except StopIteration:
            cycle = self.batches_consumed[task] // len(self.train_loaders[task])
            self.task_iters[task] = self._make_iterator(task, cycle)
            batch = next(self.task_iters[task])
        self.batches_consumed[task] += 1
        return batch

    def state_dict(self):
        return {"batches_consumed": dict(self.batches_consumed)}

    def load_state_dict(self, state):
        counts = state.get("batches_consumed", {})
        self.task_iters = {}
        for task, loader in self.train_loaders.items():
            consumed = int(counts[task])
            cycle, offset = divmod(consumed, len(loader))
            iterator = self._make_iterator(task, cycle)
            for _ in range(offset):
                next(iterator)
            self.task_iters[task] = iterator
            self.batches_consumed[task] = consumed


def build_iteration_scheduler(optimizer, warmup_iters, max_iters):
    def lr_lambda(iteration):
        if warmup_iters and iteration < warmup_iters:
            return max((iteration + 1) / warmup_iters, 1e-8)
        progress = (iteration - warmup_iters) / max(max_iters - warmup_iters, 1)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def create_training_meters(stage):
    names = {
        "restore": (
            "loss_total", "loss_rec", "loss_deg", "deg_accuracy", "grad_norm",
        ),
        "feedback": (
            "loss_total", "loss_rec", "loss_task",
            "loss_cls", "loss_seg", "loss_det", "loss_gap",
            "loss_preserve", "preserve_active", "grad_norm",
        ),
    }[stage]
    return {name: AverageMeter() for name in names}


def update_training_meters(meters, logs):
    batch_size = logs["batch_size"]
    for name, meter in meters.items():
        if name in logs:
            meter.update(logs[name], batch_size)


def reduce_training_meters(meters, device, distributed):
    return {
        name: reduce_average_meter(meter, device, distributed)
        if meter.count else 0.0
        for name, meter in meters.items()
    }


def write_training_scalars(writer, metrics, iteration):
    if writer is None:
        return
    for name, value in metrics.items():
        writer.add_scalar(f"train/{name}", value, iteration)


def _clip_gradients(optimizer, max_norm):
    parameters = [
        parameter for group in optimizer.param_groups
        for parameter in group["params"] if parameter.grad is not None
    ]
    if not parameters or max_norm <= 0:
        return 0.0
    return float(torch.nn.utils.clip_grad_norm_(parameters, max_norm).item())


def _optimize(loss, optimizer, max_norm):
    if not torch.isfinite(loss.detach()):
        raise FloatingPointError(f"Training loss is not finite: {loss.detach().item()}.")
    loss.backward()
    grad_norm = _clip_gradients(optimizer, max_norm)
    if not math.isfinite(grad_norm):
        raise FloatingPointError(f"Gradient norm is not finite: {grad_norm}.")
    optimizer.step()
    return grad_norm


def train_restore_iteration(opt, batch, net, optimizer, device):
    net.train()
    batch = to_device(batch, device)
    lq, gt = batch["lq"], batch["gt"]
    degradation_id = batch["degradation_id"].long()
    optimizer.zero_grad(set_to_none=True)
    restored, auxiliary = net(lq, return_aux=True)
    degradation_logits = auxiliary["degradation_logits"]
    loss_rec = F.l1_loss(restored, gt)
    loss_deg = F.cross_entropy(degradation_logits, degradation_id)
    loss = opt.beta_rec * loss_rec + opt.beta_deg * loss_deg
    grad_norm = _optimize(loss, optimizer, opt.grad_clip_norm)
    deg_accuracy = (
        degradation_logits.detach().argmax(dim=1) == degradation_id
    ).float().mean()
    logs = {
        "batch_size": lq.shape[0], "loss_total": loss.item(),
        "loss_rec": loss_rec.item(), "loss_deg": loss_deg.item(),
        "deg_accuracy": deg_accuracy.item(), "grad_norm": grad_norm,
    }
    return logs


def train_feedback_iteration(
    opt, batch, task_name, net, pre_net, downstream,
    trfg, optimizer, device,
):
    net.train()
    trfg.train()
    pre_net.eval()
    for model in downstream.values():
        model.eval()
    batch = to_device(batch, device)
    lq, gt = batch["lq"], batch["gt"]
    task_id = batch["task_id"]
    expected_task_id = TASK2ID[task_name]
    if task_id.numel() == 0 or not torch.all(task_id == expected_task_id):
        actual_ids = torch.unique(task_id).detach().cpu().tolist()
        raise RuntimeError(
            f"Stage-II scheduler selected {task_name!r} (task_id={expected_task_id}), "
            f"but the batch contains task_id values {actual_ids}."
        )
    optimizer.zero_grad(set_to_none=True)

    feedback, restored_0, loss_gap = build_feedback_feature(
        lq, task_id, pre_net, downstream, trfg, opt, TASK2ID, gt,
    )
    restored = net(lq, feedback_feat=feedback)
    loss_rec = F.l1_loss(restored, gt)
    loss_task, task_logs = compute_task_loss(
        restored.clamp(0, 1), batch, downstream, opt, TASK2ID, device,
    )

    error_stage2 = F.l1_loss(restored, gt, reduction="none").flatten(1).mean(1)
    with torch.no_grad():
        error_stage1 = F.l1_loss(restored_0, gt, reduction="none").flatten(1).mean(1)
    preserve_violation = error_stage2 - error_stage1
    loss_preserve = F.relu(preserve_violation).mean()

    loss = (
        opt.beta_rec * loss_rec + loss_task + opt.beta_gap * loss_gap
        + opt.beta_preserve * loss_preserve
    )
    grad_norm = _optimize(loss, optimizer, opt.grad_clip_norm)

    logs = {
        "batch_size": lq.shape[0], "loss_total": loss.item(),
        "loss_rec": loss_rec.item(), "loss_task": loss_task.item(),
        "loss_gap": loss_gap.item(),
        "loss_preserve": loss_preserve.item(),
        "preserve_active": (preserve_violation > 0).float().mean().item(),
        "grad_norm": grad_norm,
    }
    logs[f"loss_{task_name}"] = task_logs[f"loss_{task_name}"].item()
    return logs


def train_one_iteration(
    opt, batch, task_name, net, pre_net, downstream,
    trfg, optimizer, device,
):
    if opt.stage == "restore":
        return train_restore_iteration(opt, batch, net, optimizer, device)
    return train_feedback_iteration(
        opt, batch, task_name, net, pre_net,
        downstream, trfg, optimizer, device,
    )
