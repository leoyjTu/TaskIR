import argparse
import gc
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from net.model import TaskIRNet
from utils.downstream_utils import build_downstream_models
from utils.image_io import save_image_tensor
from utils.task_dataset_utils import (
    COMMON_DEGRADATIONS,
    TASK2ID,
    TaskDrivenValDataset,
)
from utils.task_feedback_utils import build_feedback_feature, build_trfg


TASKS = ("cls", "seg", "det")
FEEDBACK_SCHEME = "task_to_restoration_feedback_trf_crr"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Restore TaskIR test images with the final feedback-stage model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cuda", type=int, default=0, help="Preferred CUDA device; CUDA OOM automatically falls back to CPU.")
    parser.add_argument("--restore_ckpt", default="checkpoints/restore/latest_iter.pth", help="Frozen stage-one restoration checkpoint.")
    parser.add_argument("--feedback_ckpt", default="checkpoints/feedback/latest_iter.pth", help="Final stage-two checkpoint containing TaskIR, TRFG, TRF and CRR.")
    parser.add_argument("--cls_list", default="data_dir/lists/val_cls.jsonl", help="ImageNet-1K test JSONL list.")
    parser.add_argument("--seg_list", default="data_dir/lists/val_seg.jsonl", help="Cityscapes test JSONL list.")
    parser.add_argument("--det_list", default="data_dir/lists/val_det.jsonl", help="VOC2012 test JSONL list.")
    parser.add_argument("--task", choices=("all", *TASKS), default="all", help="Task to restore; 'all' strictly requires all three lists.")
    parser.add_argument("--output_path", default="test_results", help="Directory for restored images.")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader worker count.")
    parser.add_argument("--tile", type=int, default=512, help="Inference tile size. Use 0 to disable tiling.")
    parser.add_argument("--tile_overlap", type=int, default=64, help="Overlap between neighboring inference tiles.")
    parser.add_argument("--tile_threshold", type=int, default=1024, help="Tile images whose maximum side exceeds this value. Use 0 to tile every image.")

    parser.add_argument("--cls_ckpt", default="downstream/ResNet50/ckpt/resnet50-11ad3fa6.pth", help="Frozen ImageNet-1K ResNet-50 checkpoint.")
    parser.add_argument("--seg_ckpt", default="downstream/SegFormer/ckpt/segformer-b5-finetuned-cityscapes-1024-1024", help="Frozen Cityscapes SegFormer-B5 directory.")
    parser.add_argument("--det_ckpt", default="downstream/YOLOv8/ckpt/yolov8n-voc2012.pt", help="Frozen 20-class VOC2012 YOLOv8 checkpoint.")
    parser.add_argument("--det_imgsz", type=int, default=640, help="YOLOv8 input size used to extract feedback.")
    parser.add_argument("--det_conf", type=float, default=0.001, help="YOLOv8 prediction confidence threshold.")
    parser.add_argument("--det_iou", type=float, default=0.7, help="YOLOv8 NMS IoU threshold.")

    parser.add_argument("--feedback_dim", type=int, default=64, help="Feedback channel dimension used during training.")
    parser.add_argument("--degradation_dim", type=int, default=128, help="Stage-I degradation-representation dimension used during training.")
    parser.add_argument("--trf_proto_dim", type=int, default=64, help="TRF prototype-space channel dimension.")
    parser.add_argument("--trf_num_prototypes", type=int, default=4, help="Number of shared TRF prototype anchors.")
    parser.add_argument("--trf_proto_temperature", type=float, default=0.1, help="TRF soft-assignment temperature.")
    parser.add_argument("--trf_sinkhorn_epsilon", type=float, default=0.05, help="TRF Sinkhorn affinity temperature.")
    parser.add_argument("--trf_sinkhorn_iters", type=int, default=30, help="TRF Sinkhorn normalization iterations.")
    parser.add_argument("--trf_route_temperature", type=float, default=1.0, help="TRF feedback-route softmax temperature.")
    parser.add_argument("--crr_dim", type=int, default=64, help="CRR internal interaction dimension.")
    parser.add_argument("--crr_heads", type=int, default=4, help="CRR efficient cross-attention head count.")
    parser.add_argument("--crr_correction_scale_init", type=float, default=0.1, help="Initial CRR correction amplitude eta.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    args = parser.parse_args()

    args.cls_num_classes = 1000
    args.seg_num_classes = 19
    args.det_num_classes = 20
    args.ignore_index = 255
    args.max_det_boxes = 100
    if args.crr_dim % args.crr_heads:
        parser.error("--crr_dim must be divisible by --crr_heads.")
    if args.tile > 0 and args.tile_overlap >= args.tile:
        parser.error("--tile_overlap must be smaller than --tile.")
    return args


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def enabled_tasks(args):
    paths = {task: getattr(args, f"{task}_list") for task in TASKS}
    selected = TASKS if args.task == "all" else (args.task,)
    missing = [f"{task}: {paths[task]}" for task in selected if not os.path.isfile(paths[task])]
    if missing:
        raise FileNotFoundError("Missing required test lists: " + "; ".join(missing))
    return selected


def load_checkpoint(path, expected_stage):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("stage") != expected_stage:
        raise RuntimeError(f"Expected a new-format '{expected_stage}' training checkpoint: {path}")

    result = {"state_dict": checkpoint["state_dict"]}
    if expected_stage == "feedback":
        if checkpoint.get("feedback_scheme") != FEEDBACK_SCHEME:
            raise RuntimeError(
                f"Expected feedback scheme '{FEEDBACK_SCHEME}', got "
                f"{checkpoint.get('feedback_scheme')!r}. Retrain the feedback stage."
            )
        result["trfg"] = checkpoint["trfg"]
        result["trf_config"] = checkpoint["trf_config"]
        result["crr_config"] = checkpoint["crr_config"]
    return result


def requested_trf_config(args):
    return {
        "prototype_dim": args.trf_proto_dim,
        "num_prototypes": args.trf_num_prototypes,
        "prototype_temperature": args.trf_proto_temperature,
        "sinkhorn_epsilon": args.trf_sinkhorn_epsilon,
        "sinkhorn_iterations": args.trf_sinkhorn_iters,
        "route_temperature": args.trf_route_temperature,
    }


def requested_crr_config(args):
    return {
        "internal_dim": args.crr_dim,
        "num_heads": args.crr_heads,
        "correction_scale_initial": args.crr_correction_scale_init,
    }


def build_taskir(
    args, state_dict, attach_stage2=False, trf_config=None, crr_config=None,
):
    model = TaskIRNet(
        feedback_dim=args.feedback_dim,
        degradation_dim=args.degradation_dim,
    )
    if attach_stage2:
        expected_trf = requested_trf_config(args)
        if trf_config != expected_trf:
            raise RuntimeError(
                f"TRF test configuration mismatch: expected {expected_trf}, "
                f"got {trf_config}."
            )
        expected_crr = requested_crr_config(args)
        if crr_config != expected_crr:
            raise RuntimeError(
                f"CRR test configuration mismatch: expected {expected_crr}, "
                f"got {crr_config}."
            )
        model.attach_trf(**trf_config)
        model.attach_crr(**crr_config)
    model.load_state_dict(state_dict, strict=True)
    return model.float().eval().requires_grad_(False)


def load_task_trfg_state(trfg, trfg_state, task):
    target_state = trfg.state_dict()
    selected = {}
    for key, target in target_state.items():
        source = trfg_state.get(key)
        if source is None:
            raise KeyError(f"Feedback checkpoint has no TRFG parameter {key!r} for {task}.")
        if source.shape != target.shape:
            raise RuntimeError(
                f"TRFG parameter {key!r} has shape {tuple(source.shape)}, "
                f"expected {tuple(target.shape)} for {task}."
            )
        selected[key] = source
    trfg.load_state_dict(selected, strict=True)
    return trfg.float().eval().requires_grad_(False)


def is_cuda_oom(error, device):
    return device.type == "cuda" and "out of memory" in str(error).lower()


def clear_cuda_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def move_to_cpu(net, pre_net, downstream=None, trfg=None):
    cpu = torch.device("cpu")
    net.to(cpu)
    pre_net.to(cpu)
    if downstream is not None:
        for model in downstream.values():
            model.to(cpu)
    if trfg is not None:
        trfg.to(cpu)
    clear_cuda_cache()
    return cpu


def place_taskir_models(net, pre_net, preferred_device):
    if preferred_device.type != "cuda":
        return preferred_device
    try:
        pre_net.to(preferred_device)
        net.to(preferred_device)
        return preferred_device
    except RuntimeError as error:
        if not is_cuda_oom(error, preferred_device):
            raise
        print("CUDA OOM while loading TaskIR models; switching the full test to CPU.")
        return move_to_cpu(net, pre_net)


def build_task_components(args, task, trfg_state, device, net, pre_net):
    downstream = build_downstream_models(args, (task,), device="cpu")
    trfg = build_trfg(args, (task,), "cpu", downstream)
    trfg = load_task_trfg_state(trfg, trfg_state, task)
    if device.type != "cuda":
        return downstream, trfg, device

    try:
        for model in downstream.values():
            model.to(device)
        trfg.to(device)
        return downstream, trfg, device
    except RuntimeError as error:
        if not is_cuda_oom(error, device):
            raise
        print(f"CUDA OOM while loading {task} feedback networks; switching to CPU.")
        device = move_to_cpu(net, pre_net, downstream, trfg)
        return downstream, trfg, device


def build_dataset(args, task):
    return TaskDrivenValDataset(
        list_file=getattr(args, f"{task}_list"),
        default_task=task,
        ignore_index=args.ignore_index,
        max_det_boxes=args.max_det_boxes,
    )


def restore_image(args, lq, task_id, net, pre_net, downstream, trfg):
    feedback, _, _ = build_feedback_feature(
        lq, task_id, pre_net, downstream, trfg, args, TASK2ID, gt=None,
    )
    return net(lq, feedback_feat=feedback)


def tile_forward(tensor, forward_fn, tile, overlap):
    if tile <= 0:
        return forward_fn(tensor)

    _, _, height, width = tensor.shape
    if height <= tile and width <= tile:
        return forward_fn(tensor)

    stride = tile - overlap

    def starts(size):
        if size <= tile:
            return [0]
        values = list(range(0, size - tile + 1, stride))
        last = size - tile
        if values[-1] != last:
            values.append(last)
        return values

    output = None
    weight = tensor.new_zeros((1, 1, height, width))
    for y in starts(height):
        for x in starts(width):
            patch = tensor[:, :, y:y + tile, x:x + tile]
            restored_patch = forward_fn(patch)
            if output is None:
                output = tensor.new_zeros(
                    (tensor.shape[0], restored_patch.shape[1], height, width)
                )
            output[:, :, y:y + tile, x:x + tile] += restored_patch.to(output.dtype)
            weight[:, :, y:y + tile, x:x + tile] += 1
    return output / weight.clamp_min(1)


def infer_batch(args, batch, device, net, pre_net, downstream, trfg):
    lq = batch["lq"].to(device, non_blocking=device.type == "cuda")
    task_id = batch["task_id"].to(device, non_blocking=device.type == "cuda")
    degradation_id = batch["degradation_id"].to(
        device, non_blocking=device.type == "cuda",
    )
    use_tile = (
        args.tile
        if args.tile > 0
        and (
            args.tile_threshold <= 0
            or max(lq.shape[-2:]) > args.tile_threshold
        )
        else 0
    )

    def forward_fn(input_tensor):
        return restore_image(
            args, input_tensor, task_id, net, pre_net, downstream, trfg,
        ).clamp_(0, 1)

    restored = tile_forward(
        lq, forward_fn, use_tile, args.tile_overlap,
    )
    if restored.shape != lq.shape:
        raise RuntimeError(f"Restored/LQ shape mismatch: {restored.shape} vs {lq.shape}")
    return restored.cpu(), int(degradation_id.item())


def retry_batch_on_cpu(args, batch, net, pre_net, downstream, trfg):
    device = move_to_cpu(net, pre_net, downstream, trfg)
    print("Retrying the current tiled image on CPU; this may be slow.")
    restored, degradation_id = infer_batch(
        args, batch, device, net, pre_net, downstream, trfg,
    )
    return restored, degradation_id, device


def save_name(batch_name, sample_index):
    name = batch_name[0] if isinstance(batch_name, (list, tuple)) else batch_name
    return f"{sample_index:06d}_{Path(str(name)).stem}.png"


@torch.inference_mode()
def test_dataset(args, task, dataset, device, net, pre_net, downstream, trfg):
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    output_root = Path(args.output_path) / task / "restored"
    output_root.mkdir(parents=True, exist_ok=True)

    for sample_index, batch in enumerate(tqdm(loader, desc=f"Test-{task}")):
        cuda_oom = False
        try:
            restored, degradation_id = infer_batch(
                args, batch, device, net, pre_net, downstream, trfg,
            )
        except RuntimeError as error:
            if not is_cuda_oom(error, device):
                raise
            cuda_oom = True
        if cuda_oom:
            clear_cuda_cache()
            restored, degradation_id, device = retry_batch_on_cpu(
                args, batch, net, pre_net, downstream, trfg,
            )

        degradation_name = COMMON_DEGRADATIONS[degradation_id]
        output_dir = output_root / degradation_name
        output_dir.mkdir(parents=True, exist_ok=True)
        save_image_tensor(restored, str(output_dir / save_name(batch["name"], sample_index)))

    print(f"[{task}] Saved {len(dataset)} restored images to {output_root}")
    return device


def main():
    args = parse_args()
    seed_everything(args.seed)
    tasks = enabled_tasks(args)
    Path(args.output_path).mkdir(parents=True, exist_ok=True)

    restore_checkpoint = load_checkpoint(args.restore_ckpt, "restore")
    feedback_checkpoint = load_checkpoint(args.feedback_ckpt, "feedback")
    pre_net = build_taskir(args, restore_checkpoint["state_dict"])
    net = build_taskir(
        args, feedback_checkpoint["state_dict"], attach_stage2=True,
        trf_config=feedback_checkpoint["trf_config"],
        crr_config=feedback_checkpoint["crr_config"],
    )
    del restore_checkpoint

    preferred_device = (
        torch.device(f"cuda:{args.cuda}")
        if torch.cuda.is_available() else torch.device("cpu")
    )
    if preferred_device.type == "cuda":
        torch.cuda.set_device(preferred_device)
    device = place_taskir_models(net, pre_net, preferred_device)
    print(f"Tasks: {', '.join(tasks)}")
    print(
        f"Inference device: {device}; FP32; tile={args.tile}, "
        f"overlap={args.tile_overlap}, threshold={args.tile_threshold}; no resizing."
    )

    trfg_state = feedback_checkpoint["trfg"]
    del feedback_checkpoint
    for task in tasks:
        downstream, trfg, device = build_task_components(
            args, task, trfg_state, device, net, pre_net,
        )
        dataset = build_dataset(args, task)
        device = test_dataset(
            args, task, dataset, device, net, pre_net, downstream, trfg,
        )
        del dataset, downstream, trfg
        clear_cuda_cache()


if __name__ == "__main__":
    main()
