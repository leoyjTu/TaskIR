import argparse
import os

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.downstream_utils import build_downstream_models
from utils.task_feedback_utils import build_trfg
from utils.training_utils import (
    IterationTaskScheduler,
    TaskBatchProvider,
    build_iteration_scheduler,
    build_restore_models,
    build_train_loaders,
    cleanup_distributed,
    collect_task_lists,
    create_training_meters,
    is_main_process,
    load_training_checkpoint,
    rank_zero_print,
    reduce_training_meters,
    save_checkpoint,
    seed_everything,
    setup_distributed,
    train_one_iteration,
    update_training_meters,
    wrap_trfg_ddp,
    write_training_scalars,
)


RESUME_FILENAME = "latest_iter.pth"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train TaskIRNet with restoration pretraining and task-feedback finetuning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--stage", choices=("restore", "feedback"), default="restore",
        help="Training stage: restoration pretraining or task-feedback finetuning.",
    )
    for task, task_name in (("cls", "classification"), ("seg", "segmentation"), ("det", "detection")):
        parser.add_argument(
            f"--train_{task}_list", default=f"data_dir/lists/train_{task}.jsonl",
            help=f"Training JSONL list for {task_name}; pass an empty string to disable it.",
        )

    parser.add_argument("--patch_size", type=int, default=256, help="Square training crop size.")
    parser.add_argument("--batch_size", type=int, default=8, help="Global batch size across all ranks.")
    parser.add_argument("--num_workers", type=int, default=8, help="Global DataLoader worker budget.")
    parser.add_argument("--max_iters", type=int, default=None, help="Optimizer updates; defaults to 200k/100k by stage.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial AdamW learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="AdamW weight-decay coefficient.")
    parser.add_argument("--warmup_iters", type=int, default=5000, help="Linear learning-rate warmup updates.")
    parser.add_argument("--grad_clip_norm", type=float, default=1.0, help="Maximum global gradient norm.")

    parser.add_argument("--beta_rec", type=float, default=1.0, help="Image reconstruction loss weight.")
    parser.add_argument("--beta_deg", type=float, default=0.1, help="Stage-one degradation classification loss weight.")
    parser.add_argument("--beta_cls", type=float, default=0.01, help="Classification task-loss weight.")
    parser.add_argument("--beta_seg", type=float, default=0.01, help="Segmentation task-loss weight.")
    parser.add_argument("--beta_det", type=float, default=0.01, help="Detection task-loss weight.")
    parser.add_argument("--beta_gap", type=float, default=0.1, help="Task-gap distillation loss weight.")
    parser.add_argument("--beta_preserve", type=float, default=0.1, help="Stage-II reconstruction-preservation loss weight relative to Stage I.")
    parser.add_argument("--gap_teacher_momentum", type=float, default=0.999, help="EMA momentum for gap-target projectors.")
    parser.add_argument("--gap_direction_weight", type=float, default=0.1, help="Cosine-direction term inside the gap loss.")

    parser.add_argument("--feedback_dim", type=int, default=64, help="Channel dimension of three-level task feedback.")
    parser.add_argument("--trf_proto_dim", type=int, default=64, help="TRF prototype-space channel dimension.")
    parser.add_argument("--trf_num_prototypes", type=int, default=4, help="Number of shared TRF prototype anchors.")
    parser.add_argument("--trf_proto_temperature", type=float, default=0.1, help="TRF soft-assignment temperature.")
    parser.add_argument("--trf_sinkhorn_epsilon", type=float, default=0.05, help="TRF Sinkhorn affinity temperature.")
    parser.add_argument("--trf_sinkhorn_iters", type=int, default=30, help="TRF Sinkhorn normalization iterations.")
    parser.add_argument("--trf_route_temperature", type=float, default=1.0, help="TRF feedback-route softmax temperature.")
    parser.add_argument("--crr_dim", type=int, default=64, help="CRR internal interaction dimension.")
    parser.add_argument("--crr_heads", type=int, default=4, help="CRR efficient cross-attention head count.")
    parser.add_argument("--crr_correction_scale_init", type=float, default=0.1, help="Initial CRR correction amplitude eta.")
    parser.add_argument("--degradation_dim", type=int, default=128, help="Dimension of the learned degradation representation z_d.")

    parser.add_argument("--restore_ckpt", default="checkpoints/restore/latest_iter.pth", help="Stage-one checkpoint used to initialize feedback training.")
    parser.add_argument("--cls_ckpt", default="downstream/ResNet50/ckpt/resnet50-11ad3fa6.pth", help="Frozen ResNet-50 classification checkpoint.")
    parser.add_argument("--seg_ckpt", default="downstream/SegFormer/ckpt/segformer-b5-finetuned-cityscapes-1024-1024", help="Frozen SegFormer checkpoint directory.")
    parser.add_argument("--det_ckpt", default="downstream/YOLOv8/ckpt/yolov8n-voc2012.pt", help="Frozen 20-class YOLOv8 checkpoint.")
    parser.add_argument("--det_imgsz", type=int, default=640, help="YOLO training/evaluation image size.")
    parser.add_argument("--det_conf", type=float, default=0.001, help="YOLO evaluation confidence threshold.")
    parser.add_argument("--det_iou", type=float, default=0.7, help="YOLO NMS IoU threshold.")
    parser.add_argument("--ignore_index", type=int, default=255, help="Ignored Cityscapes train ID.")
    parser.add_argument("--max_det_boxes", type=int, default=100, help="Maximum detection boxes stored per image.")

    parser.add_argument("--ckpt_dir", default="", help="Checkpoint directory; inferred from stage when empty.")
    parser.add_argument("--log_dir", default="", help="TensorBoard directory; inferred from stage when empty.")
    parser.add_argument("--resume_ckpt", default="", help="Training checkpoint to resume; empty starts a new run.")
    parser.add_argument("--save_interval", type=int, default=10000, help="Checkpoint interval in optimizer updates.")
    parser.add_argument("--log_interval", type=int, default=200, help="TensorBoard logging interval.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument(
        "--local_rank", "--local-rank", type=int,
        default=int(os.environ.get("LOCAL_RANK", 0)), help=argparse.SUPPRESS,
    )
    opt = parser.parse_args()

    if opt.stage == "feedback" and not opt.restore_ckpt:
        parser.error("--restore_ckpt is required for feedback training.")
    if opt.crr_dim % opt.crr_heads:
        parser.error("--crr_dim must be divisible by --crr_heads.")
    if opt.max_iters is None:
        opt.max_iters = 100000 if opt.stage == "feedback" else 200000
    if opt.warmup_iters > opt.max_iters:
        parser.error("--warmup_iters cannot exceed --max_iters.")

    opt.cls_num_classes = 1000
    opt.seg_num_classes = 19
    opt.det_num_classes = 20
    opt.cls_pretrained = False
    opt.amp = False
    opt.use_feedback = opt.stage == "feedback"
    return opt


def main():
    opt = parse_args()
    distributed, rank, local_rank, world_size, device = setup_distributed()
    writer = None
    try:
        seed_everything(opt.seed, rank)
        opt.ckpt_dir = opt.ckpt_dir or f"checkpoints/{opt.stage}"
        opt.log_dir = opt.log_dir or f"logs/{opt.stage}"
        if is_main_process(rank):
            os.makedirs(opt.ckpt_dir, exist_ok=True)
            os.makedirs(opt.log_dir, exist_ok=True)
        if distributed:
            torch.distributed.barrier()

        train_lists = collect_task_lists(opt, train=True)
        enabled_tasks = list(train_lists)
        loaders, samplers, per_rank_batch, per_rank_workers = build_train_loaders(
            opt, train_lists, distributed, rank, world_size,
        )
        net, pre_net = build_restore_models(opt, device, distributed, local_rank)

        if opt.stage == "feedback":
            downstream = build_downstream_models(opt, enabled_tasks, device)
            trfg = build_trfg(opt, enabled_tasks, device, downstream)
            trfg = wrap_trfg_ddp(trfg, local_rank, distributed)
        else:
            downstream, trfg = {}, None

        trainable = [
            parameter for parameter in (
                list(net.parameters()) + ([] if trfg is None else list(trfg.parameters()))
            )
            if parameter.requires_grad
        ]
        if not trainable:
            raise RuntimeError("No trainable parameters were found.")
        optimizer = optim.AdamW(trainable, lr=opt.lr, weight_decay=opt.weight_decay)
        scheduler = build_iteration_scheduler(optimizer, opt.warmup_iters, opt.max_iters)
        task_scheduler = IterationTaskScheduler(opt.stage, loaders, opt.seed)
        batch_provider = TaskBatchProvider(loaders, samplers)

        if opt.stage == "restore":
            cycle_length = sum(len(loader) for loader in loaders.values())
            ratios = {name: len(loader) / cycle_length for name, loader in loaders.items()}
            policy = "data-proportional"
        else:
            cycle_length = len(loaders)
            ratios = {name: 1.0 / cycle_length for name in loaders}
            policy = "task-balanced"
        rank_zero_print(
            rank,
            f"Stage={opt.stage}; tasks={enabled_tasks}; sampling={policy}; ratios={ratios}; "
            f"DDP={distributed}; world_size={world_size}; global_batch={opt.batch_size}; "
            f"per_rank_batch={per_rank_batch}; per_rank_workers={per_rank_workers}; "
            f"trainable_parameters={sum(p.numel() for p in trainable):,}; FP32=True",
        )

        start_iteration = 0
        if opt.resume_ckpt:
            if not os.path.isfile(opt.resume_ckpt):
                raise FileNotFoundError(f"Resume checkpoint not found: {opt.resume_ckpt}")
            start_iteration = load_training_checkpoint(
                opt.resume_ckpt, net, trfg, optimizer, scheduler,
                task_scheduler, batch_provider, opt, world_size, device,
            )
            rank_zero_print(rank, f"Resumed from iteration {start_iteration}.")
        if start_iteration > opt.max_iters:
            raise RuntimeError("Checkpoint iteration exceeds max_iters.")

        writer = SummaryWriter(opt.log_dir, purge_step=start_iteration) if is_main_process(rank) else None
        meters = create_training_meters(opt.stage)
        progress = tqdm(
            total=opt.max_iters, initial=start_iteration,
            desc=f"Train {opt.stage}", disable=not is_main_process(rank),
            miniters=opt.log_interval,
        )

        for iteration_index in range(start_iteration, opt.max_iters):
            iteration = iteration_index + 1
            task_name = task_scheduler.next_task()
            batch = batch_provider.next_batch(task_name)
            current_lr = optimizer.param_groups[0]["lr"]
            logs = train_one_iteration(
                opt, batch, task_name, net, pre_net,
                downstream, trfg, optimizer, device,
            )
            scheduler.step()
            update_training_meters(meters, logs)

            if is_main_process(rank):
                progress.update(1)
                postfix = {
                    "task": task_name, "loss": f"{logs['loss_total']:.4f}",
                    "rec": f"{logs['loss_rec']:.4f}", "lr": f"{current_lr:.3e}",
                    "grad": f"{logs['grad_norm']:.3f}",
                }
                if "loss_deg" in logs:
                    postfix["deg"] = f"{logs['loss_deg']:.4f}"
                    postfix["deg_acc"] = f"{logs['deg_accuracy']:.3f}"
                progress.set_postfix(postfix, refresh=False)

            if iteration % opt.log_interval == 0 or iteration == opt.max_iters:
                global_metrics = reduce_training_meters(meters, device, distributed)
                if opt.stage == "feedback":
                    global_metrics["lr"] = current_lr
                if is_main_process(rank):
                    write_training_scalars(writer, global_metrics, iteration)
                meters = create_training_meters(opt.stage)

            if iteration % opt.save_interval == 0 or iteration == opt.max_iters:
                if is_main_process(rank):
                    for filename in (RESUME_FILENAME, f"iter_{iteration:07d}.pth"):
                        save_checkpoint(
                            os.path.join(opt.ckpt_dir, filename), net, trfg,
                            optimizer, scheduler, task_scheduler, batch_provider,
                            iteration, opt, world_size,
                        )
                if distributed:
                    torch.distributed.barrier()

        if is_main_process(rank):
            progress.close()
    finally:
        if writer is not None:
            writer.close()
        cleanup_distributed()


if __name__ == "__main__":
    main()
