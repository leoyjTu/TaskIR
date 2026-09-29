import csv
import gc
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

try:
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
except ImportError:
    MeanAveragePrecision = None

from utils.downstream_utils import normalize_for_task
from utils.task_dataset_utils import COMMON_DEGRADATIONS, TASK2ID, TaskDrivenValDataset
from utils.task_loss_utils import build_det_targets
from utils.val_utils import AverageMeter, compute_psnr_ssim


SUMMARY_FIELDS = [
    "timestamp", "method", "task", "degradation", "num_images", "psnr", "ssim",
    "acc_top1", "acc_top5", "miou", "dice", "map50", "map50_95",
]
MANIFEST_FIELDS = [
    "task", "sample_index", "degradation", "lq", "gt", "restored",
]
CLS_FIELDS = [
    "task", "sample_index", "degradation", "lq", "gt", "restored", "psnr", "ssim",
    "label", "top1", "top1_score", "top1_correct", "top5", "top5_scores",
    "heatmap_path", "overlay_path",
]
SEG_FIELDS = [
    "task", "sample_index", "degradation", "lq", "gt", "restored", "psnr", "ssim",
    "miou", "dice", "mask_path", "color_path", "overlay_path",
]
DET_FIELDS = [
    "task", "sample_index", "degradation", "lq", "gt", "restored", "psnr", "ssim",
    "num_regular_gt", "num_difficult_gt", "num_predictions", "num_visualized",
    "boxes", "scores", "labels", "vis_path",
]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
SEG_NUM_CLASSES = 19
_BILINEAR = Image.Resampling.BILINEAR
_NEAREST = Image.Resampling.NEAREST


def _resolve_record_path(path, root=""):
    if path is None or str(path) in ("", "None"):
        return None
    candidate = Path(os.path.expanduser(str(path)))
    if not candidate.is_absolute():
        candidate = Path(root) / candidate
    return str(candidate.resolve())


def _record_degradation(record):
    return str(record.get("deg", record.get("degradation", record.get("corruption", "")))).lower()


def _read_rgb(path):
    with Image.open(path) as image:
        return image.convert("RGB")


class TaskIRResultDataset(Dataset):
    def __init__(
        self, list_file, task, results_root=None, restored_root=None, dataset_root="",
        ignore_index=255, max_det_boxes=100,
    ):
        self.task = task
        self.dataset_root = dataset_root
        self.base_dataset = TaskDrivenValDataset(
            list_file=list_file,
            root=dataset_root,
            default_task=task,
            ignore_index=ignore_index,
            max_det_boxes=max_det_boxes,
        )
        if restored_root is None:
            restored_root = Path(results_root) / task / "restored"
        self.restored_root = Path(restored_root).expanduser().resolve()

        self.items = []
        matched_paths = set()
        actual_paths = {
            path.resolve() for path in self.restored_root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        }
        actual_by_identity = defaultdict(list)
        for path in actual_paths:
            relative_path = path.relative_to(self.restored_root)
            if len(relative_path.parts) < 2:
                continue
            degradation = relative_path.parts[0].lower()
            match = re.fullmatch(r"\d{6}_(.+)", path.stem)
            identity_stem = (match.group(1) if match else path.stem).lower()
            actual_by_identity[(degradation, identity_stem)].append(path)

        for index, record in enumerate(self.base_dataset.records):
            degradation = _record_degradation(record)
            gt_path = _resolve_record_path(record.get("gt"), dataset_root)
            restored_path = (
                self.restored_root / degradation / f"{index:06d}_{Path(gt_path).stem}.png"
            ).resolve()
            if not restored_path.is_file():
                identity = (degradation.lower(), Path(gt_path).stem.lower())
                candidates = actual_by_identity.get(identity, [])
                if len(candidates) > 1:
                    raise RuntimeError(
                        f"{task}: multiple restored images match record {index} by "
                        f"degradation and GT stem {identity}: "
                        f"{[str(path) for path in candidates[:10]]}"
                    )
                if len(candidates) == 1:
                    restored_path = candidates[0]

            if not restored_path.is_file():
                raise FileNotFoundError(
                    f"Missing TaskIR result for {task} record {index}; "
                    f"degradation={degradation}, GT stem={Path(gt_path).stem}, "
                    f"exact path={restored_path}"
                )
            if restored_path in matched_paths:
                raise RuntimeError(
                    f"{task}: restored image matched more than one JSONL record: "
                    f"{restored_path}"
                )
            self.items.append((index, str(restored_path), degradation))
            matched_paths.add(restored_path)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        base_index, restored_path, degradation = self.items[index]
        sample = self.base_dataset[base_index]
        record = self.base_dataset.records[base_index]
        restored_image = _read_rgb(restored_path)
        gt_height, gt_width = sample["gt"].shape[-2:]
        if restored_image.size != (gt_width, gt_height):
            raise RuntimeError(
                f"Restored/GT size mismatch: {restored_image.size} vs {(gt_width, gt_height)}; "
                f"restored={restored_path}"
            )
        sample.update({
            "restored": to_tensor(restored_image),
            "restored_path": restored_path,
            "result_name": Path(restored_path).name,
            "degradation_name": degradation,
            "sample_index": base_index,
            "lq_path": _resolve_record_path(record.get("lq"), self.dataset_root) or "",
            "gt_path": _resolve_record_path(record.get("gt"), self.dataset_root) or "",
        })
        return sample

    def manifest_rows(self):
        rows = []
        for base_index, restored_path, degradation in self.items:
            record = self.base_dataset.records[base_index]
            rows.append({
                "task": self.task,
                "sample_index": base_index,
                "degradation": degradation,
                "lq": _resolve_record_path(record.get("lq"), self.dataset_root) or "",
                "gt": _resolve_record_path(record.get("gt"), self.dataset_root) or "",
                "restored": restored_path,
            })
        return rows


def _update_seg_confusion(confusion, prediction, target, ignore_index=255):
    prediction = prediction.detach().reshape(-1).cpu().long()
    target = target.detach().reshape(-1).cpu().long()
    valid = (
        (target != ignore_index) & (target >= 0) & (target < SEG_NUM_CLASSES)
        & (prediction >= 0) & (prediction < SEG_NUM_CLASSES)
    )
    if valid.any():
        confusion += torch.bincount(
            SEG_NUM_CLASSES * target[valid] + prediction[valid],
            minlength=SEG_NUM_CLASSES ** 2,
        ).reshape(SEG_NUM_CLASSES, SEG_NUM_CLASSES).double()


def _miou_dice(confusion):
    intersection = confusion.diag()
    target_area = confusion.sum(1)
    prediction_area = confusion.sum(0)
    union = target_area + prediction_area - intersection
    valid_iou = union > 0
    denominator = target_area + prediction_area
    valid_dice = denominator > 0
    miou = (intersection[valid_iou] / union[valid_iou]).mean().item() if valid_iou.any() else None
    dice = (
        (2.0 * intersection[valid_dice] / denominator[valid_dice]).mean().item()
        if valid_dice.any() else None
    )
    return miou, dice


def _make_map_metric():
    if MeanAveragePrecision is None:
        raise ImportError("Detection evaluation requires torchmetrics and pycocotools.")
    return MeanAveragePrecision(box_format="xyxy", iou_type="bbox")


class MetricState:
    def __init__(self, task):
        self.task = task
        self.psnr = AverageMeter()
        self.ssim = AverageMeter()
        self.num_images = 0
        self.top1_correct = 0
        self.top5_correct = 0
        self.cls_total = 0
        self.seg_confusion = (
            torch.zeros(SEG_NUM_CLASSES, SEG_NUM_CLASSES, dtype=torch.float64)
            if task == "seg" else None
        )
        self.map_metric = _make_map_metric() if task == "det" else None

    def update_restoration(self, psnr, ssim):
        self.psnr.update(psnr)
        self.ssim.update(ssim)
        self.num_images += 1

    def update_classification(self, topk, label):
        self.top1_correct += int(topk[0] == label)
        self.top5_correct += int(label in topk)
        self.cls_total += 1

    def update_segmentation(self, prediction, target, ignore_index):
        _update_seg_confusion(self.seg_confusion, prediction, target, ignore_index)

    def update_detection(self, prediction, target):
        self.map_metric.update([prediction], [target])

    def summary(self, method, task, degradation, timestamp):
        acc_top1 = self.top1_correct / self.cls_total if self.cls_total else None
        acc_top5 = self.top5_correct / self.cls_total if self.cls_total else None
        miou, dice = _miou_dice(self.seg_confusion) if self.seg_confusion is not None else (None, None)
        map50, map50_95 = None, None
        if self.map_metric is not None:
            result = self.map_metric.compute()
            map50 = float(result["map_50"].item())
            map50_95 = float(result["map"].item())
        return {
            "timestamp": timestamp,
            "method": method,
            "task": task,
            "degradation": degradation,
            "num_images": self.num_images,
            "psnr": self.psnr.avg,
            "ssim": self.ssim.avg,
            "acc_top1": acc_top1,
            "acc_top5": acc_top5,
            "miou": miou,
            "dice": dice,
            "map50": map50,
            "map50_95": map50_95,
        }


class TaskMetricAccumulator:
    def __init__(self, task):
        self.task = task
        self.states = {
            name: MetricState(task) for name in (*COMMON_DEGRADATIONS, "overall")
        }

    def update(self, degradation, psnr, ssim, task_result, ignore_index):
        for name in (degradation, "overall"):
            state = self.states[name]
            state.update_restoration(psnr, ssim)
            if self.task == "cls":
                state.update_classification(task_result["topk"], task_result["label"])
            elif self.task == "seg":
                state.update_segmentation(
                    task_result["prediction"], task_result["target"], ignore_index,
                )
            else:
                state.update_detection(task_result["prediction"], task_result["target"])

    def summaries(self, method):
        timestamp = datetime.now().isoformat(timespec="seconds")
        return [
            self.states[name].summary(method, self.task, name, timestamp)
            for name in (*COMMON_DEGRADATIONS, "overall")
        ]


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _classification_result(model, restored, make_heatmap):
    model = _unwrap_model(model)
    heatmap = None
    with torch.no_grad():
        logits = model(normalize_for_task(restored))
    if make_heatmap:
        activation = {}

        def capture(_module, _inputs, output):
            activation["value"] = output

        handle = model.layer4.register_forward_hook(capture)
        try:
            with torch.enable_grad():
                model_input = normalize_for_task(restored.detach()).requires_grad_(True)
                visualization_logits = model(model_input)
                class_index = visualization_logits.argmax(1)
                features = activation.get("value")
                gradient = torch.autograd.grad(
                    visualization_logits[0, class_index[0]], features,
                )[0]
                weights = gradient.mean((2, 3), keepdim=True)
                heatmap = (weights * features).sum(1, keepdim=True).relu()
                heatmap = F.interpolate(
                    heatmap, restored.shape[-2:], mode="bilinear", align_corners=False,
                )[0, 0]
                heatmap = heatmap - heatmap.min()
                heatmap = (heatmap / heatmap.max().clamp_min(1e-8)).detach().cpu()
        finally:
            handle.remove()

    probabilities = logits.detach().softmax(1).cpu()
    top_scores, topk = probabilities.topk(min(5, probabilities.shape[1]), dim=1)
    return topk[0].tolist(), top_scores[0].tolist(), heatmap


def _segmentation_result(model, restored, target):
    with torch.no_grad():
        output = model(normalize_for_task(restored))
        logits = output["out"] if isinstance(output, dict) else output
        if logits.shape[-2:] != target.shape[-2:]:
            logits = F.interpolate(
                logits, target.shape[-2:], mode="bilinear", align_corners=False,
            )
        prediction = logits.argmax(1)
    return prediction[0].cpu(), target[0].cpu()


def _detection_result(model, restored, batch, args, device):
    predictions = model.predict(restored)
    det_valid = batch["task_id"] == TASK2ID["det"]
    targets = build_det_targets(
        batch, det_valid, args, device, include_difficult=True,
    )
    prediction = {
        "boxes": predictions[0]["boxes"].detach().float().cpu(),
        "scores": predictions[0]["scores"].detach().float().cpu(),
        "labels": predictions[0]["labels"].detach().long().cpu(),
    }
    target = {
        "boxes": targets[0]["boxes"].detach().float().cpu(),
        "labels": targets[0]["labels"].detach().long().cpu(),
        "iscrowd": targets[0]["iscrowd"].detach().long().cpu(),
    }
    return prediction, target


def _move_batch(batch, device):
    return {
        key: value.to(device, non_blocking=device.type == "cuda")
        if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _is_cuda_oom(error, device):
    return device.type == "cuda" and "out of memory" in str(error).lower()


def _clear_cuda_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run_task_model(task, model, batch, args, device):
    restored = batch["restored"].clamp(0, 1)
    if task == "cls":
        topk, scores, heatmap = _classification_result(
            model, restored, args.save_visualizations,
        )
        label = int(batch["cls_label"][0].item())
        return {"topk": topk, "scores": scores, "label": label, "heatmap": heatmap}
    if task == "seg":
        prediction, target = _segmentation_result(model, restored, batch["seg_mask"].long())
        return {"prediction": prediction, "target": target}
    prediction, target = _detection_result(model, restored, batch, args, device)
    return {"prediction": prediction, "target": target}


def _tensor_to_image(tensor):
    array = tensor.detach().clamp(0, 1).mul(255).byte().cpu().permute(1, 2, 0).numpy()
    return Image.fromarray(array, mode="RGB")


def _heatmap_image(heatmap):
    values = heatmap.numpy()
    red = np.clip(1.5 * values, 0, 1)
    green = np.clip(1.5 - np.abs(values - 0.55) * 3.0, 0, 1)
    blue = np.clip(1.0 - 1.5 * values, 0, 1)
    return Image.fromarray((np.stack((red, green, blue), -1) * 255).astype(np.uint8))


def _segmentation_palette():
    colors = np.zeros((256, 3), dtype=np.uint8)
    colors[:19] = np.asarray((
        (128, 64, 128),  # road
        (244, 35, 232),  # sidewalk
        (70, 70, 70),  # building
        (102, 102, 156),  # wall
        (190, 153, 153),  # fence
        (153, 153, 153),  # pole
        (250, 170, 30),  # traffic light
        (220, 220, 0),  # traffic sign
        (107, 142, 35),  # vegetation
        (152, 251, 152),  # terrain
        (70, 130, 180),  # sky
        (220, 20, 60),  # person
        (255, 0, 0),  # rider
        (0, 0, 142),  # car
        (0, 0, 70),  # truck
        (0, 60, 100),  # bus
        (0, 80, 100),  # train
        (0, 0, 230),  # motorcycle
        (119, 11, 32),  # bicycle
    ), dtype=np.uint8)
    return colors


SEGMENTATION_PALETTE = _segmentation_palette()


def _colorize_segmentation(prediction):
    values = prediction.clamp(0, 255).numpy()
    return Image.fromarray(SEGMENTATION_PALETTE[values], mode="RGB")


def _filter_detection(prediction, confidence, max_boxes):
    keep = prediction["scores"] >= confidence
    boxes = prediction["boxes"][keep]
    scores = prediction["scores"][keep]
    labels = prediction["labels"][keep]
    order = scores.argsort(descending=True)
    if max_boxes > 0:
        order = order[:max_boxes]
    return {"boxes": boxes[order], "scores": scores[order], "labels": labels[order]}


def _class_name(names, label):
    if isinstance(names, dict):
        return str(names.get(label, label))
    if isinstance(names, (list, tuple)) and 0 <= label < len(names):
        return str(names[label])
    return str(label)


def _draw_detections(image, prediction, names):
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    for box, score, label in zip(
        prediction["boxes"].tolist(),
        prediction["scores"].tolist(),
        prediction["labels"].tolist(),
    ):
        x1, y1, x2, y2 = [float(value) for value in box]
        text = f"{_class_name(names, int(label))} {float(score):.2f}"
        color = (255, 64, 64)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        text_box = draw.textbbox((x1, y1), text)
        text_h = text_box[3] - text_box[1]
        text_w = text_box[2] - text_box[0]
        label_y = max(0, y1 - text_h - 4)
        draw.rectangle(
            [x1, label_y, x1 + text_w + 4, label_y + text_h + 4], fill=color,
        )
        draw.text((x1 + 2, label_y + 2), text, fill=(255, 255, 255))
    return canvas


def _visual_path(root, task, kind, degradation, result_name):
    path = Path(root) / task / kind / degradation / result_name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _image_seg_metrics(prediction, target, ignore_index):
    confusion = torch.zeros(SEG_NUM_CLASSES, SEG_NUM_CLASSES, dtype=torch.float64)
    _update_seg_confusion(confusion, prediction, target, ignore_index)
    return _miou_dice(confusion)


def _format_numbers(values, precision=6):
    return " ".join(f"{float(value):.{precision}f}" for value in values)


def _format_boxes(boxes):
    return ";".join(",".join(f"{float(value):.2f}" for value in box) for box in boxes)


def _save_outputs(args, task, batch, task_result, psnr, ssim, model):
    sample_index = int(batch["sample_index"][0])
    degradation = batch["degradation_name"][0]
    result_name = batch["result_name"][0]
    restored_path = batch["restored_path"][0]
    base = {
        "task": task,
        "sample_index": sample_index,
        "degradation": degradation,
        "lq": batch["lq_path"][0],
        "gt": batch["gt_path"][0],
        "restored": restored_path,
        "psnr": psnr,
        "ssim": ssim,
    }
    restored_image = _tensor_to_image(batch["restored"][0])

    if task == "cls":
        topk, scores = task_result["topk"], task_result["scores"]
        heatmap_path = overlay_path = ""
        if args.save_visualizations:
            heatmap_path = _visual_path(
                args.visualization_root, task, "heatmaps", degradation, result_name,
            )
            overlay_path = _visual_path(
                args.visualization_root, task, "overlays", degradation, result_name,
            )
            heatmap_image = _heatmap_image(task_result["heatmap"])
            overlay = Image.blend(
                restored_image, heatmap_image.resize(restored_image.size, _BILINEAR), 0.45,
            )
            heatmap_image.save(heatmap_path)
            overlay.save(overlay_path)
        return {
            **base,
            "label": task_result["label"],
            "top1": topk[0],
            "top1_score": scores[0],
            "top1_correct": int(topk[0] == task_result["label"]),
            "top5": " ".join(str(value) for value in topk),
            "top5_scores": _format_numbers(scores),
            "heatmap_path": str(heatmap_path),
            "overlay_path": str(overlay_path),
        }

    if task == "seg":
        miou, dice = _image_seg_metrics(
            task_result["prediction"], task_result["target"], args.ignore_index,
        )
        mask_path = color_path = overlay_path = ""
        if args.save_visualizations:
            mask_path = _visual_path(
                args.visualization_root, task, "masks", degradation, result_name,
            )
            color_path = _visual_path(
                args.visualization_root, task, "colors", degradation, result_name,
            )
            overlay_path = _visual_path(
                args.visualization_root, task, "overlays", degradation, result_name,
            )
            mask = Image.fromarray(task_result["prediction"].byte().numpy(), mode="L")
            color = _colorize_segmentation(task_result["prediction"])
            overlay = Image.blend(restored_image, color.resize(restored_image.size, _NEAREST), 0.45)
            mask.save(mask_path)
            color.save(color_path)
            overlay.save(overlay_path)
        return {
            **base, "miou": miou, "dice": dice,
            "mask_path": str(mask_path), "color_path": str(color_path),
            "overlay_path": str(overlay_path),
        }

    filtered = _filter_detection(
        task_result["prediction"], args.det_vis_conf, args.det_vis_max_boxes,
    )
    vis_path = ""
    if args.save_visualizations:
        vis_path = _visual_path(
            args.visualization_root, task, "detections", degradation, result_name,
        )
        names = getattr(_unwrap_model(model), "names", None)
        _draw_detections(restored_image, filtered, names).save(vis_path)
    return {
        **base,
        "num_regular_gt": int((task_result["target"]["iscrowd"] == 0).sum().item()),
        "num_difficult_gt": int((task_result["target"]["iscrowd"] == 1).sum().item()),
        "num_predictions": len(task_result["prediction"]["boxes"]),
        "num_visualized": len(filtered["boxes"]),
        "boxes": _format_boxes(filtered["boxes"].tolist()),
        "scores": _format_numbers(filtered["scores"].tolist()),
        "labels": " ".join(str(int(value)) for value in filtered["labels"].tolist()),
        "vis_path": str(vis_path),
    }


def evaluate_task(args, task, dataset, model, device):
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    accumulator = TaskMetricAccumulator(task)
    output_rows = []

    for cpu_batch in tqdm(loader, desc=f"Eval-{task}"):
        cuda_oom = False
        try:
            batch = _move_batch(cpu_batch, device)
            task_result = _run_task_model(task, model, batch, args, device)
        except RuntimeError as error:
            if not _is_cuda_oom(error, device):
                raise
            cuda_oom = True
        if cuda_oom:
            print(f"CUDA OOM during {task} evaluation; switching remaining evaluation to CPU.")
            model.to("cpu")
            device = torch.device("cpu")
            _clear_cuda_cache()
            batch = _move_batch(cpu_batch, device)
            task_result = _run_task_model(task, model, batch, args, device)

        restored = batch["restored"].clamp(0, 1)
        gt = batch["gt"].clamp(0, 1)
        psnr, ssim, _ = compute_psnr_ssim(restored, gt)
        degradation = batch["degradation_name"][0]
        accumulator.update(degradation, psnr, ssim, task_result, args.ignore_index)
        output_rows.append(
            _save_outputs(args, task, batch, task_result, psnr, ssim, model)
        )

    summaries = accumulator.summaries(args.method_name)
    overall = summaries[-1]
    message = f"[{task} overall] PSNR={overall['psnr']:.2f}, SSIM={overall['ssim']:.4f}"
    if task == "cls":
        message += f", Top-1={overall['acc_top1'] * 100:.2f}, Top-5={overall['acc_top5'] * 100:.2f}"
    elif task == "seg":
        message += f", mIoU={overall['miou'] * 100:.2f}, Dice={overall['dice'] * 100:.2f}"
    else:
        message += f", mAP50={overall['map50'] * 100:.2f}, mAP50-95={overall['map50_95'] * 100:.2f}"
    print(message)
    return summaries, output_rows, device


def write_csv(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: "" if row.get(name) is None else row.get(name) for name in fieldnames})


def write_task_rows(output_dir, task, rows):
    fields = {"cls": CLS_FIELDS, "seg": SEG_FIELDS, "det": DET_FIELDS}[task]
    write_csv(Path(output_dir) / f"{task}_per_image.csv", fields, rows)
