import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor


TASK2ID = {"cls": 0, "seg": 1, "det": 2}
ID2TASK = {value: key for key, value in TASK2ID.items()}
COMMON_DEGRADATIONS = [
    "gaussian_noise", "shot_noise", "defocus_blur", "motion_blur",
    "snow", "fog", "brightness", "jpeg_compression",
]
DEG2ID = {name: index for index, name in enumerate(COMMON_DEGRADATIONS)}
NUM_DEGRADATIONS = len(COMMON_DEGRADATIONS)
CLS_NUM_CLASSES = 1000
SEG_NUM_CLASSES = 19
DET_NUM_CLASSES = 20
_BICUBIC = Image.Resampling.BICUBIC
_NEAREST = Image.Resampling.NEAREST
_FLIP_LEFT_RIGHT = Image.Transpose.FLIP_LEFT_RIGHT


def _require_offline(online_corrupt):
    if online_corrupt:
        raise ValueError(
            "Online corruption is not supported. Provide pre-generated LQ images "
            "and reference their paths in the dataset JSONL file."
        )


def _resolve_path(path, root=""):
    if path is None or str(path) in ("", "None"):
        return None
    candidate = Path(os.path.expanduser(str(path)))
    if not candidate.is_absolute():
        candidate = Path(root) / candidate
    return str(candidate.resolve())


def _read_rgb(path):
    with Image.open(path) as image:
        return image.convert("RGB")


def _read_mask(path):
    with Image.open(path) as image:
        return image.copy()


def _read_jsonl(path, default_task=None):
    records = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            if default_task is not None:
                record.setdefault("task", default_task)
            records.append(record)
    return records


def load_task_records(list_file, default_task=None):
    list_files = [list_file] if isinstance(list_file, (str, os.PathLike)) else list(list_file)
    if default_task is None or isinstance(default_task, str):
        default_tasks = [default_task] * len(list_files)
    else:
        default_tasks = list(default_task)
    if len(default_tasks) != len(list_files):
        raise ValueError("default_task must match the number of JSONL files.")

    records = []
    for path, task in zip(list_files, default_tasks):
        path = str(path)
        records.extend(_read_jsonl(path, task))
    return records


def _task_to_id(task):
    if isinstance(task, int):
        if task in ID2TASK:
            return task
        raise ValueError(f"Unknown task id: {task}")
    return TASK2ID[str(task).lower()]


def _degradation_to_id(degradation):
    if isinstance(degradation, int):
        if 0 <= degradation < NUM_DEGRADATIONS:
            return degradation
        raise ValueError(f"Degradation id must be in [0, {NUM_DEGRADATIONS - 1}].")
    return DEG2ID[str(degradation).lower()]


def _record_degradation(record):
    return record.get("deg", record.get("degradation", record.get("corruption")))


def _classification_label(record):
    value = record.get("ann", record.get("label", record.get("cls_label")))
    return int(value)


def _encode_train_ids(mask, mask_path, ignore_index):
    array = np.asarray(mask)
    if array.ndim == 3:
        array = array[:, :, 0]
    array = array.astype(np.int64, copy=False)
    return torch.from_numpy(np.ascontiguousarray(array)).long()


def _detection_targets(record, image_size):
    boxes = record.get("boxes", record.get("bboxes"))
    labels = record.get("labels", record.get("det_labels"))
    difficult = record.get("difficult")
    if record.get("box_format") != "xyxy":
        raise ValueError("VOC2012 records must declare box_format='xyxy'.")
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    difficult = np.asarray(difficult, dtype=np.int64).reshape(-1)
    regular = difficult == 0
    return boxes[regular], labels[regular], boxes[~regular], labels[~regular]


def _load_target(record, task_id, root, gt_size):
    cls_label, mask, mask_path = -1, None, None
    boxes = np.zeros((0, 4), dtype=np.float32)
    labels = np.zeros((0,), dtype=np.int64)
    ignore_boxes = np.zeros((0, 4), dtype=np.float32)
    ignore_labels = np.zeros((0,), dtype=np.int64)
    if task_id == TASK2ID["cls"]:
        cls_label = _classification_label(record)
    elif task_id == TASK2ID["seg"]:
        mask_path = _resolve_path(record.get("ann", record.get("mask")), root)
        mask = _read_mask(mask_path)
        if mask.size != gt_size:
            raise ValueError(f"Segmentation mask and GT sizes differ: {mask.size} vs {gt_size}")
    else:
        boxes, labels, ignore_boxes, ignore_labels = _detection_targets(record, gt_size)
    return cls_label, mask, mask_path, boxes, labels, ignore_boxes, ignore_labels


def _resize_boxes(boxes, scale_x, scale_y):
    boxes = boxes.astype(np.float32, copy=True)
    if boxes.size:
        boxes[:, (0, 2)] *= scale_x
        boxes[:, (1, 3)] *= scale_y
    return boxes


def _resize_for_crop(lq, gt, mask, boxes, ignore_boxes, crop_size):
    old_width, old_height = gt.size
    short_side = min(old_width, old_height)
    if short_side >= crop_size:
        return lq, gt, mask, boxes, ignore_boxes
    scale = crop_size / short_side
    new_size = (round(old_width * scale), round(old_height * scale))
    lq = lq.resize(new_size, _BICUBIC)
    gt = gt.resize(new_size, _BICUBIC)
    if mask is not None:
        mask = mask.resize(new_size, _NEAREST)
    scale_x, scale_y = new_size[0] / old_width, new_size[1] / old_height
    boxes = _resize_boxes(boxes, scale_x, scale_y)
    ignore_boxes = _resize_boxes(ignore_boxes, scale_x, scale_y)
    return lq, gt, mask, boxes, ignore_boxes


def _random_crop_box(width, height, size):
    left = random.randint(0, width - size)
    top = random.randint(0, height - size)
    return left, top, left + size, top + size


def _crop_boxes(boxes, labels, crop_box, min_size=1.0):
    if boxes.size == 0:
        return boxes.reshape(0, 4), labels.reshape(0)
    left, top, right, bottom = crop_box
    cropped = boxes.astype(np.float32, copy=True)
    cropped[:, (0, 2)] -= left
    cropped[:, (1, 3)] -= top
    cropped[:, (0, 2)] = np.clip(cropped[:, (0, 2)], 0, right - left)
    cropped[:, (1, 3)] = np.clip(cropped[:, (1, 3)], 0, bottom - top)
    keep = (
        (cropped[:, 2] - cropped[:, 0] >= min_size)
        & (cropped[:, 3] - cropped[:, 1] >= min_size)
    )
    return cropped[keep], labels[keep]


def _detection_crop_box(width, height, size, boxes, labels, attempts=10):
    for _ in range(attempts):
        crop_box = _random_crop_box(width, height, size)
        left, top, right, bottom = crop_box
        center_x = (boxes[:, 0] + boxes[:, 2]) * 0.5
        center_y = (boxes[:, 1] + boxes[:, 3]) * 0.5
        center_inside = (
            (center_x >= left) & (center_x < right)
            & (center_y >= top) & (center_y < bottom)
        )
        retained, _ = _crop_boxes(boxes, labels, crop_box)
        if center_inside.any() and len(retained):
            return crop_box

    target = boxes[random.randrange(len(boxes))]
    center_x = float(target[0] + target[2]) * 0.5
    center_y = float(target[1] + target[3]) * 0.5
    left = min(max(round(center_x - size * 0.5), 0), width - size)
    top = min(max(round(center_y - size * 0.5), 0), height - size)
    crop_box = (left, top, left + size, top + size)
    retained, _ = _crop_boxes(boxes, labels, crop_box)
    if not len(retained):
        raise RuntimeError("Unable to produce a detection crop containing an object.")
    return crop_box


def _crop_sample(
    lq, gt, mask, boxes, labels, ignore_boxes, ignore_labels, crop_box,
):
    lq, gt = lq.crop(crop_box), gt.crop(crop_box)
    if mask is not None:
        mask = mask.crop(crop_box)
    boxes, labels = _crop_boxes(boxes, labels, crop_box)
    ignore_boxes, ignore_labels = _crop_boxes(ignore_boxes, ignore_labels, crop_box)
    return lq, gt, mask, boxes, labels, ignore_boxes, ignore_labels


def _flip_boxes(boxes, width):
    if boxes.size:
        boxes = boxes.copy()
        x1, x2 = boxes[:, 0].copy(), boxes[:, 2].copy()
        boxes[:, 0], boxes[:, 2] = width - x2, width - x1
    return boxes


def _horizontal_flip(lq, gt, mask, boxes, ignore_boxes):
    if random.random() >= 0.5:
        return lq, gt, mask, boxes, ignore_boxes
    width = gt.size[0]
    lq, gt = lq.transpose(_FLIP_LEFT_RIGHT), gt.transpose(_FLIP_LEFT_RIGHT)
    if mask is not None:
        mask = mask.transpose(_FLIP_LEFT_RIGHT)
    boxes = _flip_boxes(boxes, width)
    ignore_boxes = _flip_boxes(ignore_boxes, width)
    return lq, gt, mask, boxes, ignore_boxes


def _pad_detection_targets(boxes, labels, max_boxes):
    count = min(len(boxes), max_boxes)
    padded_boxes = torch.zeros(max_boxes, 4, dtype=torch.float32)
    padded_labels = torch.full((max_boxes,), -1, dtype=torch.long)
    valid = torch.zeros(max_boxes, dtype=torch.bool)
    if count:
        padded_boxes[:count] = torch.from_numpy(np.ascontiguousarray(boxes[:count])).float()
        padded_labels[:count] = torch.from_numpy(np.ascontiguousarray(labels[:count])).long()
        valid[:count] = True
    return padded_boxes, padded_labels, valid, torch.tensor(count, dtype=torch.long)


class _TaskDatasetBase(Dataset):
    def __init__(self, list_file, root, default_task, ignore_index, max_det_boxes):
        super().__init__()
        self.root = root
        self.ignore_index = int(ignore_index)
        self.max_det_boxes = int(max_det_boxes)
        self.records = load_task_records(list_file, default_task)
        if not self.records:
            raise RuntimeError(f"No samples found in {list_file}")

    def __len__(self):
        return len(self.records)

    def _load_images(self, record):
        gt_path = _resolve_path(record.get("gt"), self.root)
        lq_path = _resolve_path(record.get("lq"), self.root)
        gt, lq = _read_rgb(gt_path), _read_rgb(lq_path)
        if lq.size != gt.size:
            raise ValueError(f"LQ and GT sizes differ: {lq.size} vs {gt.size}; LQ={lq_path}")
        return lq, gt, gt_path

    def _build_output(
        self, record, task_id, lq, gt, gt_path,
        cls_label, mask, mask_path, boxes, labels, ignore_boxes, ignore_labels,
    ):
        if mask is None:
            seg_mask = torch.full(
                (gt.size[1], gt.size[0]), self.ignore_index, dtype=torch.long,
            )
        else:
            seg_mask = _encode_train_ids(mask, mask_path, self.ignore_index)
        det_boxes, det_labels, det_valid, det_num = _pad_detection_targets(
            boxes, labels, self.max_det_boxes,
        )
        det_ignore_boxes, det_ignore_labels, det_ignore_valid, det_ignore_num = (
            _pad_detection_targets(ignore_boxes, ignore_labels, self.max_det_boxes)
        )
        return {
            "name": Path(gt_path).stem,
            "lq": to_tensor(lq),
            "gt": to_tensor(gt),
            "task_id": torch.tensor(task_id, dtype=torch.long),
            "degradation_id": torch.tensor(
                _degradation_to_id(_record_degradation(record)), dtype=torch.long,
            ),
            "cls_label": torch.tensor(cls_label, dtype=torch.long),
            "seg_mask": seg_mask,
            "det_boxes": det_boxes,
            "det_labels": det_labels,
            "det_valid": det_valid,
            "det_num": det_num,
            "det_ignore_boxes": det_ignore_boxes,
            "det_ignore_labels": det_ignore_labels,
            "det_ignore_valid": det_ignore_valid,
            "det_ignore_num": det_ignore_num,
        }


class TaskDrivenTrainDataset(_TaskDatasetBase):
    def __init__(
        self, list_file, root="", default_task=None, resolution=256,
        is_train=True, online_corrupt=False, crp_mode="common",
        ignore_index=255, max_det_boxes=100,
    ):
        _require_offline(online_corrupt)
        self.resolution = int(resolution)
        super().__init__(list_file, root, default_task, ignore_index, max_det_boxes)

    def __getitem__(self, index):
        record = self.records[index]
        task_id = _task_to_id(record.get("task"))
        lq, gt, gt_path = self._load_images(record)
        cls_label, mask, mask_path, boxes, labels, ignore_boxes, ignore_labels = _load_target(
            record, task_id, self.root, gt.size,
        )
        lq, gt, mask, boxes, ignore_boxes = _resize_for_crop(
            lq, gt, mask, boxes, ignore_boxes, self.resolution,
        )

        width, height = gt.size
        if task_id == TASK2ID["det"]:
            crop_box = _detection_crop_box(
                width, height, self.resolution, boxes, labels,
            )
        else:
            crop_box = _random_crop_box(width, height, self.resolution)
        lq, gt, mask, boxes, labels, ignore_boxes, ignore_labels = _crop_sample(
            lq, gt, mask, boxes, labels, ignore_boxes, ignore_labels, crop_box,
        )
        lq, gt, mask, boxes, ignore_boxes = _horizontal_flip(
            lq, gt, mask, boxes, ignore_boxes,
        )
        return self._build_output(
            record, task_id, lq, gt, gt_path,
            cls_label, mask, mask_path, boxes, labels, ignore_boxes, ignore_labels,
        )


class TaskDrivenValDataset(_TaskDatasetBase):
    def __init__(
        self, list_file, root="", default_task=None, online_corrupt=False,
        crp_mode="common", ignore_index=255, max_det_boxes=100,
    ):
        _require_offline(online_corrupt)
        super().__init__(list_file, root, default_task, ignore_index, max_det_boxes)

    def __getitem__(self, index):
        record = self.records[index]
        task_id = _task_to_id(record.get("task"))
        lq, gt, gt_path = self._load_images(record)
        cls_label, mask, mask_path, boxes, labels, ignore_boxes, ignore_labels = _load_target(
            record, task_id, self.root, gt.size,
        )
        return self._build_output(
            record, task_id, lq, gt, gt_path,
            cls_label, mask, mask_path, boxes, labels, ignore_boxes, ignore_labels,
        )
