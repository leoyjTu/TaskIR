import math
import os
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOWNSTREAM_ROOT = PROJECT_ROOT / "downstream"
CLS_NUM_CLASSES = 1000
SEG_NUM_CLASSES = 19
DET_NUM_CLASSES = 20
HIERARCHICAL_FEATURE_LEVELS = ("level1", "level2", "level3")
EXPECTED_HIERARCHICAL_STRIDES = (8, 16, 32)
DEFAULT_RESNET50_CKPT = str(DOWNSTREAM_ROOT / "ResNet50" / "ckpt" / "resnet50-11ad3fa6.pth")
DEFAULT_SEGFORMER_CKPT = str(
    DOWNSTREAM_ROOT / "SegFormer" / "ckpt" / "segformer-b5-finetuned-cityscapes-1024-1024"
)
DEFAULT_YOLOV8_CKPT = str(DOWNSTREAM_ROOT / "YOLOv8" / "ckpt" / "yolov8n-voc2012.pt")
_STATE_PREFIXES = ("module.", "model.", "net.", "pre_net.", "network.")


def _resolve_checkpoint_path(path):
    if not path:
        return ""
    candidate = Path(os.path.expanduser(str(path)))
    if not candidate.is_absolute():
        project_candidate = PROJECT_ROOT / candidate
        if project_candidate.exists():
            candidate = project_candidate
    return str(candidate.resolve())


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, nn.Module):
        return checkpoint.state_dict()
    if not isinstance(checkpoint, dict):
        return checkpoint
    for key in ("state_dict", "model_state_dict", "model", "net", "params_ema", "params"):
        value = checkpoint.get(key)
        if isinstance(value, nn.Module):
            return value.state_dict()
        if isinstance(value, (dict, OrderedDict)):
            return value
    return checkpoint


def _candidate_state_keys(key):
    yield key
    while True:
        prefix = next((item for item in _STATE_PREFIXES if key.startswith(item)), None)
        if prefix is None:
            return
        key = key[len(prefix):]
        yield key


def load_state_dict_flexible(model, ckpt_path, strict=False, min_coverage=0.0):
    if not ckpt_path:
        return {"used_keys": 0, "key_coverage": 0.0, "parameter_coverage": 0.0}

    ckpt_path = _resolve_checkpoint_path(ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    source = _extract_state_dict(checkpoint)
    model_state = model.state_dict()
    matched = {}
    for source_key, value in source.items():
        if not torch.is_tensor(value):
            continue
        for candidate in _candidate_state_keys(str(source_key)):
            if candidate in model_state and model_state[candidate].shape == value.shape:
                matched.setdefault(candidate, value)
                break

    total_parameters = sum(value.numel() for value in model_state.values())
    loaded_parameters = sum(model_state[key].numel() for key in matched)
    key_coverage = len(matched) / max(len(model_state), 1)
    parameter_coverage = loaded_parameters / max(total_parameters, 1)
    if parameter_coverage < float(min_coverage):
        missing = [key for key in model_state if key not in matched]
        raise RuntimeError(
            f"Checkpoint parameter coverage is {parameter_coverage:.2%}, below the required "
            f"{float(min_coverage):.2%}. Missing examples: {missing[:10]}"
        )

    model.load_state_dict(matched, strict=strict)
    return {
        "used_keys": len(matched),
        "key_coverage": key_coverage,
        "parameter_coverage": parameter_coverage,
    }


def normalize_for_task(images):
    mean = images.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = images.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    return (images.clamp(0, 1) - mean) / std


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def extract_resnet_hierarchical_features(model, images):
    model = _unwrap_model(model)
    feature = model.maxpool(model.relu(model.bn1(model.conv1(images))))
    feature = model.layer1(feature)
    features = {
        "level1": model.layer2(feature),
    }
    features["level2"] = model.layer3(features["level1"])
    features["level3"] = model.layer4(features["level2"])
    return features


def freeze_model(model):
    model.requires_grad_(False)
    model.eval()
    return model


def build_cls_net(num_classes=CLS_NUM_CLASSES, ckpt_path="", device="cpu"):
    from torchvision.models import resnet50

    ckpt_path = _resolve_checkpoint_path(ckpt_path or DEFAULT_RESNET50_CKPT)
    model = resnet50(weights=None)
    load_state_dict_flexible(model, ckpt_path, min_coverage=0.99)
    return freeze_model(model.to(device))


class HFSegFormerWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        height, width = images.shape[-2:]
        logits = self.model(pixel_values=images, return_dict=True).logits
        if logits.shape[-2:] != (height, width):
            logits = F.interpolate(logits, (height, width), mode="bilinear", align_corners=False)
        return {"out": logits}

    def extract_hierarchical_features(self, images):
        outputs = self.model.segformer(
            pixel_values=images, output_hidden_states=True, return_dict=True,
        )
        hidden_states = outputs.hidden_states
        features = {
            "level1": hidden_states[1],
            "level2": hidden_states[2],
            "level3": hidden_states[3],
        }
        return features


def build_seg_net(num_classes=SEG_NUM_CLASSES, ckpt_path="", device="cpu"):
    try:
        from transformers import SegformerForSemanticSegmentation
    except ImportError as error:
        raise ImportError(
            "SegFormer-B5 requires transformers and safetensors. Install them before training."
        ) from error

    ckpt_path = _resolve_checkpoint_path(ckpt_path or DEFAULT_SEGFORMER_CKPT)
    model = SegformerForSemanticSegmentation.from_pretrained(
        ckpt_path, local_files_only=True, use_safetensors=True,
    )
    if int(model.config.num_labels) != SEG_NUM_CLASSES:
        raise ValueError(
            f"SegFormer checkpoint has {model.config.num_labels} classes; expected {SEG_NUM_CLASSES}."
        )
    return freeze_model(HFSegFormerWrapper(model).to(device))


class YOLOv8Detector(nn.Module):
    def __init__(self, weights, device="cpu", imgsz=640, conf=0.001, iou=0.7):
        super().__init__()
        try:
            from ultralytics import YOLO
            from ultralytics.cfg import get_cfg
        except ImportError as error:
            raise ImportError("YOLOv8 requires 'ultralytics>=8.2,<9'.") from error

        yolo = YOLO(str(weights))
        object.__setattr__(self, "_yolo", yolo)
        self.detector = yolo.model.to(device)
        raw_args = getattr(self.detector, "args", None)
        if isinstance(raw_args, dict):
            self.detector.args = get_cfg(raw_args)
        elif raw_args is None:
            self.detector.args = get_cfg()
        defaults = get_cfg()
        for name in ("box", "cls", "dfl"):
            if not hasattr(self.detector.args, name):
                setattr(self.detector.args, name, getattr(defaults, name))
        if hasattr(self.detector, "criterion"):
            delattr(self.detector, "criterion")

        stride = int(torch.as_tensor(getattr(self.detector, "stride", (32,))).max().item())
        self.imgsz = int(math.ceil(int(imgsz) / stride) * stride)
        self.conf, self.iou = float(conf), float(iou)
        head = self.detector.model[-1]
        self.nc = int(getattr(head, "nc", 0))
        self.names = getattr(self.detector, "names", getattr(yolo, "names", None))
        self.feature_channels = self._infer_feature_channels()

    def _infer_feature_channels(self):
        branches = getattr(self.detector.model[-1], "cv2", None)
        if branches is None:
            return None
        channels = []
        for branch in list(branches)[:3]:
            convolution = next(
                (module for module in branch.modules() if isinstance(module, nn.Conv2d)), None,
            )
            if convolution is not None:
                channels.append(int(convolution.in_channels))
        return tuple(channels) if len(channels) == 3 else None

    def _letterbox(self, images):
        height, width = images.shape[-2:]
        scale = min(self.imgsz / height, self.imgsz / width)
        resized_height = max(round(height * scale), 1)
        resized_width = max(round(width * scale), 1)
        resized = F.interpolate(
            images, (resized_height, resized_width), mode="bilinear", align_corners=False,
        )
        pad_height, pad_width = self.imgsz - resized_height, self.imgsz - resized_width
        top, left = pad_height // 2, pad_width // 2
        padded = F.pad(
            resized,
            (left, pad_width - left, top, pad_height - top),
            value=114.0 / 255.0,
        )
        return padded, scale, left, top

    @staticmethod
    def _freeze_batch_norm_statistics(model):
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()

    def get_feature_channels(self):
        if self.feature_channels is None:
            device = next(self.detector.parameters()).device
            with torch.no_grad():
                dummy = torch.zeros(1, 3, self.imgsz, self.imgsz, device=device)
                self.feature_channels = tuple(
                    int(feature.shape[1]) for feature in self.extract_features(dummy)
                )
        return self.feature_channels

    def extract_features(self, images):
        captured = {}

        def capture_head_input(_module, inputs):
            value = inputs[0]
            captured["features"] = list(value) if isinstance(value, (list, tuple)) else [value]

        images, _, _, _ = self._letterbox(images.clamp(0, 1))
        head = self.detector.model[-1]
        handle = head.register_forward_pre_hook(capture_head_input)
        previous_mode = self.detector.training
        self.detector.eval()
        try:
            self.detector(images)
        finally:
            handle.remove()
            self.detector.train(previous_mode)

        features = captured.get("features")
        if not features or len(features) < 3:
            raise RuntimeError("Could not capture the three YOLOv8 detection-head inputs.")
        return features[:3]

    def get_hierarchical_feature_info(self, features):
        if isinstance(features, dict):
            ordered = [features[level] for level in HIERARCHICAL_FEATURE_LEVELS]
        else:
            ordered = list(features)
        if len(ordered) != 3:
            raise ValueError(f"YOLO hierarchy requires three Detect inputs, got {len(ordered)}.")

        info = {}
        for level, feature in zip(HIERARCHICAL_FEATURE_LEVELS, ordered):
            height, width = feature.shape[-2:]
            info[level] = {
                "shape": tuple(feature.shape),
                "channels": int(feature.shape[1]),
                "stride": (self.imgsz / height, self.imgsz / width),
            }
        return info

    def extract_hierarchical_features(self, images):
        raw_features = self.extract_features(images)
        features = {
            level: feature
            for level, feature in zip(HIERARCHICAL_FEATURE_LEVELS, raw_features)
        }
        info = self.get_hierarchical_feature_info(features)
        mismatches = []
        for level, expected_stride in zip(
            HIERARCHICAL_FEATURE_LEVELS, EXPECTED_HIERARCHICAL_STRIDES,
        ):
            actual_stride = info[level]["stride"]
            if any(abs(value - expected_stride) > 1e-6 for value in actual_stride):
                mismatches.append(
                    f"{level}: shape={info[level]['shape']}, stride={actual_stride}, "
                    f"expected=({expected_stride}, {expected_stride})"
                )
        if mismatches:
            raise RuntimeError(
                "YOLO Detect inputs do not match the expected P3/P4/P5 strides: "
                + "; ".join(mismatches)
            )
        return features

    def _build_loss_batch(self, images, targets):
        images = images.clamp(0, 1)
        letterboxed, scale, pad_left, pad_top = self._letterbox(images)
        batch_indices, classes, normalized_boxes = [], [], []

        for image_index, target in enumerate(targets):
            boxes = target["boxes"].to(device=images.device, dtype=torch.float32)
            labels = target["labels"].to(device=images.device, dtype=torch.long)
            if boxes.numel() == 0:
                continue
            boxes = boxes.clone()
            boxes[:, (0, 2)] = boxes[:, (0, 2)] * scale + pad_left
            boxes[:, (1, 3)] = boxes[:, (1, 3)] * scale + pad_top
            boxes.clamp_(0, self.imgsz)
            center_x = (boxes[:, 0] + boxes[:, 2]) * 0.5 / self.imgsz
            center_y = (boxes[:, 1] + boxes[:, 3]) * 0.5 / self.imgsz
            width = (boxes[:, 2] - boxes[:, 0]) / self.imgsz
            height = (boxes[:, 3] - boxes[:, 1]) / self.imgsz
            valid = (width > 0) & (height > 0)
            if not valid.any():
                continue

            normalized_boxes.append(
                torch.stack((center_x[valid], center_y[valid], width[valid], height[valid]), dim=1)
            )
            classes.append(labels[valid].to(dtype=images.dtype).unsqueeze(1))
            batch_indices.append(
                torch.full(
                    (int(valid.sum().item()),), image_index, device=images.device, dtype=torch.long,
                )
            )

        if normalized_boxes:
            boxes_tensor = torch.cat(normalized_boxes)
            classes_tensor = torch.cat(classes)
            indices_tensor = torch.cat(batch_indices)
        else:
            boxes_tensor = images.new_zeros((0, 4))
            classes_tensor = images.new_zeros((0, 1))
            indices_tensor = torch.zeros(0, device=images.device, dtype=torch.long)
        return {
            "img": letterboxed,
            "batch_idx": indices_tensor,
            "cls": classes_tensor,
            "bboxes": boxes_tensor,
        }

    def loss(self, images, targets):
        """Compute frozen-detector loss while retaining gradients to input images."""
        if isinstance(images, (list, tuple)):
            images = torch.stack(tuple(images))
        batch = self._build_loss_batch(images, targets)
        previous_mode = self.detector.training
        self.detector.train()
        self._freeze_batch_norm_statistics(self.detector)
        try:
            output = self.detector(batch)
        finally:
            self.detector.train(previous_mode)
        loss_items = output[0] if isinstance(output, (tuple, list)) else output
        if not torch.is_tensor(loss_items):
            raise RuntimeError(f"Unexpected YOLOv8 loss type: {type(loss_items).__name__}")
        return loss_items.sum()

    @torch.no_grad()
    def predict(self, images):
        if isinstance(images, (list, tuple)):
            images = torch.stack(tuple(images))
        images = images.clamp(0, 1).float()
        original_height, original_width = images.shape[-2:]
        letterboxed, scale, pad_left, pad_top = self._letterbox(images)
        device = next(self.detector.parameters()).device
        results = self._yolo.predict(
            source=letterboxed, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
            device=str(device), verbose=False, save=False,
        )

        predictions = []
        for result in results:
            result_boxes = result.boxes
            if result_boxes is None or len(result_boxes) == 0:
                predictions.append({
                    "boxes": images.new_zeros((0, 4)),
                    "scores": images.new_zeros((0,)),
                    "labels": torch.zeros(0, device=images.device, dtype=torch.long),
                })
                continue

            boxes = result_boxes.xyxy.to(images.device, dtype=torch.float32).clone()
            boxes[:, (0, 2)] = (boxes[:, (0, 2)] - pad_left) / scale
            boxes[:, (1, 3)] = (boxes[:, (1, 3)] - pad_top) / scale
            boxes[:, (0, 2)].clamp_(0, original_width)
            boxes[:, (1, 3)].clamp_(0, original_height)
            valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            predictions.append({
                "boxes": boxes[valid],
                "scores": result_boxes.conf.to(images.device, dtype=torch.float32)[valid],
                "labels": result_boxes.cls.to(images.device, dtype=torch.long)[valid],
            })
        return predictions

    def forward(self, images):
        return self.predict(images)


def build_det_net(
    num_classes=DET_NUM_CLASSES, ckpt_path="", device="cpu",
    imgsz=640, conf=0.001, iou=0.7,
):
    ckpt_path = _resolve_checkpoint_path(ckpt_path or DEFAULT_YOLOV8_CKPT)
    model = YOLOv8Detector(ckpt_path, device, imgsz, conf, iou)
    if model.nc != DET_NUM_CLASSES:
        raise ValueError(
            f"YOLOv8 checkpoint has {model.nc} classes; expected {DET_NUM_CLASSES}."
        )
    return freeze_model(model)


def extract_hierarchical_task_features(task, model, images):
    task = str(task).lower()
    model = _unwrap_model(model)
    if task == "cls":
        return extract_resnet_hierarchical_features(model, normalize_for_task(images))
    if task == "seg":
        return model.extract_hierarchical_features(normalize_for_task(images))
    if task == "det":
        return model.extract_hierarchical_features(images)
    raise ValueError(f"Unknown task {task!r}; expected 'cls', 'seg' or 'det'.")


def build_downstream_models(opt, enabled_tasks, device="cpu"):
    enabled_tasks = tuple(enabled_tasks)
    unknown = set(enabled_tasks) - {"cls", "seg", "det"}
    if unknown:
        raise ValueError(f"Unknown downstream tasks: {sorted(unknown)}")

    models = {}
    if "cls" in enabled_tasks:
        models["cls"] = build_cls_net(
            getattr(opt, "cls_num_classes", CLS_NUM_CLASSES),
            getattr(opt, "cls_ckpt", ""), device,
        )
    if "seg" in enabled_tasks:
        models["seg"] = build_seg_net(
            getattr(opt, "seg_num_classes", SEG_NUM_CLASSES),
            getattr(opt, "seg_ckpt", ""), device,
        )
    if "det" in enabled_tasks:
        models["det"] = build_det_net(
            getattr(opt, "det_num_classes", DET_NUM_CLASSES),
            getattr(opt, "det_ckpt", ""), device,
            getattr(opt, "det_imgsz", 640), getattr(opt, "det_conf", 0.001),
            getattr(opt, "det_iou", 0.7),
        )
    return models
