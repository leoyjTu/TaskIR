import torch
import torch.nn.functional as F

from utils.downstream_utils import normalize_for_task


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def compute_seg_miou(prediction, target, num_classes=19, ignore_index=255):
    prediction = prediction.detach().reshape(-1)
    target = target.detach().reshape(-1)
    valid = (target != ignore_index) & (target >= 0) & (target < num_classes)
    prediction, target = prediction[valid], target[valid]
    if target.numel() == 0:
        return None

    confusion = torch.bincount(
        num_classes * target + prediction, minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes).float()
    intersection = confusion.diag()
    union = confusion.sum(0) + confusion.sum(1) - intersection
    present = union > 0
    return (intersection[present] / union[present]).mean() if present.any() else None


def _valid_det_entries(boxes, labels, valid):
    boxes = boxes[valid].float()
    labels = labels[valid].long()
    if boxes.numel():
        keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes, labels = boxes[keep], labels[keep]
    return boxes, labels


def build_det_targets(batch, det_valid, opt, device, include_difficult=False):
    boxes = batch["det_boxes"].to(device)
    labels = batch["det_labels"].to(device)
    box_valid = batch["det_valid"].to(device)
    if include_difficult:
        ignore_boxes = batch["det_ignore_boxes"].to(device)
        ignore_labels = batch["det_ignore_labels"].to(device)
        ignore_valid = batch["det_ignore_valid"].to(device)
    targets = []

    for index in det_valid.nonzero(as_tuple=False).flatten():
        current_boxes, current_labels = _valid_det_entries(
            boxes[index], labels[index], box_valid[index],
        )
        target = {"boxes": current_boxes, "labels": current_labels}
        if include_difficult:
            difficult_boxes, difficult_labels = _valid_det_entries(
                ignore_boxes[index], ignore_labels[index], ignore_valid[index],
            )
            target["boxes"] = torch.cat((current_boxes, difficult_boxes))
            target["labels"] = torch.cat((current_labels, difficult_labels))
            target["iscrowd"] = torch.cat((
                torch.zeros(len(current_boxes), dtype=torch.long, device=device),
                torch.ones(len(difficult_boxes), dtype=torch.long, device=device),
            ))
        targets.append(target)
    return targets


def compute_det_loss(restored, batch, det_valid, det_model, opt, device):
    if det_model is None or not det_valid.any():
        return restored.new_zeros(())
    detector = _unwrap_model(det_model)
    images = restored[det_valid].clamp(0, 1)
    targets = build_det_targets(batch, det_valid, opt, device)
    return detector.loss(images, targets)


def compute_task_loss(restored, batch, downstream, opt, TASK2ID, device):
    task_id = batch["task_id"].to(device)
    loss_cls = restored.new_zeros(())
    loss_seg = restored.new_zeros(())
    loss_det = restored.new_zeros(())
    beta_cls = getattr(opt, "beta_cls", 0.0)
    beta_seg = getattr(opt, "beta_seg", 0.0)
    beta_det = getattr(opt, "beta_det", 0.0)

    labels = batch["cls_label"].to(device)
    cls_valid = (task_id == TASK2ID["cls"]) & (labels >= 0)
    if beta_cls > 0 and "cls" in downstream and cls_valid.any():
        logits = downstream["cls"](normalize_for_task(restored[cls_valid]))
        target = labels[cls_valid].long()
        loss_cls = F.cross_entropy(logits, target)

    seg_valid = task_id == TASK2ID["seg"]
    if beta_seg > 0 and "seg" in downstream and seg_valid.any():
        output = downstream["seg"](normalize_for_task(restored[seg_valid]))
        logits = output["out"] if isinstance(output, dict) else output
        target = batch["seg_mask"].to(device)[seg_valid].long()
        if logits.shape[-2:] != target.shape[-2:]:
            logits = F.interpolate(
                logits, size=target.shape[-2:], mode="bilinear", align_corners=False,
            )
        ignore_index = getattr(opt, "ignore_index", 255)
        if (target != ignore_index).any():
            loss_seg = F.cross_entropy(logits, target, ignore_index=ignore_index)

    det_valid = task_id == TASK2ID["det"]
    if beta_det > 0 and "det" in downstream and det_valid.any():
        loss_det = compute_det_loss(
            restored, batch, det_valid, downstream["det"], opt, device,
        )

    total = beta_cls * loss_cls + beta_seg * loss_seg + beta_det * loss_det
    return total, {
        "loss_cls": loss_cls, "loss_seg": loss_seg, "loss_det": loss_det,
    }
