#!/usr/bin/env python3

import argparse
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from PIL import Image
from tqdm import tqdm


DEGRADATIONS = (
    "gaussian_noise", "shot_noise", "defocus_blur", "motion_blur",
    "snow", "fog", "brightness", "jpeg_compression",
)
VOC20_NAMES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)
VOC_NAME_TO_ID = {name: index for index, name in enumerate(VOC20_NAMES)}


def find_project_root():
    current_file = Path(__file__).resolve()
    for candidate in current_file.parents:
        if (candidate / "data" / "corruption").is_dir():
            return candidate
        if (candidate / "datasets").is_dir() and (candidate / "data_dir").is_dir():
            return candidate
    return current_file.parents[2]


PROJECT_ROOT = find_project_root()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate strict VOC2012 TaskIR detection JSONL files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--voc_root", default=str(PROJECT_ROOT / "datasets" / "VOC2012"), help="VOC2012 root containing GT, Annotations and LQ.")
    parser.add_argument("--out_dir", default=str(PROJECT_ROOT / "data_dir" / "lists"), help="Directory for train_det.jsonl and val_det.jsonl.")
    parser.add_argument("--train_name", default="train_det.jsonl", help="Training JSONL filename.")
    parser.add_argument("--val_name", default="val_det.jsonl", help="Validation JSONL filename.")
    parser.add_argument("--strict", dest="strict", action="store_true", help="Stop on missing/malformed annotations or LQ images.")
    parser.add_argument("--no_strict", dest="strict", action="store_false", help="Skip missing/malformed samples and report them.")
    parser.set_defaults(strict=True)
    return parser.parse_args()


def collect_images(root):
    root = Path(root)
    images = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in (".jpg", ".jpeg")
    )
    if not images:
        raise RuntimeError(f"No VOC2012 images found under: {root}")
    return images


def _read_coordinate(box_node, name):
    return float(box_node.findtext(name))


def parse_voc_annotation(xml_path, image_size):
    """Convert VOC 1-based inclusive boxes to zero-based half-open xyxy boxes."""
    root = ET.parse(xml_path).getroot()

    width, height = image_size
    boxes, labels, difficult, names = [], [], [], []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        label = VOC_NAME_TO_ID[name]
        box_node = obj.find("bndbox")
        x1 = _read_coordinate(box_node, "xmin") - 1.0
        y1 = _read_coordinate(box_node, "ymin") - 1.0
        x2 = _read_coordinate(box_node, "xmax")
        y2 = _read_coordinate(box_node, "ymax")
        is_difficult = int(obj.findtext("difficult") or 0)

        if is_difficult not in (0, 1):
            raise ValueError(f"VOC difficult must be 0 or 1: {xml_path}")
        x1, x2 = min(max(x1, 0.0), width), min(max(x2, 0.0), width)
        y1, y2 = min(max(y1, 0.0), height), min(max(y2, 0.0), height)
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid VOC box after clipping: {(x1, y1, x2, y2)}; {xml_path}")
        boxes.append([x1, y1, x2, y2])
        labels.append(label)
        difficult.append(is_difficult)
        names.append(name)
    if not boxes:
        raise ValueError(f"VOC annotation contains no objects: {xml_path}")
    return boxes, labels, difficult, names


def _handle_problem(message, strict):
    if strict:
        raise RuntimeError(message)
    print(message)
    return False


def write_voc_split(gt_root, annotation_root, lq_root, split, output_path, strict=True):
    gt_root = Path(gt_root).resolve()
    annotation_root = Path(annotation_root).resolve()
    lq_root = Path(lq_root).resolve()
    output_path = Path(output_path).resolve()

    images = collect_images(gt_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    samples = used_images = regular_boxes = difficult_boxes = 0
    missing_annotations = malformed_annotations = missing_lq = skipped_all_difficult = 0
    class_histogram = Counter()

    try:
        with temporary_path.open("w", encoding="utf-8") as output:
            for clean_path in tqdm(images, desc=f"Building VOC2012 {split} JSONL"):
                relative_path = clean_path.relative_to(gt_root)
                annotation_path = annotation_root / relative_path.with_suffix(".xml")
                if not annotation_path.is_file():
                    missing_annotations += 1
                    if not _handle_problem(f"Missing VOC annotation: {annotation_path}", strict):
                        continue
                try:
                    with Image.open(clean_path) as image:
                        boxes, labels, difficult, names = parse_voc_annotation(
                            annotation_path, image.size,
                        )
                except (
                    AttributeError, KeyError, OSError, ET.ParseError,
                    TypeError, ValueError,
                ) as error:
                    malformed_annotations += 1
                    if not _handle_problem(str(error), strict):
                        continue

                trainable = [index for index, flag in enumerate(difficult) if not flag]
                if split == "train" and not trainable:
                    skipped_all_difficult += 1
                    continue

                written_for_image = 0
                for degradation in DEGRADATIONS:
                    lq_path = lq_root / degradation / relative_path
                    if not lq_path.is_file():
                        missing_lq += 1
                        if not _handle_problem(f"Missing VOC LQ image: {lq_path}", strict):
                            continue
                    record = {
                        "task": "det",
                        "lq": str(lq_path.resolve()),
                        "gt": str(clean_path.resolve()),
                        "ann": str(annotation_path.resolve()),
                        "boxes": boxes,
                        "labels": labels,
                        "difficult": difficult,
                        "box_format": "xyxy",
                        "image_id": clean_path.stem,
                        "deg": degradation,
                    }
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    samples += 1
                    written_for_image += 1

                if written_for_image:
                    used_images += 1
                    regular_boxes += len(trainable)
                    difficult_boxes += sum(difficult)
                    class_histogram.update(names[index] for index in trainable)
        temporary_path.replace(output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    print("\n" + "=" * 80)
    print(f"VOC2012 {split} JSONL summary")
    print(f"GT images: {len(images)}")
    print(f"Used GT images: {used_images}")
    print(f"Written samples: {samples}")
    print(f"Regular boxes: {regular_boxes}")
    print(f"Difficult boxes retained as ignore regions: {difficult_boxes}")
    print(f"Training images skipped because every object is difficult: {skipped_all_difficult}")
    print(f"Missing annotations: {missing_annotations}")
    print(f"Malformed annotations: {malformed_annotations}")
    print(f"Missing LQ images: {missing_lq}")
    print(f"Regular-object class coverage: {len(class_histogram)}/{len(VOC20_NAMES)}")
    print(f"Regular-object class histogram: {dict(sorted(class_histogram.items()))}")
    print(f"Saved to: {output_path}")
    print("=" * 80)


def main():
    args = parse_args()
    root = Path(args.voc_root).resolve()
    output_dir = Path(args.out_dir).resolve()
    for split, filename in (("train", args.train_name), ("val", args.val_name)):
        write_voc_split(
            gt_root=root / "GT" / split,
            annotation_root=root / "Annotations" / split,
            lq_root=root / "LQ" / split,
            split=split,
            output_path=output_dir / filename,
            strict=args.strict,
        )


if __name__ == "__main__":
    main()
