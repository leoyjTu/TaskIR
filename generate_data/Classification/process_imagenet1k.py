#!/usr/bin/env python3

import argparse
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from PIL import Image
from tqdm import tqdm


DEGRADATIONS = (
    "gaussian_noise", "shot_noise", "defocus_blur", "motion_blur",
    "snow", "fog", "brightness", "jpeg_compression",
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
WNID_PATTERN = re.compile(r"n\d{8}")


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
        description="Generate strict ImageNet-1K TaskIR classification JSONL files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--imagenet_root",
        default=str(PROJECT_ROOT / "datasets" / "ImageNet1K"),
        help="ImageNet-1K root containing GT, LQ and Annotations.",
    )
    parser.add_argument(
        "--out_dir", default=str(PROJECT_ROOT / "data_dir" / "lists"),
        help="Directory for train_cls.jsonl and val_cls.jsonl.",
    )
    parser.add_argument(
        "--train_name", default="train_cls.jsonl", help="Training JSONL filename.",
    )
    parser.add_argument(
        "--val_name", default="val_cls.jsonl", help="Validation JSONL filename.",
    )
    parser.add_argument(
        "--expected_train_images", type=int, default=4000,
        help="Required number of selected training images; use 0 to disable.",
    )
    parser.add_argument(
        "--expected_val_images", type=int, default=1000,
        help="Required number of selected validation images; use 0 to disable.",
    )
    parser.add_argument(
        "--expected_train_per_class", type=int, default=4,
        help="Required training images per class; use 0 to disable.",
    )
    parser.add_argument(
        "--expected_val_per_class", type=int, default=1,
        help="Required validation images per class; use 0 to disable.",
    )
    parser.add_argument(
        "--strict", dest="strict", action="store_true",
        help="Stop on missing, malformed or size-mismatched files.",
    )
    parser.add_argument(
        "--no_strict", dest="strict", action="store_false",
        help="Skip invalid samples and report them instead of stopping.",
    )
    parser.set_defaults(strict=True)
    return parser.parse_args()


def collect_images(root):
    root = Path(root)
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def build_class_to_idx(train_root):
    train_root = Path(train_root).resolve()
    synsets = sorted(path.name for path in train_root.iterdir() if path.is_dir())
    invalid = [synset for synset in synsets if WNID_PATTERN.fullmatch(synset) is None]
    if len(synsets) != 1000 or len(set(synsets)) != 1000 or invalid:
        raise RuntimeError(
            f"Expected 1000 unique ImageNet WNID folders under {train_root}; "
            f"found {len(synsets)}, invalid={invalid[:10]}."
        )
    return {synset: index for index, synset in enumerate(synsets)}


def parse_xml_synset(xml_path):
    root = ET.parse(xml_path).getroot()
    names = {
        node.text.strip() for node in root.findall(".//object/name")
        if node.text and node.text.strip()
    }
    if not names:
        names = {
            node.text.strip() for node in root.findall(".//name")
            if node.text and node.text.strip()
        }
    if len(names) != 1:
        raise ValueError(f"Expected one ImageNet WNID in {xml_path}, found {sorted(names)}")
    synset = next(iter(names))
    return synset


def annotation_path_for(clean_path, clean_root, annotation_root):
    relative_path = clean_path.relative_to(clean_root)
    return annotation_root / relative_path.with_suffix(".xml")


def resolve_label(clean_path, clean_root, annotation_root, split, class_to_idx):
    relative_path = clean_path.relative_to(clean_root)
    xml_path = annotation_path_for(clean_path, clean_root, annotation_root)
    xml_synset = parse_xml_synset(xml_path)

    if split == "train":
        folder_synset = relative_path.parts[0]
        if folder_synset != xml_synset:
            raise ValueError(
                f"ImageNet training folder/XML mismatch: {folder_synset} vs {xml_synset}; "
                f"{clean_path}"
            )
    elif split == "val":
        if len(relative_path.parts) >= 2:
            folder_synset = relative_path.parts[0]
            if folder_synset in class_to_idx and folder_synset != xml_synset:
                raise ValueError(
                    f"ImageNet validation folder/XML mismatch: {folder_synset} vs "
                    f"{xml_synset}; {clean_path}"
                )
    return class_to_idx[xml_synset], xml_synset, xml_path


def image_size(path):
    with Image.open(path) as image:
        image.load()
        return image.size


def validate_distribution(split, image_count, class_counts, expected_images, expected_per_class):
    problems = []
    if expected_images > 0 and image_count != expected_images:
        problems.append(
            f"contains {image_count} selected images; expected {expected_images}"
        )
    if len(class_counts) != 1000:
        problems.append(f"covers {len(class_counts)}/1000 classes")
    if expected_per_class > 0:
        incorrect = {
            synset: count for synset, count in class_counts.items()
            if count != expected_per_class
        }
        if incorrect:
            problems.append(
                f"expected {expected_per_class} image(s) per class; "
                f"invalid examples={list(sorted(incorrect.items()))[:10]}"
            )
    if problems:
        raise RuntimeError(f"ImageNet {split}: " + "; ".join(problems))


def handle_problem(message, strict):
    if strict:
        raise RuntimeError(message)
    print(message)


def write_imagenet_split(
    clean_root, lq_root, annotation_root, split, output_path, class_to_idx,
    expected_images, expected_per_class, strict=True,
):
    clean_root = Path(clean_root).resolve()
    lq_root = Path(lq_root).resolve()
    annotation_root = Path(annotation_root).resolve()
    output_path = Path(output_path).resolve()

    resolved = []
    class_counts = Counter()
    invalid_gt = invalid_label = 0
    for clean_path in tqdm(collect_images(clean_root), desc=f"Checking ImageNet {split}"):
        try:
            clean_size = image_size(clean_path)
        except OSError as error:
            invalid_gt += 1
            handle_problem(f"Unreadable ImageNet GT image: {clean_path}; {error}", strict)
            continue
        try:
            label, synset, xml_path = resolve_label(
                clean_path, clean_root, annotation_root, split, class_to_idx,
            )
        except (OSError, ET.ParseError, KeyError, ValueError) as error:
            invalid_label += 1
            handle_problem(str(error), strict)
            continue
        resolved.append((clean_path, clean_size, label, synset, xml_path))
        class_counts[synset] += 1

    validate_distribution(
        split, len(resolved), class_counts, expected_images, expected_per_class,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    samples = complete_images = missing_lq = invalid_lq = 0

    try:
        with temporary_path.open("w", encoding="utf-8") as output:
            for clean_path, clean_size, label, synset, xml_path in tqdm(
                resolved, desc=f"Building ImageNet {split} JSONL",
            ):
                relative_path = clean_path.relative_to(clean_root)
                written_for_image = 0
                for degradation in DEGRADATIONS:
                    lq_path = lq_root / degradation / relative_path
                    if not lq_path.is_file():
                        missing_lq += 1
                        handle_problem(f"Missing ImageNet LQ image: {lq_path}", strict)
                        continue
                    try:
                        lq_size = image_size(lq_path)
                    except OSError as error:
                        invalid_lq += 1
                        handle_problem(f"Unreadable ImageNet LQ image: {lq_path}; {error}", strict)
                        continue
                    if lq_size != clean_size:
                        invalid_lq += 1
                        handle_problem(
                            f"ImageNet GT/LQ size mismatch: {clean_size} vs {lq_size}; {lq_path}",
                            strict,
                        )
                        continue
                    record = {
                        "task": "cls",
                        "lq": str(lq_path.resolve()),
                        "gt": str(clean_path.resolve()),
                        "ann": label,
                        "class_name": synset,
                        "annotation": str(xml_path.resolve()),
                        "image_id": clean_path.stem,
                        "deg": degradation,
                    }
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    samples += 1
                    written_for_image += 1
                if written_for_image == len(DEGRADATIONS):
                    complete_images += 1
        temporary_path.replace(output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    print("\n" + "=" * 80)
    print(f"ImageNet {split} JSONL summary")
    print(f"Selected GT images: {len(resolved)}")
    print(f"Covered classes: {len(class_counts)}/1000")
    print(f"Images per class: min={min(class_counts.values())}, max={max(class_counts.values())}")
    print(f"Expected samples: {len(resolved) * len(DEGRADATIONS)}")
    print(f"Written samples: {samples}")
    print(f"Complete GT images: {complete_images}/{len(resolved)}")
    print(f"Invalid GT images: {invalid_gt}")
    print(f"Missing or invalid labels/XML files: {invalid_label}")
    print(f"Missing LQ images: {missing_lq}")
    print(f"Invalid LQ images: {invalid_lq}")
    print(f"Saved to: {output_path}")
    print("=" * 80)


def main():
    args = parse_args()
    root = Path(args.imagenet_root).resolve()
    output_dir = Path(args.out_dir).resolve()
    class_to_idx = build_class_to_idx(root / "GT" / "train")
    print(f"ImageNet classes from sorted training WNID folders: {len(class_to_idx)}")

    for split, filename, expected_images, expected_per_class in (
        ("train", args.train_name, args.expected_train_images, args.expected_train_per_class),
        ("val", args.val_name, args.expected_val_images, args.expected_val_per_class),
    ):
        write_imagenet_split(
            clean_root=root / "GT" / split,
            lq_root=root / "LQ" / split,
            annotation_root=root / "Annotations" / split,
            split=split,
            output_path=output_dir / filename,
            class_to_idx=class_to_idx,
            expected_images=expected_images,
            expected_per_class=expected_per_class,
            strict=args.strict,
        )


if __name__ == "__main__":
    main()
