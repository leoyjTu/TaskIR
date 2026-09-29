#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


DEGRADATIONS = (
    "gaussian_noise", "shot_noise", "defocus_blur", "motion_blur",
    "snow", "fog", "brightness", "jpeg_compression",
)


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
        description="Generate strict Cityscapes TaskIR segmentation JSONL files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--cityscapes_root",
        default=str(PROJECT_ROOT / "datasets" / "Cityscapes"),
        help="Cityscapes root containing GT, annotations and LQ.",
    )
    parser.add_argument(
        "--out_dir",
        default=str(PROJECT_ROOT / "data_dir" / "lists"),
        help="Directory for train_seg.jsonl and val_seg.jsonl.",
    )
    parser.add_argument(
        "--train_name", default="train_seg.jsonl", help="Training JSONL filename.",
    )
    parser.add_argument(
        "--val_name", default="val_seg.jsonl", help="Validation JSONL filename.",
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
    images = sorted(
        path for path in root.rglob("*_leftImg8bit.png") if path.is_file()
    )
    if not images:
        raise RuntimeError(f"No Cityscapes GT images found under: {root}")
    return images


def annotation_path_for(clean_path, image_root, annotation_root):
    relative_path = clean_path.relative_to(image_root)
    annotation_name = relative_path.name.replace(
        "_leftImg8bit.png", "_gtFine_labelTrainIds.png",
    )
    return annotation_root / relative_path.parent / annotation_name


def image_size(path):
    with Image.open(path) as image:
        image.load()
        return image.size


def validate_cityscapes_mask(mask_path, expected_size):
    with Image.open(mask_path) as mask:
        mask.load()
        mask_size = mask.size
        array = np.asarray(mask)
    if mask_size != expected_size:
        raise ValueError(
            f"Cityscapes GT/mask size mismatch: {expected_size} vs {mask_size}; {mask_path}"
        )
    if array.ndim != 2:
        raise ValueError(f"Cityscapes labelTrainIds mask must be single-channel: {mask_path}")
    valid = ((array >= 0) & (array <= 18)) | (array == 255)
    if not valid.all():
        values = np.unique(array[~valid])[:10].tolist()
        raise ValueError(f"Invalid Cityscapes trainIds {values}: {mask_path}")


def handle_problem(message, strict):
    if strict:
        raise RuntimeError(message)
    print(message)


def write_cityscapes_split(
    image_root, annotation_root, lq_root, split, output_path, strict=True,
):
    image_root = Path(image_root).resolve()
    annotation_root = Path(annotation_root).resolve()
    lq_root = Path(lq_root).resolve()
    output_path = Path(output_path).resolve()

    images = collect_images(image_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    samples = used_images = invalid_gt = missing_annotations = invalid_annotations = 0
    missing_lq = invalid_lq = 0

    try:
        with temporary_path.open("w", encoding="utf-8") as output:
            for clean_path in tqdm(images, desc=f"Building Cityscapes {split} JSONL"):
                relative_path = clean_path.relative_to(image_root)
                annotation_path = annotation_path_for(
                    clean_path, image_root, annotation_root,
                )
                if not annotation_path.is_file():
                    missing_annotations += 1
                    handle_problem(f"Missing Cityscapes annotation: {annotation_path}", strict)
                    continue
                try:
                    clean_size = image_size(clean_path)
                except OSError as error:
                    invalid_gt += 1
                    handle_problem(f"Unreadable Cityscapes GT image: {clean_path}; {error}", strict)
                    continue
                try:
                    validate_cityscapes_mask(annotation_path, clean_size)
                except (OSError, ValueError) as error:
                    invalid_annotations += 1
                    handle_problem(str(error), strict)
                    continue

                written_for_image = 0
                for degradation in DEGRADATIONS:
                    lq_path = lq_root / degradation / relative_path
                    if not lq_path.is_file():
                        missing_lq += 1
                        handle_problem(f"Missing Cityscapes LQ image: {lq_path}", strict)
                        continue
                    try:
                        lq_size = image_size(lq_path)
                    except OSError as error:
                        invalid_lq += 1
                        handle_problem(f"Unreadable Cityscapes LQ image: {lq_path}; {error}", strict)
                        continue
                    if lq_size != clean_size:
                        invalid_lq += 1
                        handle_problem(
                            f"Cityscapes GT/LQ size mismatch: {clean_size} vs {lq_size}; {lq_path}",
                            strict,
                        )
                        continue

                    record = {
                        "task": "seg",
                        "lq": str(lq_path.resolve()),
                        "gt": str(clean_path.resolve()),
                        "ann": str(annotation_path.resolve()),
                        "image_id": clean_path.stem,
                        "mask_format": "cityscapes_train_ids",
                        "deg": degradation,
                    }
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    samples += 1
                    written_for_image += 1
                if written_for_image:
                    used_images += 1
        temporary_path.replace(output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    print("\n" + "=" * 80)
    print(f"Cityscapes {split} JSONL summary")
    print(f"GT images: {len(images)}")
    print(f"Used GT images: {used_images}")
    print(f"Expected samples: {len(images) * len(DEGRADATIONS)}")
    print(f"Written samples: {samples}")
    print(f"Invalid GT images: {invalid_gt}")
    print(f"Missing annotations: {missing_annotations}")
    print(f"Invalid annotations: {invalid_annotations}")
    print(f"Missing LQ images: {missing_lq}")
    print(f"Invalid LQ images: {invalid_lq}")
    print(f"Saved to: {output_path}")
    print("=" * 80)


def main():
    args = parse_args()
    root = Path(args.cityscapes_root).resolve()
    output_dir = Path(args.out_dir).resolve()
    for split, filename in (("train", args.train_name), ("val", args.val_name)):
        write_cityscapes_split(
            image_root=root / "GT" / split,
            annotation_root=root / "annotations" / split,
            lq_root=root / "LQ" / split,
            split=split,
            output_path=output_dir / filename,
            strict=args.strict,
        )


if __name__ == "__main__":
    main()
