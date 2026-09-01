"""
Blitzball YOLO Dataset Splitter & Config Generator (prepare_dataset.py)

Splits labeled images and YOLO annotations into standard train/val/test splits,
verifies annotation integrity, generates the dataset's `data.yaml` configuration,
and optionally packages the dataset into a zip file ready for Google Colab / cloud training.

Usage Examples:
    # 1. Standard 80/20 train/val split
    python prepare_dataset.py

    # 2. Custom 70/20/10 train/val/test split with random seed
    python prepare_dataset.py --val-ratio 0.2 --test-ratio 0.1 --seed 42

    # 3. Split and package directly to dataset.zip for Colab upload
    python prepare_dataset.py --zip
"""

import argparse
import glob
import os
import random
import shutil
import sys
import zipfile
from typing import Dict, List, Optional, Tuple

try:
    import yaml
except ImportError:
    yaml = None


def find_labeled_pairs(
    images_dir: str,
    labels_dir: Optional[str] = None,
    include_background: bool = True,
    skip_unlabeled: bool = True,
) -> List[Dict[str, any]]:
    """
    Find matching (image, label) pairs.
    
    Returns list of dicts with keys:
    {
        'stem': str,
        'img_path': str,
        'lbl_path': str or None,
        'num_boxes': int,
        'is_background': bool
    }
    """
    lbl_dir = labels_dir if labels_dir else images_dir
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(images_dir, ext)))
    image_paths.sort()

    pairs = []
    for img_path in image_paths:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(lbl_dir, f"{stem}.txt")

        # Also check next to image if lbl_dir was separate
        if not os.path.exists(lbl_path):
            alt_lbl = os.path.join(os.path.dirname(img_path), f"{stem}.txt")
            if os.path.exists(alt_lbl):
                lbl_path = alt_lbl

        has_label_file = os.path.exists(lbl_path)

        if not has_label_file:
            if skip_unlabeled:
                continue
            pairs.append({
                "stem": stem,
                "img_path": img_path,
                "lbl_path": None,
                "num_boxes": 0,
                "is_background": True,
            })
            continue

        # Count valid YOLO bounding boxes in label file
        box_count = 0
        try:
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        box_count += 1
        except Exception as e:
            print(f"[Warning] Could not parse label file '{lbl_path}': {e}")
            continue

        is_bg = (box_count == 0)

        if is_bg and not include_background:
            continue

        pairs.append({
            "stem": stem,
            "img_path": img_path,
            "lbl_path": lbl_path,
            "num_boxes": box_count,
            "is_background": is_bg,
        })

    return pairs


def clean_yolo_directories(dataset_dir: str) -> None:
    """Create or reset standard YOLO dataset directories."""
    dirs_to_clean = [
        os.path.join(dataset_dir, "images", "train"),
        os.path.join(dataset_dir, "images", "val"),
        os.path.join(dataset_dir, "images", "test"),
        os.path.join(dataset_dir, "labels", "train"),
        os.path.join(dataset_dir, "labels", "val"),
        os.path.join(dataset_dir, "labels", "test"),
    ]

    for d in dirs_to_clean:
        if os.path.exists(d):
            # Remove existing split contents to prevent stale duplicates
            for f in glob.glob(os.path.join(d, "*")):
                try:
                    if os.path.isfile(f):
                        os.remove(f)
                    elif os.path.isdir(f):
                        shutil.rmtree(f)
                except Exception:
                    pass
        os.makedirs(d, exist_ok=True)


def generate_data_yaml(
    output_dir: str,
    class_names: Optional[Dict[int, str]] = None,
    has_test: bool = False,
    use_relative_path: bool = True,
) -> str:
    """Write standard YOLO data.yaml configuration file."""
    if class_names is None:
        class_names = {0: "blitzball"}

    yaml_path = os.path.join(output_dir, "data.yaml")

    # Format paths
    if use_relative_path:
        root_path = "."
    else:
        root_path = os.path.abspath(output_dir).replace("\\", "/")

    lines = [
        "# Blitzball YOLO Detection Dataset Configuration",
        f"path: {root_path}",
        "train: images/train",
        "val: images/val",
    ]

    if has_test:
        lines.append("test: images/test")

    lines.append("")
    lines.append("# Classes")
    lines.append("names:")
    for cid, name in sorted(class_names.items()):
        lines.append(f"  {cid}: {name}")
    lines.append("")

    content = "\n".join(lines)
    with open(yaml_path, "w") as f:
        f.write(content)

    return yaml_path


def create_dataset_zip(dataset_dir: str, zip_output_path: str = "dataset.zip") -> str:
    """Package dataset folder into a clean zip archive for Google Colab."""
    print(f"\n[Packaging] Compressing '{dataset_dir}' into '{zip_output_path}'...")
    with zipfile.ZipFile(zip_output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Include data.yaml
        yaml_file = os.path.join(dataset_dir, "data.yaml")
        if os.path.exists(yaml_file):
            zf.write(yaml_file, arcname="dataset/data.yaml")

        # Include images & labels splits
        for split_dir in ["images", "labels"]:
            base_split = os.path.join(dataset_dir, split_dir)
            if os.path.exists(base_split):
                for root, _, files in os.walk(base_split):
                    for file in files:
                        full_p = os.path.join(root, file)
                        rel_p = os.path.relpath(full_p, dataset_dir)
                        zf.write(full_p, arcname=os.path.join("dataset", rel_p))

    zip_size_mb = os.path.getsize(zip_output_path) / (1024 * 1024)
    print(f"[Done] Created '{zip_output_path}' ({zip_size_mb:.2f} MB). Ready for Google Colab upload!")
    return zip_output_path


def main():
    parser = argparse.ArgumentParser(
        description="Split Blitzball dataset into Train/Val splits and generate data.yaml config."
    )
    parser.add_argument(
        "--raw-images",
        type=str,
        default="dataset/raw_images",
        help="Path to folder with raw extracted images (default: 'dataset/raw_images').",
    )
    parser.add_argument(
        "--raw-labels",
        type=str,
        default=None,
        help="Path to folder with .txt labels (default: same as --raw-images).",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="dataset",
        help="Root YOLO dataset output directory (default: 'dataset').",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.20,
        help="Ratio of data for validation (default: 0.20 for 80/20 train/val).",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.0,
        help="Ratio of data for optional test set (default: 0.0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible dataset splits (default: 42).",
    )
    parser.add_argument(
        "--no-background",
        action="store_true",
        help="Exclude unannotated / empty background frames.",
    )
    parser.add_argument(
        "--zip",
        action="store_true",
        help="Automatically generate a dataset.zip archive for easy upload to Google Colab.",
    )

    args = parser.parse_args()

    # Discover pairs
    pairs = find_labeled_pairs(
        images_dir=args.raw_images,
        labels_dir=args.raw_labels,
        include_background=not args.no_background,
        skip_unlabeled=True,
    )

    if not pairs:
        print(f"\n[Error] No labeled samples found in '{args.raw_images}'.")
        print("Please annotate your extracted frames first using:")
        print("  python label_ball.py")
        sys.exit(1)

    # Shuffle deterministically
    random.seed(args.seed)
    random.shuffle(pairs)

    total_samples = len(pairs)
    val_count = int(total_samples * args.val_ratio)
    test_count = int(total_samples * args.test_ratio)
    train_count = total_samples - val_count - test_count

    # Partition slices
    val_set = pairs[:val_count]
    test_set = pairs[val_count : val_count + test_count]
    train_set = pairs[val_count + test_count :]

    splits = [
        ("train", train_set),
        ("val", val_set),
    ]
    if test_count > 0:
        splits.append(("test", test_set))

    # Setup directories
    clean_yolo_directories(args.output_dir)

    print("\n" + "=" * 60)
    print(" Splitting Blitzball YOLO Dataset")
    print("=" * 60)
    print(f"Total labeled pairs found : {total_samples}")
    print(f"Random seed               : {args.seed}")
    print(f"Train set                 : {train_count} images ({train_count/total_samples*100:4.1f}%)")
    print(f"Validation set            : {val_count} images ({val_count/total_samples*100:4.1f}%)")
    if test_count > 0:
        print(f"Test set                  : {test_count} images ({test_count/total_samples*100:4.1f}%)")
    print("=" * 60)

    # Copy files into respective train / val folders
    stats = {}
    for split_name, sample_list in splits:
        img_dest_dir = os.path.join(args.output_dir, "images", split_name)
        lbl_dest_dir = os.path.join(args.output_dir, "labels", split_name)

        ball_boxes = 0
        bg_frames = 0

        for item in sample_list:
            img_src = item["img_path"]
            lbl_src = item["lbl_path"]
            stem = item["stem"]
            ext = os.path.splitext(img_src)[1]

            img_dst = os.path.join(img_dest_dir, f"{stem}{ext}")
            lbl_dst = os.path.join(lbl_dest_dir, f"{stem}.txt")

            shutil.copy2(img_src, img_dst)

            if lbl_src and os.path.exists(lbl_src):
                shutil.copy2(lbl_src, lbl_dst)
            else:
                with open(lbl_dst, "w") as f:
                    pass

            ball_boxes += item["num_boxes"]
            if item["is_background"]:
                bg_frames += 1

        stats[split_name] = {
            "images": len(sample_list),
            "boxes": ball_boxes,
            "background": bg_frames,
        }

    # Generate data.yaml
    yaml_path = generate_data_yaml(
        output_dir=args.output_dir,
        class_names={0: "blitzball"},
        has_test=(test_count > 0),
        use_relative_path=True,
    )

    print(f"\n[Generated] {os.path.abspath(yaml_path)}")
    with open(yaml_path, "r") as f:
        print("-" * 35)
        print(f.read().strip())
        print("-" * 35)

    print("\n Split Breakdown:")
    for s_name, s_data in stats.items():
        print(f" - {s_name.upper():5s}: {s_data['images']:4d} images | {s_data['boxes']:4d} blitzball targets | {s_data['background']:3d} background frames")

    # Packaging zip archive if requested
    if args.zip:
        create_dataset_zip(args.output_dir, "dataset.zip")

    print("\n" + "=" * 60)
    print(" Dataset is fully configured and ready for training!")
    print("=" * 60)
    print("To train on Google Colab, open 'train_colab.ipynb' and run.")
    print("Or to train locally with Ultralytics:")
    print(f"  yolo detect train data={yaml_path} model=yolov8n.pt epochs=60 imgsz=640 batch=16\n")


if __name__ == "__main__":
    main()
