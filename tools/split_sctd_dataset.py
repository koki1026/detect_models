#!/usr/bin/env python3
import os
import shutil
import random
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="source dataset path")
    parser.add_argument("--dst", required=True, help="output dataset path")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    args = parser.parse_args()

    src_path = Path(args.src)
    dst_path = Path(args.dst)

    labels_src = src_path / "labels"
    images_src = src_path / "sonar"

    assert labels_src.exists(), f"Not found: {labels_src}"
    assert images_src.exists(), f"Not found: {images_src}"

    # 出力ディレクトリ作成
    for d in [
        dst_path / "images" / "train",
        dst_path / "images" / "val",
        dst_path / "labels" / "train",
        dst_path / "labels" / "val",
    ]:
        d.mkdir(parents=True, exist_ok=True)

    label_paths = sorted(labels_src.glob("*.txt"))
    random.seed(42)
    random.shuffle(label_paths)

    num_val = int(len(label_paths) * args.val_ratio)

    val_set = label_paths[:num_val]
    train_set = label_paths[num_val:]

    def move_files(subset, subset_name):
        for label_path in subset:
            base = label_path.stem
            img_path_jpg = images_src / f"{base}.jpg"
            img_path_png = images_src / f"{base}.png"

            if img_path_jpg.exists():
                img_src = img_path_jpg
            elif img_path_png.exists():
                img_src = img_path_png
            else:
                print(f"Warning: image not found for {base}")
                continue

            label_dst = dst_path / "labels" / subset_name / label_path.name
            img_dst = dst_path / "images" / subset_name / img_src.name

            shutil.copy(label_path, label_dst)
            shutil.copy(img_src, img_dst)

    move_files(train_set, "train")
    move_files(val_set, "val")

    print(f"Done! Train={len(train_set)}  Val={len(val_set)}")
    print("Output saved to:", dst_path)


if __name__ == "__main__":
    main()