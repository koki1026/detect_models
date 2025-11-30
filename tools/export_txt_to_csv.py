#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

from PIL import Image


def find_image_path(stem: str, images_dir: Path, exts=(".jpg", ".jpeg", ".png", ".bmp")):
    """stemから対応する画像パスを探す"""
    for ext in exts:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def export_gt_or_yolo_pred(txt_dir: Path, images_dir: Path, out_csv: Path, has_conf: bool):
    """
    GT or YOLO-style prediction:
      class x_center y_center width height [conf]
      すべて 0〜1 正規化
    """
    rows = []

    for txt_path in sorted(txt_dir.glob("*.txt")):
        stem = txt_path.stem
        img_path = find_image_path(stem, images_dir)
        if img_path is None:
            print(f"[WARN] image for {txt_path} not found, skip")
            continue

        with Image.open(img_path) as im:
            W, H = im.size

        with txt_path.open() as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue

                if (has_conf and len(parts) != 6) or (not has_conf and len(parts) != 5):
                    print(f"[WARN] unexpected format in {txt_path}: {line.strip()}")
                    continue

                cls_id = int(parts[0])
                x_c = float(parts[1])
                y_c = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
                conf = float(parts[5]) if has_conf else None

                # ピクセル座標もついでに計算
                x_c_px = x_c * W
                y_c_px = y_c * H
                w_px = w * W
                h_px = h * H

                row = {
                    "image_stem": stem,
                    "image_path": str(img_path),
                    "width_px": W,
                    "height_px": H,
                    "cls_id": cls_id,
                    "x_center": x_c,
                    "y_center": y_c,
                    "w": w,
                    "h": h,
                    "x_center_px": x_c_px,
                    "y_center_px": y_c_px,
                    "w_px": w_px,
                    "h_px": h_px,
                }
                if has_conf:
                    row["conf"] = conf

                rows.append(row)

    fieldnames = [
        "image_stem", "image_path", "width_px", "height_px",
        "cls_id",
        "x_center", "y_center", "w", "h",
        "x_center_px", "y_center_px", "w_px", "h_px",
    ]
    if has_conf:
        fieldnames.append("conf")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[INFO] wrote {len(rows)} rows to {out_csv}")


def export_sahi_pred(txt_dir: Path, images_dir: Path, out_csv: Path):
    """
    SAHI batch用:
      class conf x1 y1 x2 y2  (ピクセル座標)
    正規化xywhも一緒に書き出す
    """
    rows = []

    for txt_path in sorted(txt_dir.glob("*.txt")):
        stem = txt_path.stem
        # 例: "000221_sahi" → 画像stem "000221" だけ取りたい場合
        # 末尾の "_sahi" を消す
        if stem.endswith("_sahi"):
            img_stem = stem[:-5]
        else:
            img_stem = stem

        img_path = find_image_path(img_stem, images_dir)
        if img_path is None:
            print(f"[WARN] image for {txt_path} not found, skip")
            continue

        with Image.open(img_path) as im:
            W, H = im.size

        with txt_path.open() as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 6:
                    print(f"[WARN] unexpected format in {txt_path}: {line.strip()}")
                    continue

                cls_id = int(parts[0])
                conf = float(parts[1])
                x1 = float(parts[2])
                y1 = float(parts[3])
                x2 = float(parts[4])
                y2 = float(parts[5])

                # xywh (pixel)
                w_px = max(x2 - x1, 0.0)
                h_px = max(y2 - y1, 0.0)
                x_c_px = x1 + w_px / 2.0
                y_c_px = y1 + h_px / 2.0

                # 正規化 (0〜1)
                x_c = x_c_px / W
                y_c = y_c_px / H
                w = w_px / W
                h = h_px / H

                rows.append({
                    "image_stem": img_stem,
                    "image_path": str(img_path),
                    "width_px": W,
                    "height_px": H,
                    "cls_id": cls_id,
                    "conf": conf,
                    "x1_px": x1,
                    "y1_px": y1,
                    "x2_px": x2,
                    "y2_px": y2,
                    "x_center": x_c,
                    "y_center": y_c,
                    "w": w,
                    "h": h,
                    "x_center_px": x_c_px,
                    "y_center_px": y_c_px,
                    "w_px": w_px,
                    "h_px": h_px,
                })

    fieldnames = [
        "image_stem", "image_path", "width_px", "height_px",
        "cls_id", "conf",
        "x1_px", "y1_px", "x2_px", "y2_px",
        "x_center", "y_center", "w", "h",
        "x_center_px", "y_center_px", "w_px", "h_px",
    ]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[INFO] wrote {len(rows)} rows to {out_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["gt", "pred_yolo", "pred_sahi"], required=True,
                    help="gt=真値, pred_yolo=通常YOLO形式, pred_sahi=SAHI batch形式")
    ap.add_argument("--txt-dir", type=str, required=True,
                    help="*.txt が入っているディレクトリ")
    ap.add_argument("--images-dir", type=str, required=True,
                    help="対応する画像が入っているディレクトリ")
    ap.add_argument("--out-csv", type=str, required=True,
                    help="出力するCSVパス")
    args = ap.parse_args()

    txt_dir = Path(args.txt_dir)
    images_dir = Path(args.images_dir)
    out_csv = Path(args.out_csv)

    assert txt_dir.is_dir(), f"{txt_dir} is not dir"
    assert images_dir.is_dir(), f"{images_dir} is not dir"

    if args.mode == "gt":
        export_gt_or_yolo_pred(txt_dir, images_dir, out_csv, has_conf=False)
    elif args.mode == "pred_yolo":
        export_gt_or_yolo_pred(txt_dir, images_dir, out_csv, has_conf=True)
    elif args.mode == "pred_sahi":
        export_sahi_pred(txt_dir, images_dir, out_csv)
    else:
        raise ValueError(args.mode)


if __name__ == "__main__":
    main()