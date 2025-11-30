#!/usr/bin/env python3
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

import sys
from pathlib import Path

# batch_sahi_hard.py の1つ上の階層(/detect_models)をパスに追加
sys.path.append(str(Path(__file__).resolve().parents[1]))
from dbnet.sahi_infer_dbnet import sahi_infer, draw_detections  # さっき作ったやつをimport


# ★ ここを自分の環境に合わせて書き換え
IMAGES_DIR = Path("dataset/SCTD_yolo/runs/SCTD_train/sonar")
OUT_DIR = Path("runs_dbnet_yolo/exp3_sahi_hard_conf02")
WEIGHTS = "runs_dbnet_yolo/exp2_dual_backbone_v1/weights/best.pt"

# 予測が出なかった画像IDのリスト
HARD_IDS = [
    20, 44, 46, 60, 86, 88, 93, 97, 98, 99,
    101, 108, 110, 115, 125, 130, 160, 165, 181, 182,
    195, 221, 227, 240, 261, 283, 289, 312, 328, 363,
    389, 401, 409, 414, 431, 441, 447, 455, 461, 462,
    493,
]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(WEIGHTS)
    model.to(device)

    class_names = model.names

    for idx in HARD_IDS:
        stem = f"{idx:06d}"           # ← 000020 みたいに6桁ゼロ埋め
        img_path = IMAGES_DIR / f"{stem}.jpg"
        if not img_path.exists():
            print(f"[WARN] {img_path} not found, skip")
            continue

        print(f"[INFO] SAHI infer on {img_path}")
        boxes, scores, labels = sahi_infer(
            model=model,
            img_path=str(img_path),
            patch_size=768,   # 必要なら 768 など試してもOK
            overlap=0.5,      # hard例なので 0.5 とやや広めを推奨
            conf_thres=0.2,   # まずは recall 重視で0.2くらい
            iou_nms=0.6,
            imgsz=640,
        )

        img = cv2.imread(str(img_path))
        vis = draw_detections(img, boxes, scores, labels, class_names, conf_thres=0.2)

        out_img = OUT_DIR / f"{stem}_sahi.jpg"
        cv2.imwrite(str(out_img), vis)

        # 簡単なログもtxtで残す
        out_txt = OUT_DIR / f"{stem}_sahi.txt"
        with out_txt.open("w") as f:
            for b, s, c in zip(boxes, scores, labels):
                x1, y1, x2, y2 = b.tolist()
                f.write(f"{int(c)} {s:.4f} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}\n")

        print(f"  -> saved {out_img.name}, detections={len(boxes)}")


if __name__ == "__main__":
    main()