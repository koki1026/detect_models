#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from torchvision.ops import batched_nms


def sahi_infer(
    model: YOLO,
    img_path: str,
    patch_size: int = 640,
    overlap: float = 0.25,
    conf_thres: float = 0.25,
    iou_nms: float = 0.5,
    imgsz: int = 640,
):
    """SAHI: 大きな画像をスライスして推論し、NMSで統合した結果を返す"""
    img_path = Path(img_path)
    assert img_path.is_file()

    img_bgr = cv2.imread(str(img_path))
    assert img_bgr is not None, f"failed to read {img_path}"
    H, W = img_bgr.shape[:2]

    step = int(patch_size * (1.0 - overlap))
    step = max(1, step)

    all_boxes = []
    all_scores = []
    all_labels = []

    for y0 in range(0, H, step):
        y1 = min(y0 + patch_size, H)
        if y1 <= y0:
            continue

        for x0 in range(0, W, step):
            x1 = min(x0 + patch_size, W)
            if x1 <= x0:
                continue

            patch = img_bgr[y0:y1, x0:x1]  # (h,w,3) BGR
            if patch.size == 0:
                continue

            # Ultralytics は内部でレターボックスしてくれるのでそのまま渡す
            results = model.predict(
                source=patch,
                imgsz=imgsz,
                conf=conf_thres,
                iou=0.7,          # 学習時と同じIOU設定でもよい
                verbose=False,
            )

            r = results[0]
            if r.boxes is None or r.boxes.shape[0] == 0:
                continue

            boxes = r.boxes.xyxy.cpu()  # (N,4)
            scores = r.boxes.conf.cpu()  # (N,)
            labels = r.boxes.cls.cpu().to(torch.int64)  # (N,)

            # パッチ座標 -> 元画像座標へ平行移動
            boxes[:, [0, 2]] += x0
            boxes[:, [1, 3]] += y0

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

    if len(all_boxes) == 0:
        return (
            torch.zeros((0, 4)),
            torch.zeros((0,)),
            torch.zeros((0,), dtype=torch.int64),
        )

    boxes = torch.cat(all_boxes, dim=0)
    scores = torch.cat(all_scores, dim=0)
    labels = torch.cat(all_labels, dim=0)

    keep = batched_nms(boxes, scores, labels, iou_nms)
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    return boxes, scores, labels


def draw_detections(img_bgr, boxes, scores, labels, class_names, conf_thres=0.25):
    img = img_bgr.copy()
    for box, score, cls in zip(boxes, scores, labels):
        if score < conf_thres:
            continue
        x1, y1, x2, y2 = box.int().tolist()
        c = int(cls.item())
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
        text = f"{class_names[c]} {score:.2f}"
        cv2.putText(img, text, (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--weights",
        default="runs_dbnet_yolo/exp2_dual_backbone_v1/weights/best.pt",
        help="学習済みDBnet-YOLOの重み",
    )
    ap.add_argument("--source", required=True, help="入力画像（大きなSSS画像）")
    ap.add_argument("--patch-size", type=int, default=640)
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou-nms", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--save-vis", action="store_true")
    ap.add_argument("--save-path", default="sahi_out.jpg")
    args = ap.parse_args()

    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(args.weights)
    model.to(device)

    boxes, scores, labels = sahi_infer(
        model=model,
        img_path=args.source,
        patch_size=args.patch_size,
        overlap=args.overlap,
        conf_thres=args.conf,
        iou_nms=args.iou_nms,
        imgsz=args.imgsz,
    )

    print(f"SAHI detections: {len(boxes)}")
    for b, s, c in zip(boxes, scores, labels):
        print(b.tolist(), float(s), int(c))

    if args.save_vis:
        img = cv2.imread(args.source)
        class_names = model.names
        vis = draw_detections(img, boxes, scores, labels, class_names, conf_thres=args.conf)
        cv2.imwrite(args.save_path, vis)
        print(f"saved visualization to {args.save_path}")


if __name__ == "__main__":
    main()