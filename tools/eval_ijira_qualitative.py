#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import yaml
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ultralytics import YOLO


# ----------------------------
# Utils: IO / Dataset
# ----------------------------

def load_data_yaml(data_yaml: Path):
    with open(data_yaml, "r") as f:
        y = yaml.safe_load(f)

    root = Path(y.get("path", "")).expanduser()
    names = y.get("names", None)

    if isinstance(names, dict):
        names = [names[i] for i in sorted(names.keys())]
    elif isinstance(names, list):
        pass
    else:
        raise ValueError("data.yaml に names が見つからない/形式が不正です。")

    return y, root, names


def resolve_split_paths(yaml_dict, root: Path, split: str):
    if split not in yaml_dict:
        raise ValueError(f"data.yaml に '{split}' がありません。train/val/test のどれかを指定してください。")

    img_rel = Path(yaml_dict[split])
    images_dir = (root / img_rel).resolve()

    # images/<split> → labels/<split> を想定
    if "images" in img_rel.parts:
        parts = list(img_rel.parts)
        parts[parts.index("images")] = "labels"
        labels_dir = (root / Path(*parts)).resolve()
    else:
        labels_dir = (root / "labels" / split).resolve()

    return images_dir, labels_dir


def list_images(images_dir: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    imgs = [p for p in images_dir.rglob("*") if p.suffix.lower() in exts]
    imgs.sort()
    return imgs


def read_image(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"画像が読めません: {path}")
    return img


# ----------------------------
# GT loader (YOLO txt)
# ----------------------------

def yolo_txt_to_xyxy(label_line, w, h):
    cls, cx, cy, bw, bh = label_line
    x1 = (cx - bw / 2.0) * w
    y1 = (cy - bh / 2.0) * h
    x2 = (cx + bw / 2.0) * w
    y2 = (cy + bh / 2.0) * h
    return int(cls), [x1, y1, x2, y2]


def load_gt_yolo(labels_dir: Path, img_path: Path, img_w: int, img_h: int):
    label_path = labels_dir / (img_path.stem + ".txt")
    gts = []
    if not label_path.exists():
        return gts

    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            cx, cy, bw, bh = map(float, parts[1:5])
            cls_id, xyxy = yolo_txt_to_xyxy([cls, cx, cy, bw, bh], img_w, img_h)
            gts.append({
                "class_id": cls_id,
                "bbox": xyxy,
                "score": 1.0,
            })
    return gts


# ----------------------------
# Predictions via Ultralytics (YOLOv8n / SSYolo / DBNet)
# ----------------------------

def predict_ultralytics(model: YOLO, img_bgr: np.ndarray, conf: float, iou_nms: float):
    results = model.predict(
        source=img_bgr,
        conf=conf,
        iou=iou_nms,
        verbose=False
    )
    out = []
    if not results:
        return out
    r = results[0]
    if r.boxes is None:
        return out

    boxes = r.boxes
    xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else boxes.xyxy.numpy()
    cls = boxes.cls.cpu().numpy().astype(int) if hasattr(boxes.cls, "cpu") else boxes.cls.numpy().astype(int)
    confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else boxes.conf.numpy()

    for c, b, s in zip(cls, xyxy, confs):
        out.append({
            "class_id": int(c),
            "bbox": [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
            "score": float(s),
        })
    return out


# ----------------------------
# Matching / Metrics
# ----------------------------

def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def greedy_match(gt_list, pred_list, iou_thr: float, require_same_class=True):
    unmatched_gt = set(range(len(gt_list)))
    unmatched_pred = set(range(len(pred_list)))
    matches = []

    pred_order = sorted(range(len(pred_list)), key=lambda i: pred_list[i]["score"], reverse=True)

    for pi in pred_order:
        best_gi = None
        best_iou = 0.0
        for gi in list(unmatched_gt):
            if require_same_class and gt_list[gi]["class_id"] != pred_list[pi]["class_id"]:
                continue
            iou = iou_xyxy(gt_list[gi]["bbox"], pred_list[pi]["bbox"])
            if iou >= iou_thr and iou > best_iou:
                best_iou = iou
                best_gi = gi
        if best_gi is not None:
            matches.append((best_gi, pi, best_iou))
            unmatched_gt.remove(best_gi)
            unmatched_pred.remove(pi)

    return matches, unmatched_gt, unmatched_pred


# ----------------------------
# Visualization
# ----------------------------

def draw_boxes(img_bgr, boxes, names, color, thickness=2, show_score=True, prefix=""):
    out = img_bgr.copy()
    for b in boxes:
        cls_id = b["class_id"]
        x1, y1, x2, y2 = map(int, b["bbox"])
        label = f"{prefix}{names[cls_id] if cls_id < len(names) else cls_id}"
        if show_score:
            label += f" {b.get('score', 0.0):.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(out, label, (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    return out


def annotate_image_tp_fp_fn(img_bgr, gt, preds, matches, unmatched_gt, unmatched_pred, names):
    tp_pred_idx = set([m[1] for m in matches])
    tp_preds = [preds[i] for i in sorted(tp_pred_idx)]
    fp_preds = [preds[i] for i in sorted(unmatched_pred)]
    fn_gts = [gt[i] for i in sorted(unmatched_gt)]

    out = img_bgr.copy()
    out = draw_boxes(out, tp_preds, names, color=(0, 255, 0), show_score=True, prefix="TP:")
    out = draw_boxes(out, fp_preds, names, color=(0, 0, 255), show_score=True, prefix="FP:")
    out = draw_boxes(out, fn_gts, names, color=(255, 0, 0), show_score=False, prefix="FN_GT:")
    return out


def make_compare_panel(img_gt, img_yolo, img_ssyolo, img_dbnet, titles=("GT", "YOLOv8n", "SSYolo", "DBNet")):
    def put_title(im, t):
        out = im.copy()
        cv2.putText(out, t, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, t, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
        return out

    imgs = [img_gt, img_yolo, img_ssyolo, img_dbnet]
    h = min([im.shape[0] for im in imgs])
    w = min([im.shape[1] for im in imgs])
    imgs = [cv2.resize(im, (w, h)) for im in imgs]
    imgs = [put_title(im, t) for im, t in zip(imgs, titles)]

    top = cv2.hconcat([imgs[0], imgs[1]])
    bot = cv2.hconcat([imgs[2], imgs[3]])
    panel = cv2.vconcat([top, bot])
    return panel


# ----------------------------
# Scatter: bbox size vs success
# ----------------------------

def bbox_area(b):
    x1, y1, x2, y2 = b
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def save_scatter(per_gt_records: pd.DataFrame, out_path, title: str):
    plt.figure()
    for model_name, dfm in per_gt_records.groupby("model"):
        x = dfm["area"].values
        y = dfm["success"].values
        plt.scatter(x, y, label=model_name, alpha=0.6)
    plt.xlabel("GT bbox area (px^2)")
    plt.ylabel("Detected (1) / Missed (0)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_yaml", type=str, required=True)
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--yolo8_weights", type=str, required=True)
    ap.add_argument("--ssyolo_weights", type=str, required=True)
    ap.add_argument("--dbnet_weights", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou_nms", type=float, default=0.60)
    ap.add_argument("--iou_match", type=float, default=0.50)
    ap.add_argument("--compare_scope", type=str, default="fn", choices=["fn", "all"])
    ap.add_argument("--max_compare", type=int, default=0, help="0 = no limit")
    args = ap.parse_args()

    data_yaml = Path(args.data_yaml).expanduser().resolve()
    y, root, names = load_data_yaml(data_yaml)
    images_dir, labels_dir = resolve_split_paths(y, root, args.split)

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    qual_dir = out_dir / "qualitative"
    fn_dir = out_dir / "fn_only"
    compare_dir = out_dir / "compare"
    scatter_dir = out_dir / "scatter"
    for d in [qual_dir, fn_dir, compare_dir, scatter_dir]:
        d.mkdir(parents=True, exist_ok=True)

    model_dirs = ["Yolov8n", "SSYolo", "DBNet"]
    for mn in model_dirs:
        (qual_dir / mn).mkdir(parents=True, exist_ok=True)
        (fn_dir / mn).mkdir(parents=True, exist_ok=True)

    # Load models
    yolo8 = YOLO(str(Path(args.yolo8_weights).expanduser()))
    ssyolo = YOLO(str(Path(args.ssyolo_weights).expanduser()))
    dbnet = YOLO(str(Path(args.dbnet_weights).expanduser()))

    images = list_images(images_dir)
    if not images:
        raise RuntimeError(f"画像が見つかりません: {images_dir}")

    rows_target = []
    rows_image = []
    per_gt_scatter_rows = []

    compare_count = 0

    for img_path in images:
        img = read_image(img_path)
        h, w = img.shape[:2]
        gt = load_gt_yolo(labels_dir, img_path, w, h)

        preds_y8 = predict_ultralytics(yolo8, img, conf=args.conf, iou_nms=args.iou_nms)
        preds_ss = predict_ultralytics(ssyolo, img, conf=args.conf, iou_nms=args.iou_nms)
        preds_db = predict_ultralytics(dbnet, img, conf=args.conf, iou_nms=args.iou_nms)

        m_y8, ug_y8, up_y8 = greedy_match(gt, preds_y8, iou_thr=args.iou_match, require_same_class=True)
        m_ss, ug_ss, up_ss = greedy_match(gt, preds_ss, iou_thr=args.iou_match, require_same_class=True)
        m_db, ug_db, up_db = greedy_match(gt, preds_db, iou_thr=args.iou_match, require_same_class=True)

        ann_y8 = annotate_image_tp_fp_fn(img, gt, preds_y8, m_y8, ug_y8, up_y8, names)
        ann_ss = annotate_image_tp_fp_fn(img, gt, preds_ss, m_ss, ug_ss, up_ss, names)
        ann_db = annotate_image_tp_fp_fn(img, gt, preds_db, m_db, ug_db, up_db, names)

        cv2.imwrite(str((qual_dir / "Yolov8n" / f"{img_path.stem}.png")), ann_y8)
        cv2.imwrite(str((qual_dir / "SSYolo" / f"{img_path.stem}.png")), ann_ss)
        cv2.imwrite(str((qual_dir / "DBNet" / f"{img_path.stem}.png")), ann_db)

        fn_y8 = len(ug_y8) > 0
        fn_ss = len(ug_ss) > 0
        fn_db = len(ug_db) > 0

        if fn_y8:
            cv2.imwrite(str((fn_dir / "Yolov8n" / f"{img_path.stem}.png")), ann_y8)
        if fn_ss:
            cv2.imwrite(str((fn_dir / "SSYolo" / f"{img_path.stem}.png")), ann_ss)
        if fn_db:
            cv2.imwrite(str((fn_dir / "DBNet" / f"{img_path.stem}.png")), ann_db)

        def hit_all(gt_list, unmatched_gt_set):
            if len(gt_list) == 0:
                return True
            return len(unmatched_gt_set) == 0

        rows_image += [
            {"model": "Yolov8n", "image": img_path.name, "has_gt": int(len(gt) > 0), "hit_all_gt": int(hit_all(gt, ug_y8)), "miss_any_gt": int(len(ug_y8) > 0)},
            {"model": "SSYolo",  "image": img_path.name, "has_gt": int(len(gt) > 0), "hit_all_gt": int(hit_all(gt, ug_ss)), "miss_any_gt": int(len(ug_ss) > 0)},
            {"model": "DBNet",   "image": img_path.name, "has_gt": int(len(gt) > 0), "hit_all_gt": int(hit_all(gt, ug_db)), "miss_any_gt": int(len(ug_db) > 0)},
        ]

        def tp_fp_fn_counts(gt_list, pred_list, matches, unmatched_gt_set, unmatched_pred_set):
            tp = []
            fp = []
            fn = []
            for gi, pi, _ in matches:
                tp.append(gt_list[gi]["class_id"])
            for pi in unmatched_pred_set:
                fp.append(pred_list[pi]["class_id"])
            for gi in unmatched_gt_set:
                fn.append(gt_list[gi]["class_id"])
            return tp, fp, fn

        for model_label, preds, matches, ug, up in [
            ("Yolov8n", preds_y8, m_y8, ug_y8, up_y8),
            ("SSYolo",  preds_ss, m_ss, ug_ss, up_ss),
            ("DBNet",   preds_db, m_db, ug_db, up_db),
        ]:
            tp_cls, fp_cls, fn_cls = tp_fp_fn_counts(gt, preds, matches, ug, up)

            for cls_id in range(len(names)):
                rows_target.append({
                    "model": model_label,
                    "image": img_path.name,
                    "class_id": cls_id,
                    "class_name": names[cls_id],
                    "TP": int(tp_cls.count(cls_id)),
                    "FP": int(fp_cls.count(cls_id)),
                    "FN": int(fn_cls.count(cls_id)),
                })

            matched_gt = set([gi for gi, _, _ in matches])
            for gi, g in enumerate(gt):
                per_gt_scatter_rows.append({
                    "model": model_label,
                    "image": img_path.name,
                    "class_id": g["class_id"],
                    "class_name": names[g["class_id"]] if g["class_id"] < len(names) else str(g["class_id"]),
                    "area": bbox_area(g["bbox"]),
                    "success": 1 if gi in matched_gt else 0
                })

        gt_only = draw_boxes(img, gt, names, color=(255, 255, 255), show_score=False, prefix="GT:")

        need_compare = (args.compare_scope == "all") or (fn_y8 or fn_ss or fn_db)
        if need_compare and (args.max_compare <= 0 or compare_count < args.max_compare):
            panel = make_compare_panel(
                gt_only, ann_y8, ann_ss, ann_db,
                titles=("GT", "Yolov8n", "SSYolo", "DBNet")
            )
            (compare_dir / img_path.stem).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(compare_dir / img_path.stem / "panel.png"), panel)
            cv2.imwrite(str(compare_dir / img_path.stem / "gt.png"), gt_only)
            cv2.imwrite(str(compare_dir / img_path.stem / "yolov8n.png"), ann_y8)
            cv2.imwrite(str(compare_dir / img_path.stem / "ssyolo.png"), ann_ss)
            cv2.imwrite(str(compare_dir / img_path.stem / "dbnet.png"), ann_db)
            compare_count += 1

    df_target = pd.DataFrame(rows_target)
    df_image = pd.DataFrame(rows_image)
    df_gt = pd.DataFrame(per_gt_scatter_rows)

    df_target.to_csv(out_dir / "per_image_per_class_counts.csv", index=False)
    df_image.to_csv(out_dir / "per_image_hit_miss.csv", index=False)
    df_gt.to_csv(out_dir / "per_gt_scatter_records.csv", index=False)

    df_summary = (df_target
                  .groupby(["model", "class_id", "class_name"], as_index=False)[["TP", "FP", "FN"]]
                  .sum())
    df_summary.to_csv(out_dir / "summary_per_class.csv", index=False)

    df_img_sum = (df_image
                  .groupby("model", as_index=False)[["has_gt", "hit_all_gt", "miss_any_gt"]]
                  .sum())
    df_img_sum["total_images"] = df_image.groupby("model").size().values
    df_img_sum.to_csv(out_dir / "summary_per_image.csv", index=False)

    scatter_dir.mkdir(parents=True, exist_ok=True)
    save_scatter(df_gt, scatter_dir / "area_vs_success_all.png", "GT bbox area vs detection success (all classes)")
    for cls_name, dfc in df_gt.groupby("class_name"):
        safe = "".join([c if c.isalnum() or c in "_-" else "_" for c in cls_name])
        save_scatter(dfc, scatter_dir / f"area_vs_success_{safe}.png", f"GT bbox area vs success ({cls_name})")

    print("Done.")
    print(f"Outputs saved to: {out_dir}")
    print("- qualitative/: all images annotated per model")
    print("- fn_only/: FN images annotated per model")
    print("- compare/: compare panels (and per-model images)")
    print("- scatter/: area vs success plots")
    print("- CSV: summary_per_class.csv, summary_per_image.csv, ...")


if __name__ == "__main__":
    main()
