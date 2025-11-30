#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def cxcywh_to_xyxy(cx, cy, w, h):
    """中心座標+幅高さ(px) -> x1,y1,x2,y2(px)"""
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return np.stack([x1, y1, x2, y2], axis=-1)


def boxes_iou_matrix(boxes1, boxes2):
    """
    boxes1: (N,4), boxes2: (M,4) in xyxy
    return: (N,M) IoU matrix
    """
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)

    b1 = boxes1[:, None, :]  # (N,1,4)
    b2 = boxes2[None, :, :]  # (1,M,4)

    x1 = np.maximum(b1[..., 0], b2[..., 0])
    y1 = np.maximum(b1[..., 1], b2[..., 1])
    x2 = np.minimum(b1[..., 2], b2[..., 2])
    y2 = np.minimum(b1[..., 3], b2[..., 3])

    inter_w = np.clip(x2 - x1, a_min=0.0, a_max=None)
    inter_h = np.clip(y2 - y1, a_min=0.0, a_max=None)
    inter = inter_w * inter_h

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    union = area1[:, None] + area2[None, :] - inter
    iou = np.zeros_like(inter)
    mask = union > 0
    iou[mask] = inter[mask] / union[mask]
    return iou


def greedy_match(iou_mat, thr):
    """
    N_gt x N_pred の IoU 行列から貪欲にマッチングを決める
    return: list of (gt_idx, pred_idx, iou)
    """
    matches = []
    if iou_mat.size == 0:
        return matches

    iou = iou_mat.copy()
    while True:
        max_idx = np.unravel_index(np.argmax(iou), iou.shape)
        max_iou = iou[max_idx]
        if max_iou < thr:
            break
        gi, pi = max_idx
        matches.append((gi, pi, float(max_iou)))
        iou[gi, :] = -1.0
        iou[:, pi] = -1.0
    return matches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-csv", required=True, help="gt_labels.csv")
    ap.add_argument("--dbnet-csv", required=True, help="pred_dbnet.csv")
    ap.add_argument("--sahi-csv", required=True, help="pred_sahi_hard.csv")
    ap.add_argument("--iou-thr", type=float, default=0.4, help="IoU threshold")
    ap.add_argument("--out-prefix", required=True, help="出力ファイルのprefix（例: analysis/compare_iou04）")
    args = ap.parse_args()

    gt_df = pd.read_csv(args.gt_csv)
    db_df = pd.read_csv(args.dbnet_csv)
    sahi_df = pd.read_csv(args.sahi_csv)

    # xyxy(px)を用意
    gt_boxes = cxcywh_to_xyxy(gt_df["x_center_px"].values,
                              gt_df["y_center_px"].values,
                              gt_df["w_px"].values,
                              gt_df["h_px"].values)
    gt_df["x1_px"] = gt_boxes[:, 0]
    gt_df["y1_px"] = gt_boxes[:, 1]
    gt_df["x2_px"] = gt_boxes[:, 2]
    gt_df["y2_px"] = gt_boxes[:, 3]

    db_boxes = cxcywh_to_xyxy(db_df["x_center_px"].values,
                              db_df["y_center_px"].values,
                              db_df["w_px"].values,
                              db_df["h_px"].values)
    db_df["x1_px"] = db_boxes[:, 0]
    db_df["y1_px"] = db_boxes[:, 1]
    db_df["x2_px"] = db_boxes[:, 2]
    db_df["y2_px"] = db_boxes[:, 3]

    # SAHIはすでにx1_px...を持っている前提
    # 念のため無ければ作る（中心+whから）
    if not {"x1_px", "y1_px", "x2_px", "y2_px"}.issubset(sahi_df.columns):
        sahi_boxes = cxcywh_to_xyxy(sahi_df["x_center_px"].values,
                                    sahi_df["y_center_px"].values,
                                    sahi_df["w_px"].values,
                                    sahi_df["h_px"].values)
        sahi_df["x1_px"] = sahi_boxes[:, 0]
        sahi_df["y1_px"] = sahi_boxes[:, 1]
        sahi_df["x2_px"] = sahi_boxes[:, 2]
        sahi_df["y2_px"] = sahi_boxes[:, 3]

    # 画像ごと・クラスごとにマッチング
    classes = sorted(gt_df["cls_id"].unique().tolist())
    all_image_stems = sorted(gt_df["image_stem"].unique().tolist())

    per_img_rows = []
    global_stats = {c: {"gt": 0, "tp_dbnet": 0, "fp_dbnet": 0, "fn_dbnet": 0,
                        "tp_sahi": 0, "fp_sahi": 0, "fn_sahi": 0} for c in classes}

    for img in all_image_stems:
        for cls in classes:
            gt_sub = gt_df[(gt_df["image_stem"] == img) & (gt_df["cls_id"] == cls)].copy()
            db_sub = db_df[(db_df["image_stem"] == img) & (db_df["cls_id"] == cls)].copy()
            sahi_sub = sahi_df[(sahi_df["image_stem"] == img) & (sahi_df["cls_id"] == cls)].copy()

            n_gt = len(gt_sub)
            idx_gt = np.arange(n_gt)

            # DBnet マッチング
            db_boxes = db_sub[["x1_px", "y1_px", "x2_px", "y2_px"]].values
            gt_boxes = gt_sub[["x1_px", "y1_px", "x2_px", "y2_px"]].values
            iou_db = boxes_iou_matrix(gt_boxes, db_boxes)
            matches_db = greedy_match(iou_db, args.iou_thr)
            tp_db = len(matches_db)
            matched_gt_db = set(m[0] for m in matches_db)
            matched_db_idx = set(m[1] for m in matches_db)
            fn_db = n_gt - tp_db
            fp_db = len(db_sub) - tp_db

            # SAHI マッチング
            sahi_boxes = sahi_sub[["x1_px", "y1_px", "x2_px", "y2_px"]].values
            iou_sh = boxes_iou_matrix(gt_boxes, sahi_boxes)
            matches_sh = greedy_match(iou_sh, args.iou_thr)
            tp_sh = len(matches_sh)
            matched_gt_sh = set(m[0] for m in matches_sh)
            matched_sh_idx = set(m[1] for m in matches_sh)
            fn_sh = n_gt - tp_sh
            fp_sh = len(sahi_sub) - tp_sh

            # per-image-row
            per_img_rows.append({
                "image_stem": img,
                "cls_id": cls,
                "n_gt": n_gt,
                "dbnet_n_pred": len(db_sub),
                "dbnet_tp": tp_db,
                "dbnet_fp": fp_db,
                "dbnet_fn": fn_db,
                "sahi_n_pred": len(sahi_sub),
                "sahi_tp": tp_sh,
                "sahi_fp": fp_sh,
                "sahi_fn": fn_sh,
            })

            # global stats 更新
            g = global_stats[cls]
            g["gt"] += n_gt
            g["tp_dbnet"] += tp_db
            g["fp_dbnet"] += fp_db
            g["fn_dbnet"] += fn_db
            g["tp_sahi"] += tp_sh
            g["fp_sahi"] += fp_sh
            g["fn_sahi"] += fn_sh

    per_img_df = pd.DataFrame(per_img_rows)

    # クラスごとの集計
    summary_rows = []
    for cls, st in global_stats.items():
        gt_n = st["gt"]
        tp_db, fp_db, fn_db = st["tp_dbnet"], st["fp_dbnet"], st["fn_dbnet"]
        tp_sh, fp_sh, fn_sh = st["tp_sahi"], st["fp_sahi"], st["fn_sahi"]

        prec_db = tp_db / (tp_db + fp_db) if (tp_db + fp_db) > 0 else 0.0
        rec_db = tp_db / gt_n if gt_n > 0 else 0.0

        prec_sh = tp_sh / (tp_sh + fp_sh) if (tp_sh + fp_sh) > 0 else 0.0
        rec_sh = tp_sh / gt_n if gt_n > 0 else 0.0

        summary_rows.append({
            "cls_id": cls,
            "gt": gt_n,
            "dbnet_tp": tp_db,
            "dbnet_fp": fp_db,
            "dbnet_fn": fn_db,
            "dbnet_precision": prec_db,
            "dbnet_recall": rec_db,
            "sahi_tp": tp_sh,
            "sahi_fp": fp_sh,
            "sahi_fn": fn_sh,
            "sahi_precision": prec_sh,
            "sahi_recall": rec_sh,
        })

    summary_df = pd.DataFrame(summary_rows)

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    per_img_path = Path(str(out_prefix) + "_per_image.csv")
    summary_path = Path(str(out_prefix) + "_summary.csv")  # 例: compare_iou04_summary.csv のイメージ

    per_img_df.to_csv(per_img_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"[INFO] IoU threshold = {args.iou_thr}")
    print(f"[INFO] per-image stats saved to {per_img_path}")
    print(f"[INFO] class summary saved to {summary_path}")
    print(summary_df)


if __name__ == "__main__":
    main()