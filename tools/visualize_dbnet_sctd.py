#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import sys

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ★ プロジェクトルート（detect_mdoels）を sys.path に追加
THIS_DIR = Path(__file__).resolve().parent          # tools/
ROOT_DIR = THIS_DIR.parent                          # detect_mdoels/
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

# ★ ここを修正： dnet.train_dbnet_sctd からインポート
from dbnet.train_dbnet_sctd import (
    SonarYOLODataset,
    DBNet,
    decode_predictions,
)

CLASS_NAMES = ["aircraft", "ship", "human"]  # クラスID 0,1,2 に対応

def load_model(weights_path, num_classes=3, anchors=None, device="cuda"):
    # anchors が None のときだけデフォルト3アンカーにしておく（必要なら）
    if anchors is None:
        anchors = [(10, 13), (16, 30), (33, 23)]
    model = DBNet(num_classes=num_classes, anchors=anchors).to(device)
    ckpt = torch.load(weights_path, map_location=device)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, anchors

def yolo_norm_to_xyxy(labels, img_size):
    """
    labels: [N,5] (cls, cx, cy, w, h), すべて 0〜1（640 で正規化された値）
    戻り値: boxes_xyxy [N,4], cls_ids [N]
    """
    if labels.numel() == 0:
        return torch.zeros((0, 4)), torch.zeros((0,), dtype=torch.long)

    cls = labels[:, 0].long()
    cx  = labels[:, 1] * img_size
    cy  = labels[:, 2] * img_size
    w   = labels[:, 3] * img_size
    h   = labels[:, 4] * img_size

    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    boxes = torch.stack([x1, y1, x2, y2], dim=1)
    return boxes, cls

def visualize_sample(
    img_tensor,
    gt_boxes, gt_cls,
    pred_boxes, pred_scores, pred_cls,
    save_path=None,
    title_extra="",
):
    """
    img_tensor: [3,H,W] (0〜1)
    *_boxes: [N,4] xyxy (pixels)
    *_cls: [N], *_scores: [N]
    """
    img = img_tensor.permute(1, 2, 0).cpu().numpy()

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(img)
    ax.axis("off")

    # GT: 緑
    for b, c in zip(gt_boxes, gt_cls):
        x1, y1, x2, y2 = b.tolist()
        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor="g",
            facecolor="none",
        )
        ax.add_patch(rect)
        cls_name = CLASS_NAMES[int(c)]
        ax.text(
            x1,
            y1 - 2,
            f"GT:{cls_name}",
            fontsize=8,
            color="g",
            bbox=dict(facecolor="black", alpha=0.5, pad=1),
        )

    # Pred: 赤
    for b, s, c in zip(pred_boxes, pred_scores, pred_cls):
        x1, y1, x2, y2 = b.tolist()
        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=1.5,
            edgecolor="r",
            facecolor="none",
            linestyle="--",
        )
        ax.add_patch(rect)
        cls_name = CLASS_NAMES[int(c)]
        ax.text(
            x1,
            y2 + 10,
            f"P:{cls_name} {s:.2f}",
            fontsize=8,
            color="r",
            bbox=dict(facecolor="black", alpha=0.5, pad=1),
        )

    ax.set_title(title_extra)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchors", type=str, default="")
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--conf-thres", type=float, default=0.05)
    parser.add_argument("--max-det", type=int, default=60)
    parser.add_argument("--topk-per-level", type=int, default=1000)
    parser.add_argument("--num-images", type=int, default=16)
    parser.add_argument("--save-dir", type=str, default="vis_dbnet")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ★ ここで anchors 文字列をパース（main のローカル変数 anchors を作る）
    if args.anchors:
        # 例: "(77,73),(105,212),(317,171),(172,429),(467,382)"
        import ast
        anchors = ast.literal_eval(f"[{args.anchors}]")
    else:
        anchors = None

    ds = SonarYOLODataset(
        args.dataset_root,
        args.run_name,
        img_size=args.img_size,
        augment=False,
    )

    # ★ anchors 変数をここで渡す
    model, anchors = load_model(
        args.weights,
        num_classes=args.num_classes,
        anchors=anchors,
        device=device,
    )

    # model, anchors = load_model(
    #     args.weights, num_classes=args.num_classes, device=device
    # )

    # 適当なインデックスをサンプリング
    import random
    indices = list(range(len(ds)))
    random.seed(0)
    random.shuffle(indices)
    if args.num_images > 0:
        indices = indices[:args.num_images]

    for idx in indices:
        img, labels, path = ds[idx]

        img_b = img.unsqueeze(0).to(device)
        with torch.no_grad():
            p = model(img_b)
            preds_list = decode_predictions(
                p,
                num_classes=args.num_classes,
                stride=8,
                conf_thres=args.conf_thres,
                max_det=args.max_det,
                topk_per_level=args.topk_per_level,
                anchors_px=anchors,
            )
            preds = preds_list[0] if isinstance(preds_list, (list, tuple)) else preds_list

        if preds is None:
            pred_boxes = torch.zeros((0, 4))
            pred_scores = torch.zeros((0,))
            pred_cls = torch.zeros((0,), dtype=torch.long)
        else:
            pred_boxes, pred_scores, pred_cls = preds
            pred_boxes = pred_boxes.cpu()
            pred_scores = pred_scores.cpu()
            pred_cls = pred_cls.cpu().long()

        gt_boxes, gt_cls = yolo_norm_to_xyxy(labels, img_size=args.img_size)

        title = f"{Path(path).name} | GT={len(gt_boxes)} Pred={len(pred_boxes)} (conf>{args.conf_thres})"
        save_path = os.path.join(args.save_dir, f"{idx:04d}.png")

        visualize_sample(
            img, gt_boxes, gt_cls,
            pred_boxes, pred_scores, pred_cls,
            save_path=save_path,
            title_extra=title,
        )
        print(f"saved: {save_path}")

if __name__ == "__main__":
    main()