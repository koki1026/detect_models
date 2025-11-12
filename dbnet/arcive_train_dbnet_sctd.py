# train_dbnet_sctd.py
import os, json, math, random, time, argparse
from pathlib import Path
from typing import List, Tuple
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.ops import box_iou, nms

import yaml
import csv
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 画面表示なしで保存
import matplotlib.pyplot as plt
# ----------------------------
# Utils
# ----------------------------
def set_seed(seed=0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def xywhn_to_xyxy(boxes, img_w, img_h): # normalized (xc,yc,w,h) を絶対座標 (xmin,ymin,xmax,ymax) に変換する
    # boxes: [N,4] normalized (xc,yc,w,h) -> absolute xyxy
    x = boxes[:,0]*img_w; y = boxes[:,1]*img_h
    w = boxes[:,2]*img_w; h = boxes[:,3]*img_h
    xy1 = torch.stack([x - w/2, y - h/2], 1)
    xy2 = torch.stack([x + w/2, y + h/2], 1)
    return torch.cat([xy1, xy2], 1)

def ciou_loss(pred_xywh, tgt_xywh, eps=1e-7): # IoUベースのCIoU損失
    # pred,tgt: [N,4] absolute xywh (x,y,w,h)
    px, py, pw, ph = pred_xywh.unbind(1)
    gx, gy, gw, gh = tgt_xywh.unbind(1)
    # IoU
    pred_xyxy = torch.stack([px - pw/2, py - ph/2, px + pw/2, py + ph/2], 1)
    tgt_xyxy  = torch.stack([gx - gw/2, gy - gh/2, gx + gw/2, gy + gh/2], 1)
    i = (torch.min(pred_xyxy[:,2], tgt_xyxy[:,2]) - torch.max(pred_xyxy[:,0], tgt_xyxy[:,0])).clamp(0)
    j = (torch.min(pred_xyxy[:,3], tgt_xyxy[:,3]) - torch.max(pred_xyxy[:,1], tgt_xyxy[:,1])).clamp(0)
    inter = i * j
    area_p = (pred_xyxy[:,2]-pred_xyxy[:,0]).clamp(0) * (pred_xyxy[:,3]-pred_xyxy[:,1]).clamp(0)
    area_g = (tgt_xyxy[:,2]-tgt_xyxy[:,0]).clamp(0) * (tgt_xyxy[:,3]-tgt_xyxy[:,1]).clamp(0)
    union = area_p + area_g - inter + eps
    iou = inter / union

    # enclosing box
    cx1 = torch.min(pred_xyxy[:,0], tgt_xyxy[:,0])
    cy1 = torch.min(pred_xyxy[:,1], tgt_xyxy[:,1])
    cx2 = torch.max(pred_xyxy[:,2], tgt_xyxy[:,2])
    cy2 = torch.max(pred_xyxy[:,3], tgt_xyxy[:,3])
    cw = (cx2 - cx1).clamp(0); ch = (cy2 - cy1).clamp(0)
    c2 = cw*cw + ch*ch + eps

    # center distance
    rho2 = (px - gx)**2 + (py - gy)**2

    # aspect ratio term
    v = (4/(math.pi**2)) * torch.pow(torch.atan(gw/(gh+eps)) - torch.atan(pw/(ph+eps)), 2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)
    ciou = iou - (rho2 / c2) - alpha * v
    return 1 - ciou  # loss

# クラス別APと精度/再現率を出す
def compute_map50_per_class(preds, gts, img_size=640, num_classes=3):
    from collections import defaultdict
    per_cls = defaultdict(list)
    for (p_boxes,p_scores,p_labels), gt in zip(preds, gts):
        device = p_boxes.device
        gt_boxes = xywhn_to_xyxy(gt[:,1:], img_size, img_size) if gt.numel() else torch.zeros((0,4),device=device)
        gt_cls   = gt[:,0].long() if gt.numel() else torch.zeros((0,),dtype=torch.long,device=device)
        for c in range(num_classes):
            pb = p_boxes[p_labels.long()==c]
            ps = p_scores[p_labels.long()==c]
            gb = gt_boxes[gt_cls==c]
            matched = torch.zeros((gb.size(0),), dtype=torch.bool, device=device)
            order = torch.argsort(ps, descending=True)
            tp=[]; fp=[]
            for idx in order:
                if pb.size(0)==0: break
                if gb.size(0)==0:
                    tp.append(0); fp.append(1); continue
                ious = box_iou(pb[idx:idx+1], gb).squeeze(0)
                iou, j = (ious.max(0))
                if iou >= 0.5 and (not matched[j]):
                    tp.append(1); fp.append(0); matched[j]=True
                else:
                    fp.append(1); tp.append(0)
            if len(tp)==0 and gb.size(0)==0:
                per_cls[c].append(1.0)
            elif len(tp)==0:
                per_cls[c].append(0.0)
            else:
                tp = torch.tensor(tp, dtype=torch.float32, device=device)
                fp = torch.tensor(fp, dtype=torch.float32, device=device)
                cum_tp = torch.cumsum(tp,0); cum_fp = torch.cumsum(fp,0)
                recalls = cum_tp / max(1, gb.size(0))
                precis  = cum_tp / torch.clamp(cum_tp+cum_fp, min=1)
                ps=[]
                for r in torch.linspace(0,1,11,device=device):
                    mask = recalls>=r
                    p = precis[mask].max() if mask.any() else torch.tensor(0.,device=device)
                    ps.append(p)
                per_cls[c].append(torch.stack(ps).mean().item())
    return {c: float(np.mean(v)) if v else 0.0 for c,v in per_cls.items()}


def _save_plots_from_csv(csv_path: Path, out_dir: Path):
    df = pd.read_csv(csv_path)
    if "epoch" not in df.columns:
        return
    df = df.sort_values("epoch")

    # 1) mAP@0.5
    plt.figure()
    plt.plot(df["epoch"], df["map50"], marker="o")
    plt.xlabel("Epoch"); plt.ylabel("mAP@0.5"); plt.title("mAP@0.5 over Epochs"); plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir/"plot_map50.png", dpi=180)
    plt.close()

    # 2) Per-class AP
    if all(c in df.columns for c in ["ap_aircraft","ap_ship","ap_human"]):
        plt.figure()
        plt.plot(df["epoch"], df["ap_aircraft"], marker="o", label="aircraft")
        plt.plot(df["epoch"], df["ap_ship"], marker="o", label="ship")
        plt.plot(df["epoch"], df["ap_human"], marker="o", label="human")
        plt.xlabel("Epoch"); plt.ylabel("AP@0.5"); plt.title("Per-class AP@0.5 over Epochs")
        plt.legend(); plt.grid(True)
        plt.tight_layout()
        plt.savefig(out_dir/"plot_ap_per_class.png", dpi=180)
        plt.close()

    # 3) Losses
    if all(c in df.columns for c in ["l_box","l_obj","l_cls"]):
        plt.figure()
        plt.plot(df["epoch"], df["l_box"], marker="o", label="l_box")
        plt.plot(df["epoch"], df["l_obj"], marker="o", label="l_obj")
        plt.plot(df["epoch"], df["l_cls"], marker="o", label="l_cls")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Losses over Epochs")
        plt.legend(); plt.grid(True)
        plt.tight_layout()
        plt.savefig(out_dir/"plot_losses.png", dpi=180)
        plt.close()

    # 4) Predictions kept
    if "val_preds_kept" in df.columns:
        plt.figure()
        plt.plot(df["epoch"], df["val_preds_kept"], marker="o")
        plt.xlabel("Epoch"); plt.ylabel("val_preds_kept"); plt.title("Predictions Kept per Epoch")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out_dir/"plot_preds_kept.png", dpi=180)
        plt.close()

def _load_yaml_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    # 想定するセクションを平坦化（存在すれば）
    flat = {}
    for k, v in cfg.items():
        if isinstance(v, dict) and k in {"data","train","eval","model","logging"}:
            for kk, vv in v.items():
                flat[kk] = vv
        else:
            flat[k] = v
    return flat

# ----------------------------
# Dataset (dataset_root/runs/<RUN>/)
# ----------------------------
class SonarYOLODataset(Dataset): # YOLO形式ラベルのSonarデータセット
    def __init__(self, dataset_root, run_name, img_size=640, augment=False):
        self.root = Path(dataset_root)/"runs"/run_name
        self.img_dir = self.root/"sonar"
        self.lbl_dir = self.root/"labels"
        self.img_size = img_size
        self.augment = augment
        self.samples = sorted([p for p in self.img_dir.glob("*") if p.suffix.lower() in [".jpg",".png",".jpeg"]])

    def __len__(self): return len(self.samples)

    def _load_labels(self, stem):
        p = self.lbl_dir/f"{stem}.txt"
        if not p.exists(): return torch.zeros((0,5), dtype=torch.float32)
        lines = []
        for line in p.read_text().strip().splitlines():
            ss = line.strip().split()
            if len(ss) != 5: continue
            c, xc, yc, w, h = map(float, ss)
            lines.append([c, xc, yc, w, h])
        if not lines:
            return torch.zeros((0,5), dtype=torch.float32)
        return torch.tensor(lines, dtype=torch.float32)

    def __getitem__(self, i):
        img_path = self.samples[i]
        img = Image.open(img_path).convert("RGB").resize((self.img_size,self.img_size))
        img = torch.from_numpy(np.array(img)).permute(2,0,1).float()/255.0
        labels = self._load_labels(img_path.stem)  # [N,5] (cls, xc, yc, w, h) normalized
        return img, labels, str(img_path)

def build_indices_by_class(ds):
    idx_pos_by_cls = {0:[],1:[],2:[]}
    idx_empty = []
    for i in range(len(ds)):
        _, labels, _ = ds[i]
        if labels.numel() == 0:
            idx_empty.append(i)
            continue
        for c in labels[:,0].long().tolist():
            if int(c) in idx_pos_by_cls:
                idx_pos_by_cls[int(c)].append(i)
    # 重複あるので一意化
    for k in idx_pos_by_cls: idx_pos_by_cls[k] = sorted(set(idx_pos_by_cls[k]))
    idx_empty = sorted(set(idx_empty))
    return idx_pos_by_cls, idx_empty

class CurriculumSubset(torch.utils.data.Dataset):
    """空画像取り込み確率 p_empty を握って forward するミニラッパー"""
    def __init__(self, ds, idx_pos_by_cls, idx_empty, p_empty=0.2):
        self.ds = ds
        self.idx_pos_by_cls = idx_pos_by_cls
        self.idx_empty = idx_empty
        self.p_empty = p_empty
        # positives のプール
        self.pool_pos = sorted(set().union(*idx_pos_by_cls.values()))
        self.len = len(self.pool_pos) + int(self.p_empty*len(self.idx_empty))

    def __len__(self): return self.len

    def __getitem__(self, _):
        # 空を引くか決める
        if (len(self.idx_empty)>0) and (random.random() < self.p_empty):
            i = random.choice(self.idx_empty)
            return self.ds[i]
        # それ以外は positive から
        i = random.choice(self.pool_pos)
        return self.ds[i]

def make_weighted_sampler(idx_pos_by_cls, idx_empty, weight_empty=1.0):
    # aircraft/human を ship より重めに
    n0, n1, n2 = len(idx_pos_by_cls[0]), len(idx_pos_by_cls[1]), len(idx_pos_by_cls[2])
    # 例：逆頻度の平方根（過補正しない）
    def w(n): return 1.0 / max(1, math.sqrt(n))
    w0, w1, w2 = w(n0), w(n1), w(n2)
    weights = {}
    for i in idx_pos_by_cls[0]: weights[i] = w0
    for i in idx_pos_by_cls[1]: weights[i] = max(weights.get(i,0), w1)
    for i in idx_pos_by_cls[2]: weights[i] = max(weights.get(i,0), w2)
    for i in idx_empty: weights[i] = weight_empty
    # テンソル化（順番は ds 全体順に）
    return weights

def collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch])
    lbls = [b[1] for b in batch]
    paths= [b[2] for b in batch]
    return imgs, lbls, paths

# ----------------------------
# Model: Dual Backbone -> Simple Neck -> YOLO-like Head
# （PP-LCNet/GhostNet差し替え前の軽量版）
# ----------------------------
class ConvBNAct(nn.Sequential): # Convolution + BatchNorm + SiLU
    def __init__(self, c1, c2, k=3, s=1, p=None):
        if p is None: p = k//2
        super().__init__(nn.Conv2d(c1,c2,k,s,p,bias=False), nn.BatchNorm2d(c2), nn.SiLU(True))

class TinyBackbone(nn.Module): # 超軽量バックボーン（3層Conv）
    def __init__(self, c_in=3, c_out=128):
        super().__init__()
        self.stem = ConvBNAct(c_in, 32, 3, 2)  # /2
        self.b1 = ConvBNAct(32, 64, 3, 2)      # /4
        self.b2 = ConvBNAct(64, c_out, 3, 2)   # /8
        self.out_ch = c_out
    def forward(self, x):
        return self.b2(self.b1(self.stem(x)))

class DualBackbone(nn.Module): # 2つのTinyBackboneを並列に動かして特徴量を結合
    def __init__(self):
        super().__init__()
        self.b1 = TinyBackbone(3,128)
        self.b2 = TinyBackbone(3,128)
        self.merge = ConvBNAct(256, 192, 1, 1)
    def forward(self, x):
        f1 = self.b1(x); f2 = self.b2(x)
        return self.merge(torch.cat([f1,f2],1))  # /8, C=192

class SimpleNeck(nn.Module): # 単純なNeck（2層Conv）
    def __init__(self, c=192):
        super().__init__()
        self.m = nn.Sequential(ConvBNAct(c,192,3,1), ConvBNAct(192,192,3,1))
    def forward(self, x): return self.m(x)

class DetectHead(nn.Module): # 検出ヘッド
    # single stride head (/8), 3 anchors
    def __init__(self, c=192, num_classes=3, anchors=[(10,13),(16,30),(33,23)]):
        super().__init__()
        self.na = len(anchors)
        self.nc = num_classes
        self.anchors = torch.tensor(anchors, dtype=torch.float32)  # in pixels at stride 8 scale
        self.cv = nn.Conv2d(c, self.na*(5+self.nc), 1, 1)

        with torch.no_grad():
            b = self.cv.bias.view(self.na, 5 + self.nc)
            b[:, 4] = -4.5   # objectness ロジット初期値 ~ 1%
            b[:, 5:] = -2.0  # 各クラス確率も控えめ
            self.cv.bias = nn.Parameter(b.view(-1))
    def forward(self, x):
        b,c,h,w = x.shape
        p = self.cv(x)
        p = p.view(b, self.na, 5+self.nc, h, w).permute(0,1,3,4,2).contiguous()
        return p  # raw

class DBNet(nn.Module): # DBNetモデル本体
    def __init__(self, num_classes=3):
        super().__init__()
        self.backbone = DualBackbone()
        self.neck = SimpleNeck(192)
        self.head = DetectHead(192, num_classes=num_classes)
        self.stride = 8
    def forward(self, x):
        f = self.backbone(x)
        f = self.neck(f)
        p = self.head(f)
        return p

# ----------------------------
# Target assignment (very simple YOLO-style, single scale)
# ----------------------------
def build_grids(h, w, device, stride): # グリッド座標を作成
    gy, gx = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
    grid = torch.stack((gx, gy), 2).float()  # [h,w,2]
    return grid, stride

def assign_targets(labels_list, pshape, num_classes, anchors, stride, img_size): # ターゲット割り当て
    """
    labels_list: list of [N,5] (cls,xc,yc,w,h) normalized
    pshape: (B, na, H, W, 5+nc)
    returns:
      tobj [B,na,H,W], tcls [B,na,H,W,nc], tbox [B,na,H,W,4], indices mask
    """
    device = anchors.device
    B, na, H, W, _ = pshape
    tobj = torch.zeros((B,na,H,W), device=device)
    tcls = torch.zeros((B,na,H,W,num_classes), device=device)
    tbox = torch.zeros((B,na,H,W,4), device=device)

    grid, _ = build_grids(H, W, device, stride)
    a_wh = anchors.view(na,1,1,2)  # anchors は grid 単位

    for b, labels in enumerate(labels_list):
        if labels.numel() == 0: 
            continue
        lt = labels.clone()
        lt[:,1:] = lt[:,1:] * img_size
        gcls = lt[:,0].long()
        gx, gy, gw, gh = lt[:,1:].t()

        gi = (gx / stride).clamp(0, W-1e-3)
        gj = (gy / stride).clamp(0, H-1e-3)
        gi_idx = gi.long(); gj_idx = gj.long()

        gt_anchor = torch.stack([gw/stride, gh/stride], 1)  # grid 単位
        ious = []
        for a in range(na):
            aw, ah = a_wh[a,0,0]
            inter = torch.minimum(gt_anchor[:,0], aw) * torch.minimum(gt_anchor[:,1], ah)
            union = gt_anchor[:,0]*gt_anchor[:,1] + aw*ah - inter + 1e-9
            ious.append(inter/union)
        ious = torch.stack(ious, 1)
        best_a = ious.argmax(1)

        for i in range(lt.size(0)):
            a = int(best_a[i].item())
            # 近傍 3x3 を正例に
            for di in (-1,0,1):
                for dj in (-1,0,1):
                    ii = int((gi_idx[i] + di).clamp(0, W-1))
                    jj = int((gj_idx[i] + dj).clamp(0, H-1))
                    tobj[b,a,jj,ii] = 1.0
                    tcls[b,a,jj,ii, gcls[i]] = 1.0
                    tbox[b,a,jj,ii] = torch.tensor([gx[i], gy[i], gw[i], gh[i]], device=device)
    return tobj, tcls, tbox
    

# ----------------------------
# Loss
# ----------------------------
class DBLoss(nn.Module):  # DBNet用損失関数
    def __init__(self, num_classes, box=7.5, obj=1.0, cls=0.5, stride=8, img_size=640, anchors=[(10,13),(16,30),(33,23)]):
        super().__init__()
        self.lw_box, self.lw_obj, self.lw_cls = box, obj, cls
        self.num_classes = num_classes
        self.stride = stride
        self.img_size = img_size
        self.register_buffer("anchors", torch.tensor(anchors, dtype=torch.float32))

    def forward(self, p_raw, labels_list):
        device = p_raw.device
        B, na, H, W, D = p_raw.shape

        # anchors: image px -> grid 単位
        anchors_px   = self.anchors.to(device)
        anchors_grid = anchors_px / self.stride

        # --- center/size decode（従来通り） ---
        px = torch.sigmoid(p_raw[..., 0])
        py = torch.sigmoid(p_raw[..., 1])
        pw = torch.exp(p_raw[..., 2]) * anchors_grid[:, 0].view(na,1,1)
        ph = torch.exp(p_raw[..., 3]) * anchors_grid[:, 1].view(na,1,1)

        # --- 重要：obj/cls はロジットのまま損失へ ---
        logits_obj = p_raw[..., 4]      # no sigmoid
        logits_cls = p_raw[..., 5:]     # no sigmoid

        # 予測 box を画像ピクセルへ
        grid, _ = build_grids(H, W, device, self.stride)
        gx = (grid[..., 0].unsqueeze(0).unsqueeze(0) + px) * self.stride
        gy = (grid[..., 1].unsqueeze(0).unsqueeze(0) + py) * self.stride
        pxywh = torch.stack([gx, gy, pw * self.stride, ph * self.stride], -1)

        # ターゲット（anchors は grid 単位で渡す）
        tobj, tcls, tbox = assign_targets(
            labels_list, p_raw.shape, self.num_classes, anchors_grid, self.stride, self.img_size
        )

        # --- Box loss (posのみ) ---
        pos = tobj > 0.5
        n_pos = pos.sum().clamp(min=1)
        if pos.any():
            l_box = ciou_loss(pxywh[pos], tbox[pos]).mean()
        else:
            l_box = torch.tensor(0., device=device)

        # --- Obj/Cls loss = with_logits 版で安定化 ---
        l_obj = F.binary_cross_entropy_with_logits(logits_obj, tobj)
        if pos.any():
            l_cls = F.binary_cross_entropy_with_logits(logits_cls[pos], tcls[pos])
        else:
            l_cls = torch.tensor(0., device=device)

        total = self.lw_box * l_box + self.lw_obj * l_obj + self.lw_cls * l_cls
        return total, {
            "l_box": float(l_box.detach()),
            "l_obj": float(l_obj.detach()),
            "l_cls": float(l_cls.detach()),
            "n_pos": int(n_pos.item()) if hasattr(n_pos, "item") else int(n_pos),
        }

# ----------------------------
# Inference + NMS + mAP(0.5)
# ----------------------------
def decode_predictions(p_raw, num_classes, stride=8,
                       conf_thres=0.1, max_det=300, topk_per_level=1000):
    device = p_raw.device
    B, na, H, W, D = p_raw.shape
    anchors_px = torch.tensor([(10,13),(16,30),(33,23)], dtype=torch.float32, device=device)
    anchors_grid = anchors_px / stride

    px = torch.sigmoid(p_raw[...,0])
    py = torch.sigmoid(p_raw[...,1])
    pw = torch.exp(p_raw[...,2]) * anchors_grid[:,0].view(na,1,1)
    ph = torch.exp(p_raw[...,3]) * anchors_grid[:,1].view(na,1,1)
    pobj = torch.sigmoid(p_raw[...,4])
    pcls = torch.sigmoid(p_raw[...,5:])

    grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    gx = (grid_x.unsqueeze(0).unsqueeze(0) + px) * stride
    gy = (grid_y.unsqueeze(0).unsqueeze(0) + py) * stride
    gw = pw * stride
    gh = ph * stride

    outs = []
    for b in range(B):
        boxes=[]; scores=[]; labels=[]
        for a in range(na):
            conf = pobj[b,a] * pcls[b,a].max(-1).values  # obj * best-class
            flat = conf.view(-1)
            # 上位だけ採用（レベルごとtop-K）
            if flat.numel() > topk_per_level:
                vals, idxs = torch.topk(flat, topk_per_level)
                mask = torch.zeros_like(flat, dtype=torch.bool)
                mask[idxs] = True
                mask = mask.view_as(conf)
            else:
                mask = conf > conf_thres
            if mask.sum() == 0:
                continue
            x = gx[b,a][mask]; y = gy[b,a][mask]; w = gw[b,a][mask]; h = gh[b,a][mask]
            x1 = x - w/2; y1 = y - h/2; x2 = x + w/2; y2 = y + h/2
            boxes.append(torch.stack([x1,y1,x2,y2],1))
            scores.append(conf[mask])
            labels.append(pcls[b,a][mask].argmax(-1).float())
        if boxes:
            boxes = torch.cat(boxes,0); scores = torch.cat(scores,0); labels = torch.cat(labels,0)
            keep = nms(boxes, scores, 0.5)
            keep = keep[:max_det] if keep.numel() > max_det else keep
            outs.append((boxes[keep], scores[keep], labels[keep]))
        else:
            outs.append((torch.zeros((0,4),device=device), torch.zeros((0,),device=device), torch.zeros((0,),device=device)))
    return outs


def compute_map50(preds, gts, img_size=640): # mAP@0.5計算
    """
    preds: list of (boxes[x1y1x2y2], scores, labels)
    gts:   list of labels tensor [N,5] (cls,xc,yc,w,h) normalized
    """
    aps=[]
    for (p_boxes,p_scores,p_labels), gt in zip(preds, gts):
        gt_boxes = xywhn_to_xyxy(gt[:,1:], img_size, img_size) if gt.numel() else torch.zeros((0,4),device=p_boxes.device)
        gt_cls = gt[:,0].long() if gt.numel() else torch.zeros((0,),dtype=torch.long,device=p_boxes.device)

        # simple AP@0.5 per-image (micro):
        matched = torch.zeros((gt_boxes.size(0),), dtype=torch.bool, device=p_boxes.device)
        order = torch.argsort(p_scores, descending=True)
        tp=[]; fp=[]
        for idx in order:
            if p_boxes.size(0)==0: break
            pb = p_boxes[idx:idx+1]
            pl = int(p_labels[idx].item())
            if gt_boxes.size(0)==0:
                tp.append(0); fp.append(1); continue
            ious = box_iou(pb, gt_boxes).squeeze(0)  # [G]
            iou, j = (ious.max(0))
            if iou >= 0.5 and (not matched[j]) and (pl==int(gt_cls[j].item())):
                tp.append(1); fp.append(0); matched[j]=True
            else:
                fp.append(1); tp.append(0)
        if not tp and gt_boxes.size(0)==0:
            aps.append(1.0)  # empty->empty: define as perfect
            continue
        if not tp and tp!=[]:
            aps.append(0.0); continue
        tp = torch.tensor(tp, dtype=torch.float32, device=p_boxes.device)
        fp = torch.tensor(fp, dtype=torch.float32, device=p_boxes.device)
        cum_tp = torch.cumsum(tp,0); cum_fp = torch.cumsum(fp,0)
        recalls = cum_tp / max(1, gt_boxes.size(0))
        precis = cum_tp / torch.clamp(cum_tp+cum_fp, min=1)
        # 11-point interpolation (simple)
        ps=[]
        for r in torch.linspace(0,1,11,device=p_boxes.device):
            mask = recalls>=r
            p = precis[mask].max() if mask.any() else torch.tensor(0.,device=p_boxes.device)
            ps.append(p)
        aps.append(torch.stack(ps).mean().item())
    return float(np.mean(aps)) if aps else 0.0

# ----------------------------
# Train / Eval
# ----------------------------
def train(args): # メインの学習ループ
    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # data
    ds = SonarYOLODataset(args.dataset_root, args.run_name, args.img_size, augment=False)
    # 分割
    idx_pos_by_cls, idx_empty = build_indices_by_class(ds)
    print({k:len(v) for k,v in idx_pos_by_cls.items()}, "empty:", len(idx_empty))

    # まず valid は「GTあり画像」中心に切る（ratioは適宜）
    pos_all = sorted(set().union(*idx_pos_by_cls.values()))
    random.Random(args.seed).shuffle(pos_all)
    nva_pos = max(1, int(0.15*len(pos_all)))
    va_pos = set(pos_all[:nva_pos]); tr_pos = set(pos_all[nva_pos:])

    # 空は val にも少しだけ入れる
    nva_empty = max(0, int(0.05*len(idx_empty)))
    va_empty = set(random.Random(args.seed+1).sample(idx_empty, nva_empty)) if nva_empty>0 else set()
    tr_empty = set([i for i in idx_empty if i not in va_empty])

    tr_idx = sorted(tr_pos.union(tr_empty))
    va_idx = sorted(va_pos.union(va_empty))

    ds_tr = torch.utils.data.Subset(ds, tr_idx)
    ds_va = torch.utils.data.Subset(ds, va_idx)

    # カリキュラム：前半は空画像20%取り込み
    ds_tr_cur = CurriculumSubset(ds, idx_pos_by_cls, idx_empty=list(tr_empty), p_empty=0.2)

    # サンプラ（少数クラスを重めに出す）
    w_map = make_weighted_sampler(idx_pos_by_cls, list(tr_empty), weight_empty=0.5)
    sample_weights = torch.tensor([w_map.get(i,1.0) for i in tr_idx], dtype=torch.float)
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(tr_idx), replacement=True)

    def _worker_init_fn(worker_id):
        import os, torch
        try:
            import cv2
            cv2.setNumThreads(0)
        except Exception:
            pass
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        torch.set_num_threads(1)

    dl_tr = DataLoader(
        ds_tr_cur, batch_size=args.batch_size, num_workers=args.workers,
        shuffle=False, sampler=None, collate_fn=collate_fn, pin_memory=True,
        persistent_workers=True, prefetch_factor=4, worker_init_fn=_worker_init_fn,
    )
    dl_va = DataLoader(
        ds_va, batch_size=args.batch_size, num_workers=args.workers,
        shuffle=False, collate_fn=collate_fn, pin_memory=True,
        persistent_workers=True, prefetch_factor=4, worker_init_fn=_worker_init_fn,
    )
    
    # model
    model = DBNet(num_classes=args.num_classes).to(device)
    criterion = DBLoss(num_classes=args.num_classes, img_size=args.img_size)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)

    # logging dir
    exp_dir = Path("runs")/time.strftime("exp_%Y%m%d_%H%M%S")
    exp_dir.mkdir(parents=True, exist_ok=True)

    if args.save_config:
        try:
            resolved = vars(args).copy()
            (exp_dir / "config_resolved.yaml").write_text(
                yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True)
            )
        except Exception as e:
            print(f"[warn] failed to save resolved config: {e}")

    csv_path = exp_dir/"metrics.csv"
    logs_root = Path("runs_logs")
    logs_root.mkdir(parents=True, exist_ok=True)
    csv_path_mirror = logs_root/f"{exp_dir.name}.csv"

    csv_header = [
        "epoch","map50","l_box","l_obj","l_cls","n_pos",
        "val_preds_kept","val_gt_images","val_gt_boxes",
        "ap_aircraft","ap_ship","ap_human"
    ]
    for pth in [csv_path, csv_path_mirror]:
        with open(pth, "w", newline="") as f:
            csv.writer(f).writerow(csv_header)

    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    amp_enabled = bool(args.amp and device_type == 'cuda')


    for ep in range(args.epochs):
        model.train()
        m = []
        for imgs, labels_list, _ in dl_tr:
            imgs = imgs.to(device)
            labels_list = [l.to(device) for l in labels_list]
            optim.zero_grad(set_to_none=True)
            device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
            amp_enabled = bool(args.amp and device_type == 'cuda')
            with torch.amp.autocast(device_type=device_type, enabled=amp_enabled):
                p = model(imgs)  # [B,na,H,W,5+C]
                loss, logs = criterion(p, labels_list)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            m.append(logs)
        # epoch log
        if m:
            avg = {k: float(np.mean([x[k] for x in m])) for k in m[0].keys()}
        else:
            avg = {"l_box":0,"l_obj":0,"l_cls":0,"n_pos":0}
        print(f"[ep {ep+1}/{args.epochs}] loss: box={avg['l_box']:.3f} obj={avg['l_obj']:.3f} cls={avg['l_cls']:.3f} npos={avg['n_pos']:.1f}")

        # val (mAP@0.5)
        model.eval()
        all_preds=[]; all_gts=[]
        with torch.no_grad():
            for imgs, labels_list, _ in dl_va:
                imgs = imgs.to(device)
                p = model(imgs)
                preds = decode_predictions(
                    p, args.num_classes, 
                    stride=8,
                    conf_thres=args.conf_thres, 
                    max_det=args.max_det, 
                    topk_per_level=args.topk_per_level
                )
                all_preds += preds
                all_gts   += [l.to(device) for l in labels_list]

        # 集計
        map50 = compute_map50(all_preds, all_gts, img_size=args.img_size)
        val_gt_imgs  = sum(1 for l in all_gts if l.numel() > 0)
        val_gt_boxes = sum(int(l.size(0)) for l in all_gts if l.numel() > 0)
        val_preds_kept = sum(int(x[0].size(0)) for x in all_preds)
        pc = compute_map50_per_class(all_preds, all_gts, img_size=args.img_size, num_classes=args.num_classes)
        ap_air = pc.get(0, 0.0); ap_ship = pc.get(1, 0.0); ap_human = pc.get(2, 0.0)

        print(f"  val mAP@0.5: {map50:.4f}")
        print(f"  val gt images: {val_gt_imgs}  boxes: {val_gt_boxes}")
        print(f"  val preds kept: {val_preds_kept} (<= {len(dl_va.dataset)*100})")
        print("  per-class AP@0.5:", {0:"aircraft",1:"ship",2:"human"}, pc)

        # save ckpt & json
        ckpt = {"model": model.state_dict(), "epoch": ep, "args": vars(args)}
        torch.save(ckpt, exp_dir/f"ckpt_ep{ep+1}.pt")
        (exp_dir/"metrics.json").write_text(json.dumps({"map50": map50, "epoch": ep+1, **avg}, indent=2))

        # --- CSV 1行追記 ---
        row = [
            ep+1, float(map50), avg["l_box"], avg["l_obj"], avg["l_cls"], avg["n_pos"],
            int(val_preds_kept), int(val_gt_imgs), int(val_gt_boxes),
            float(ap_air), float(ap_ship), float(ap_human)
        ]
        for pth in [csv_path, csv_path_mirror]:
            with open(pth, "a", newline="") as f:
                csv.writer(f).writerow(row)
    try:
        _save_plots_from_csv(csv_path, exp_dir)
        # ミラーにも同じ図が欲しければ以下を追加
        # _save_plots_from_csv(csv_path_mirror, logs_root)
        print(f"Saved plots to: {exp_dir}")
    except Exception as e:
        print(f"[plotting skipped] {e}")
    print(f"Done. Artifacts -> {exp_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--config", type=str, default=None, help="YAML config file path")
    tmp_args, _ = ap.parse_known_args()

    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset-root", type=str, required=False)
    ap.add_argument("--run-name", type=str, required=False)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--img-size", type=int, default=640)
    ap.add_argument("--num-classes", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--conf-thres", type=float, default=0.30, help="confidence threshold at eval")
    ap.add_argument("--max-det", type=int, default=80, help="max detections per image at eval")
    ap.add_argument("--topk-per-level", type=int, default=300, help="topK per anchor level at eval")
    ap.add_argument("--config", type=str, default=tmp_args.config, help="YAML config file path")
    ap.add_argument("--save-config", action="store_true", help="save used config to output dir")
    
    # ------ YAML があればデフォルトを上書き ------
    if tmp_args.config:
        cfg = _load_yaml_config(tmp_args.config)
        # argparse の defaults を YAML で上書き（CLI がさらに上書きする）
        ap.set_defaults(**{k.replace("-", "_"): v for k, v in cfg.items()})
    # ------ 最終パース（CLIが最優先） ------
    args = ap.parse_args()
    # 必須の data 引数が YAML 側にある前提で required=False としているため、最後にチェック
    for must in ["dataset_root","run_name"]:
        if getattr(args, must, None) in (None, ""):
            raise SystemExit(f"[error] --{must.replace('_','-')} is required (CLI or YAML)")
    train(args)
