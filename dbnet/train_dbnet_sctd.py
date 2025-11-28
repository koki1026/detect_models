#!/usr/bin/env python3
import os, json, math, random, time, argparse, csv, re, shutil
from pathlib import Path
from copy import deepcopy
from typing import List, Tuple
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.ops import box_iou, nms
from torchvision.ops import batched_nms

# Optional (YAML config)
try:
    import yaml
except Exception:
    yaml = None

# ----------------------------
# Utils
# ----------------------------
def set_seed(seed=0):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def xywhn_to_xyxy(boxes, img_w, img_h):  # normalized (xc,yc,w,h) -> absolute (x1,y1,x2,y2)
    x = boxes[:,0]*img_w; y = boxes[:,1]*img_h
    w = boxes[:,2]*img_w; h = boxes[:,3]*img_h
    xy1 = torch.stack([x - w/2, y - h/2], 1)
    xy2 = torch.stack([x + w/2, y + h/2], 1)
    return torch.cat([xy1, xy2], 1)

def ciou_loss(pred_xywh, tgt_xywh, eps=1e-7):
    px, py, pw, ph = pred_xywh.unbind(1)
    gx, gy, gw, gh = tgt_xywh.unbind(1)
    pred_xyxy = torch.stack([px - pw/2, py - ph/2, px + pw/2, py + ph/2], 1)
    tgt_xyxy  = torch.stack([gx - gw/2, gy - gh/2, gx + gw/2, gy + gh/2], 1)
    iw = (torch.min(pred_xyxy[:,2], tgt_xyxy[:,2]) - torch.max(pred_xyxy[:,0], tgt_xyxy[:,0])).clamp(0)
    ih = (torch.min(pred_xyxy[:,3], tgt_xyxy[:,3]) - torch.max(pred_xyxy[:,1], tgt_xyxy[:,1])).clamp(0)
    inter = iw * ih
    area_p = (pred_xyxy[:,2]-pred_xyxy[:,0]).clamp(0) * (pred_xyxy[:,3]-pred_xyxy[:,1]).clamp(0)
    area_g = (tgt_xyxy[:,2]-tgt_xyxy[:,0]).clamp(0) * (tgt_xyxy[:,3]-tgt_xyxy[:,1]).clamp(0)
    union = area_p + area_g - inter + eps
    iou = inter / union

    cx1 = torch.min(pred_xyxy[:,0], tgt_xyxy[:,0]); cy1 = torch.min(pred_xyxy[:,1], tgt_xyxy[:,1])
    cx2 = torch.max(pred_xyxy[:,2], tgt_xyxy[:,2]); cy2 = torch.max(pred_xyxy[:,3], tgt_xyxy[:,3])
    cw = (cx2 - cx1).clamp(0); ch = (cy2 - cy1).clamp(0)
    c2 = cw*cw + ch*ch + eps

    rho2 = (px - gx)**2 + (py - gy)**2
    v = (4/(math.pi**2)) * torch.pow(torch.atan(gw/(gh+eps)) - torch.atan(pw/(ph+eps)), 2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)
    ciou = iou - (rho2 / c2) - alpha * v
    return 1 - ciou

# ----------------------------
# Dataset (dataset_root/runs/<RUN>/)
# ----------------------------
class SonarYOLODataset(Dataset):
    def __init__(self, dataset_root, run_name, img_size=640, augment=False):
        self.root = Path(dataset_root) / "runs" / run_name
        self.img_dir = self.root / "sonar"
        self.lbl_dir = self.root / "labels"
        self.img_size = img_size
        self.augment = augment
        self.samples = sorted([
            p for p in self.img_dir.glob("*")
            if p.suffix.lower() in [".jpg", ".png", ".jpeg"]
        ])

    def __len__(self):
        return len(self.samples)

    def _load_labels(self, stem):
        p = self.lbl_dir / f"{stem}.txt"
        if not p.exists():
            return torch.zeros((0, 5), dtype=torch.float32)
        lines = []
        for line in p.read_text().strip().splitlines():
            ss = line.strip().split()
            if len(ss) != 5:
                continue
            c, xc, yc, w, h = map(float, ss)
            lines.append([c, xc, yc, w, h])
        if not lines:
            return torch.zeros((0, 5), dtype=torch.float32)
        return torch.tensor(lines, dtype=torch.float32)

    def __getitem__(self, i):
        img_path = self.samples[i]
        img = Image.open(img_path).convert("RGB")
        labels = self._load_labels(img_path.stem)

        # --- Step 2.5: keep-ratio resize (letterbox) ---
        def letterbox_keep_ratio(img, target=640, pad_value=114):
            w0, h0 = img.size
            scale = target / max(w0, h0)
            w1, h1 = int(round(w0 * scale)), int(round(h0 * scale))
            dw, dh = (target - w1) // 2, (target - h1) // 2
            img_resized = img.resize((w1, h1), Image.BILINEAR)
            canvas = Image.new("RGB", (target, target), (pad_value,) * 3)
            canvas.paste(img_resized, (dw, dh))
            return canvas, dict(scale=scale, dw=dw, dh=dh, W0=w0, H0=h0, target=target)

        img, p = letterbox_keep_ratio(img, target=self.img_size)

        # --- ラベル補正 ---
        if labels.size(0) > 0:
            cls = labels[:, 0:1]
            cx = labels[:, 1] * p["W0"]
            cy = labels[:, 2] * p["H0"]
            w = labels[:, 3] * p["W0"]
            h = labels[:, 4] * p["H0"]
            cx = (cx * p["scale"] + p["dw"]) / p["target"]
            cy = (cy * p["scale"] + p["dh"]) / p["target"]
            w = (w * p["scale"]) / p["target"]
            h = (h * p["scale"]) / p["target"]
            labels = torch.from_numpy(
                np.concatenate([cls, cx[:, None], cy[:, None], w[:, None], h[:, None]], axis=1)
            ).float()
        else:
            labels = torch.zeros((0, 5), dtype=torch.float32)

        # --- 画像テンソル化 ---
        img = torch.from_numpy(np.array(img, dtype=np.float32)).permute(2, 0, 1) / 255.0

        # 元の仕様と同じく3要素を返す
        return img, labels, str(img_path)

def build_indices_by_class(ds):
    idx_pos_by_cls = {0:[],1:[],2:[]}
    idx_empty = []
    for i in range(len(ds)):
        _, labels, _ = ds[i]
        if labels.numel() == 0:
            idx_empty.append(i); continue
        for c in labels[:,0].long().tolist():
            if int(c) in idx_pos_by_cls:
                idx_pos_by_cls[int(c)].append(i)
    for k in idx_pos_by_cls: idx_pos_by_cls[k] = sorted(set(idx_pos_by_cls[k]))
    idx_empty = sorted(set(idx_empty))
    return idx_pos_by_cls, idx_empty

class CurriculumSubset(torch.utils.data.Dataset):
    def __init__(self, ds, idx_pos_by_cls, idx_empty, p_empty=0.2):
        self.ds = ds
        self.idx_pos_by_cls = idx_pos_by_cls
        self.idx_empty = idx_empty
        self.p_empty = p_empty
        self.pool_pos = sorted(set().union(*idx_pos_by_cls.values()))
        self.len = len(self.pool_pos) + int(self.p_empty*len(self.idx_empty))

    def __len__(self): return self.len

    def __getitem__(self, _):
        if (len(self.idx_empty)>0) and (random.random() < self.p_empty):
            i = random.choice(self.idx_empty)
            return self.ds[i]
        i = random.choice(self.pool_pos)
        return self.ds[i]

def collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch])
    lbls = [b[1] for b in batch]
    paths= [b[2] for b in batch]
    return imgs, lbls, paths

# ----------------------------
# Model
# ----------------------------
class ConvBNAct(nn.Sequential):
    def __init__(self, c1, c2, k=3, s=1, p=None):
        if p is None: p = k//2
        super().__init__(nn.Conv2d(c1,c2,k,s,p,bias=False), nn.BatchNorm2d(c2), nn.SiLU(True))

class HardSwish(nn.Module):
    def __init__(self, inplace: bool = True):
        super().__init__()
        self.act = nn.Hardswish(inplace=inplace)

    def forward(self, x):
        return self.act(x)

class DepthSepConv(nn.Module):
    """
    PP-LCNet 風の depthwise separable conv ブロック
    """
    def __init__(self, c_in, c_out, k=3, s=1):
        super().__init__()
        p = k // 2
        self.dw = nn.Conv2d(c_in, c_in, k, s, p, groups=c_in, bias=False)
        self.dw_bn = nn.BatchNorm2d(c_in)
        self.pw = nn.Conv2d(c_in, c_out, 1, 1, 0, bias=False)
        self.pw_bn = nn.BatchNorm2d(c_out)
        self.act = HardSwish()

    def forward(self, x):
        x = self.dw(x)
        x = self.dw_bn(x)
        x = self.act(x)
        x = self.pw(x)
        x = self.pw_bn(x)
        x = self.act(x)
        return x

class GhostConv(nn.Module):
    """
    GhostNet 風の GhostConv（簡略版）
    """
    def __init__(self, c_in, c_out, k=1, s=1, ratio=2):
        super().__init__()
        c_prim = int(math.ceil(c_out / ratio))
        c_cheap = c_out - c_prim
        p = k // 2

        self.primary = nn.Sequential(
            nn.Conv2d(c_in, c_prim, k, s, p, bias=False),
            nn.BatchNorm2d(c_prim),
            HardSwish(),
        )
        self.cheap = nn.Sequential(
            nn.Conv2d(c_prim, c_cheap, 3, 1, 1, groups=c_prim, bias=False),
            nn.BatchNorm2d(c_cheap),
            HardSwish(),
        )

    def forward(self, x):
        x1 = self.primary(x)
        x2 = self.cheap(x1)
        return torch.cat([x1, x2], dim=1)


class GhostBlock(nn.Module):
    """
    GhostNet のボトルネックをかなり簡略化したもの
    stride=2 のときだけ空間分解能を 1/2 にする depthwise conv を挟む
    """
    def __init__(self, c_in, c_out, s=1):
        super().__init__()
        self.gc = GhostConv(c_in, c_out, k=1, s=1)
        self.s = s
        if s == 2:
            self.dw = nn.Conv2d(c_out, c_out, 3, 2, 1, groups=c_out, bias=False)
            self.dw_bn = nn.BatchNorm2d(c_out)
        else:
            self.dw = None
        self.act = HardSwish()

    def forward(self, x):
        x = self.gc(x)
        if self.dw is not None:
            x = self.dw(x)
            x = self.dw_bn(x)
            x = self.act(x)
        return x

class SPPF(nn.Module):
    """
    YOLOv5/8 系で使われる SPPF の簡略版
    受容野を広げるためのモジュール
    """
    def __init__(self, c_in, c_out, k=5):
        super().__init__()
        self.cv1 = ConvBNAct(c_in, c_out, 1, 1)
        self.cv2 = ConvBNAct(c_out * 4, c_out, 1, 1)
        self.k = k

    def forward(self, x):
        x = self.cv1(x)
        y1 = F.max_pool2d(x, self.k, 1, self.k // 2)
        y2 = F.max_pool2d(y1, self.k, 1, self.k // 2)
        y3 = F.max_pool2d(y2, self.k, 1, self.k // 2)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))

# class TinyBackbone(nn.Module):
#     def __init__(self, c_in=3, c_out=128):
#         super().__init__()
#         self.stem = ConvBNAct(c_in, 32, 3, 2)  # /2
#         self.b1 = ConvBNAct(32, 64, 3, 2)      # /4
#         self.b2 = ConvBNAct(64, c_out, 3, 2)   # /8
#         self.out_ch = c_out
#     def forward(self, x):
#         return self.b2(self.b1(self.stem(x)))

# class DualBackbone(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.b1 = TinyBackbone(3,128)
#         self.b2 = TinyBackbone(3,128)
#         self.merge = ConvBNAct(256, 192, 1, 1)
#     def forward(self, x):
#         f1 = self.b1(x); f2 = self.b2(x)
#         return self.merge(torch.cat([f1,f2],1))  # /8, C=192

class GhostBackbone(nn.Module):
    """
    GhostNetベースのバックボーン(future mapはstride=8で出力)
    入力: 3x640x640 -> 出力: 96x80x80
    """
    def __init__(self, c_in=3, c_out=96):
        super().__init__()
        self.stem = ConvBNAct(c_in, 16, 3, 2) # /2 -> 16x320x320
        self.b1 = GhostBlock(16, 48, s=2)   # /4 -> 48x160x160
        self.b2 = GhostBlock(48, c_out, s=2)   # /8 -> 96x80x80
        self.sppf = SPPF(c_out, c_out, k=5) # 受容野拡大, 80x80のまま
        self.out_ch = c_out

    def forward(self, x):
        return self.sppf(self.b2(self.b1(self.stem(x))))  # /8, C=96

class PPBackbone(nn.Module):
    """
    PP-LCNetベースのバックボーン(DepthSepConv+一部5x5)
    入力: 3x640x640 -> 出力: 96x80x80
    """
    def __init__(self, c_in=3, c_out=96):
        super().__init__()
        self.stem = ConvBNAct(c_in, 24, 3, 2) # /2 -> 24x320x320
        self.b1 = DepthSepConv(24, 48, k=3, s=2)   # /4 -> 48x160x160
        self.b2 = DepthSepConv(48, c_out, k=3, s=2)   # /8 -> 96x80x80
        self.out_ch = c_out

    def forward(self, x):
        return self.b2(self.b1(self.stem(x)))  # /8, C=96

class DualBackbone(nn.Module):
    """
    GhostBackbone と PPBackbone を並列に通して特徴量を結合する
    結合後 1x1 Conv で192チャネルに整形(論文の「融合後192ch」相当)
    """
    def __init__(self):
        super().__init__()
        self.backbone_ghost = GhostBackbone()
        self.backbone_pp = PPBackbone()
        merged_ch = self.backbone_ghost.out_ch + self.backbone_pp.out_ch
        self.merge = ConvBNAct(merged_ch, 192, 1, 1)

    def forward(self, x):
        f1 = self.backbone_ghost(x) # 96 x 80 x 80
        f2 = self.backbone_pp(x) # 96 x 80 x 80
        return self.merge(torch.cat([f1, f2], 1))  # 192 x 80 x 80

class SimpleNeck(nn.Module):
    def __init__(self, c=192):
        super().__init__()
        self.m = nn.Sequential(ConvBNAct(c,192,3,1), ConvBNAct(192,192,3,1))
    def forward(self, x): return self.m(x)

class DetectHead(nn.Module):
    def __init__(self, c=192, num_classes=3, anchors=[(10,13),(16,30),(33,23)]):
        super().__init__()
        self.na = len(anchors)
        self.nc = num_classes
        self.anchors = torch.tensor(anchors, dtype=torch.float32)  # pixels at stride scale (later /stride)
        self.cv = nn.Conv2d(c, self.na*(5+self.nc), 1, 1)
    def forward(self, x):
        b,c,h,w = x.shape
        p = self.cv(x)
        p = p.view(b, self.na, 5+self.nc, h, w).permute(0,1,3,4,2).contiguous()
        return p  # raw logits (tx,ty,tw,th,obj,cls...)

class DBNet(nn.Module):
    def __init__(self, num_classes=3, anchors=[(10,13),(16,30),(33,23)]):
        super().__init__()
        self.backbone = DualBackbone()
        self.neck = SimpleNeck(192)
        self.head = DetectHead(192, num_classes=num_classes, anchors=anchors)
        self.stride = 8
    def forward(self, x):
        f = self.backbone(x)
        f = self.neck(f)
        p = self.head(f)
        return p

# ----------------------------
# Target assignment
# ----------------------------
def build_grids(h, w, device, stride):
    gy, gx = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
    grid = torch.stack((gx, gy), 2).float()  # [H,W,2]
    return grid, stride

def assign_targets(labels_list, pshape, num_classes, anchors_grid, stride, img_size):
    device = anchors_grid.device
    B, na, H, W, _ = pshape
    tobj = torch.zeros((B,na,H,W), device=device)
    tcls = torch.zeros((B,na,H,W,num_classes), device=device)
    tbox = torch.zeros((B,na,H,W,4), device=device)

    grid, _ = build_grids(H, W, device, stride)
    a_wh = anchors_grid.view(na,1,1,2)  # grid units

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

        gt_anchor = torch.stack([gw/stride, gh/stride], 1)
        ious = []
        for a in range(na):
            aw, ah = a_wh[a,0,0]
            inter = torch.minimum(gt_anchor[:,0], aw) * torch.minimum(gt_anchor[:,1], ah)
            union = gt_anchor[:,0]*gt_anchor[:,1] + aw*ah - inter + 1e-9
            ious.append(inter/union)
        #近傍3x3を正例にをやめる
        # ious = torch.stack(ious, 1)
        # best_a = ious.argmax(1)

        # for i in range(lt.size(0)):
        #     a = int(best_a[i].item())
        #     # 近傍 3x3 を正例に
        #     for di in (-1,0,1):
        #         for dj in (-1,0,1):
        #             ii = int((gi_idx[i] + di).clamp(0, W-1))
        #             jj = int((gj_idx[i] + dj).clamp(0, H-1))
        #             tobj[b,a,jj,ii] = 1.0
        #             tcls[b,a,jj,ii, gcls[i]] = 1.0
        #             tbox[b,a,jj,ii] = torch.tensor([gx[i], gy[i], gw[i], gh[i]], device=device)

        ious = torch.stack(ious, 1) # [num_gt, na]
        #各GTに対して最もIoUの高いアンカーを正例に
        best_iou, best_a = ious.max(1)

        iou_pos_thr = 0.2
        pos_radius = 0

        for i in range(lt.size(0)):
            # IoUが閾値未満なら無視
            if best_iou[i] < iou_pos_thr:
                continue
            # 近傍(2*pos_radius+1)^2を正例に
            a = int(best_a[i].item())
            for di in range(-pos_radius, pos_radius+1):
                for dj in range(-pos_radius, pos_radius+1):
                    ii = int((gi_idx[i] + di).clamp(0, W-1))
                    jj = int((gj_idx[i] + dj).clamp(0, H-1))
                    tobj[b,a,jj,ii] = 1.0
                    tcls[b,a,jj,ii, gcls[i]] = 1.0
                    tbox[b,a,jj,ii] = torch.tensor([gx[i], gy[i], gw[i], gh[i]], device=device)
    return tobj, tcls, tbox

# ----------------------------
# Focal loss wrapper
# ----------------------------
def focal_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 1.5,
    reduction: str = "mean"
):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    prob = torch.sigmoid(logits)
    p_t = prob * targets + (1 - prob) * (1 - targets)
    mod = (1 - p_t).clamp_(0, 1).pow(gamma)
    loss = alpha * mod * bce
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss

def _set_lr(optimizer, lr: float):
    for g in optimizer.param_groups:
        g['lr'] = float(lr)

# ----------------------------
# Loss
# ----------------------------
class DBLoss(nn.Module):
    def __init__(self, num_classes, box=7.0, obj=1.2, cls=0.5, stride=8, img_size=640,
                 anchors=[(77, 73), (105, 212), (317, 171), (172, 429), (467, 382)],
                 focal_obj=False, focal_cls=False, focal_gamma=1.5, focal_alpha=0.25, cls_weight=(1.2, 1.0, 3.0)):
        super().__init__()
        self.register_buffer("cls_weight", torch.tensor(cls_weight, dtype=torch.float32))
        self.lw_box, self.lw_obj, self.lw_cls = box, obj, cls
        self.num_classes = num_classes
        self.stride = stride
        self.img_size = img_size
        self.focal_obj = focal_obj
        self.focal_cls = focal_cls
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.register_buffer("anchors_px", torch.tensor(anchors, dtype=torch.float32))

    def forward(self, p_raw, labels_list):
        device = p_raw.device
        B, na, H, W, D = p_raw.shape

        anchors_grid = (self.anchors_px.to(device) / self.stride)
        # raw logits
        logits_x = p_raw[...,0]
        logits_y = p_raw[...,1]
        logits_w = p_raw[...,2]
        logits_h = p_raw[...,3]
        logits_obj = p_raw[...,4]      # raw (for BCEWithLogits)
        logits_cls = p_raw[...,5:]     # raw (for BCEWithLogits)

        # decode (center offsets + anchor scaling in grid)
        px = torch.sigmoid(logits_x)
        py = torch.sigmoid(logits_y)
        pw = torch.exp(logits_w) * anchors_grid[:,0].view(na,1,1)
        ph = torch.exp(logits_h) * anchors_grid[:,1].view(na,1,1)

        grid,_ = build_grids(H,W,device,self.stride)
        gx = (grid[...,0].unsqueeze(0).unsqueeze(0) + px) * self.stride
        gy = (grid[...,1].unsqueeze(0).unsqueeze(0) + py) * self.stride
        pxywh = torch.stack([gx, gy, pw*self.stride, ph*self.stride], -1)

        # targets
        tobj, tcls, tbox = assign_targets(
            labels_list, p_raw.shape, self.num_classes, anchors_grid, self.stride, self.img_size
        )

        # box loss
        pos = tobj > 0.5
        n_pos = pos.sum().clamp(min=1)
        if pos.any():
            l_box = ciou_loss(pxywh[pos], tbox[pos]).mean()
        else:
            l_box = torch.tensor(0., device=device)

        # obj loss
        if self.focal_obj:
            l_obj = focal_bce_with_logits(logits_obj, tobj, alpha=self.focal_alpha, gamma=self.focal_gamma)
        else:
            l_obj = F.binary_cross_entropy_with_logits(logits_obj, tobj)

        # cls loss (posのみ)
        if pos.any():
            if self.focal_cls:
                loss_raw = focal_bce_with_logits(logits_cls[pos], tcls[pos], alpha=self.focal_alpha, gamma=self.focal_gamma, reduction="none")
            else:
                loss_raw = F.binary_cross_entropy_with_logits(logits_cls[pos], tcls[pos], reduction="none")
            w = self.cls_weight.view(1, self.num_classes)  # shape [1,C]
            l_cls = (loss_raw * w).mean()
        else:
            l_cls = torch.tensor(0., device=device)

        total = self.lw_box*l_box + self.lw_obj*l_obj + self.lw_cls*l_cls
        return total, {
            "l_box": float(l_box.detach()),
            "l_obj": float(l_obj.detach()),
            "l_cls": float(l_cls.detach()),
            "n_pos": int(n_pos.item()) if hasattr(n_pos, "item") else int(n_pos),
        }

def letterbox_keep_ratio_pil(img, target=640, pad_value=114):
    """PIL画像をYOLO風に640x640へレターボックス + メタ情報返却"""
    w0, h0 = img.size
    scale = target / max(w0, h0)
    w1 = int(round(w0 * scale))
    h1 = int(round(h0 * scale))
    dw = (target - w1) // 2
    dh = (target - h1) // 2

    img_resized = img.resize((w1, h1), Image.BILINEAR)
    canvas = Image.new("RGB", (target, target), (pad_value,) * 3)
    canvas.paste(img_resized, (dw, dh))

    meta = dict(scale=scale, dw=dw, dh=dh, W0=w0, H0=h0, target=target)
    return canvas, meta

@torch.no_grad()
def sahi_infer_on_image(
    model,
    img_path,
    device,
    num_classes=3,
    img_size=640,
    patch_size=640,
    overlap=0.2,
    conf_thres=0.25,
    max_det=300,
    topk_per_level=1000,
    anchors_px=None,
    iou_nms=0.5,
):
    """
    大きな滝画像(SSS)に対して SAHI でスライスしながら推論し、元画像座標系で NMS した結果を返す。

    return: boxes [N,4], scores [N], labels [N]
    """
    model.eval()
    img_big = Image.open(img_path).convert("RGB")
    W_big, H_big = img_big.size

    stride_xy = int(patch_size * (1.0 - overlap))
    stride_x = max(1, stride_xy)
    stride_y = max(1, stride_xy)

    all_boxes = []
    all_scores = []
    all_labels = []

    # モデルのアンカーを取得（decode_predictionsに渡す）
    if anchors_px is None:
        # DetectHeadの anchors をそのまま使う
        anchors_px = model.head.anchors.detach().cpu().tolist()

    for y0 in range(0, H_big, stride_y):
        y1 = min(y0 + patch_size, H_big)
        if y1 - y0 <= 0:
            continue
        for x0 in range(0, W_big, stride_x):
            x1 = min(x0 + patch_size, W_big)
            if x1 - x0 <= 0:
                continue

            patch = img_big.crop((x0, y0, x1, y1))  # ローカルパッチ (PIL)
            patch_w, patch_h = patch.size

            # レターボックスで 640x640 へ
            patch_lb, meta = letterbox_keep_ratio_pil(patch, target=img_size)
            scale = meta["scale"]
            dw = meta["dw"]
            dh = meta["dh"]

            # テンソル化
            img_tensor = torch.from_numpy(np.array(patch_lb, dtype=np.float32)).permute(2, 0, 1) / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(device)

            # 推論
            p_raw = model(img_tensor)  # [1,na,H,W,D]
            boxes, scores, labels = decode_predictions(
                p_raw,
                num_classes=num_classes,
                stride=model.stride if hasattr(model, "stride") else 8,
                conf_thres=conf_thres,
                max_det=max_det,
                topk_per_level=topk_per_level,
                anchors_px=anchors_px,
            )

            if boxes.numel() == 0:
                continue

            # decode_predictionsは 640x640 座標系なので、レターボックスとスケールを元に
            # パッチ内の元座標へ戻す
            # x,y 共に [dw,dh] を引いてスケールで割る
            boxes_patch = boxes.clone()
            # x1,x2
            boxes_patch[:, [0, 2]] = (boxes_patch[:, [0, 2]] - dw) / max(scale, 1e-9)
            # y1,y2
            boxes_patch[:, [1, 3]] = (boxes_patch[:, [1, 3]] - dh) / max(scale, 1e-9)

            # パッチ外にはみ出した分をクリップ
            boxes_patch[:, 0].clamp_(0, patch_w)
            boxes_patch[:, 2].clamp_(0, patch_w)
            boxes_patch[:, 1].clamp_(0, patch_h)
            boxes_patch[:, 3].clamp_(0, patch_h)

            # 元の大きな画像座標系に平行移動
            boxes_patch[:, [0, 2]] += x0
            boxes_patch[:, [1, 3]] += y0

            all_boxes.append(boxes_patch)
            all_scores.append(scores)
            all_labels.append(labels)

    if len(all_boxes) == 0:
        return (
            torch.zeros((0, 4), device=device),
            torch.zeros((0,), device=device),
            torch.zeros((0,), device=device),
        )

    all_boxes = torch.cat(all_boxes, dim=0)
    all_scores = torch.cat(all_scores, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # batched NMS でマージ
    keep = batched_nms(all_boxes, all_scores, all_labels, iou_nms)
    if keep.numel() > max_det:
        keep = keep[:max_det]

    return all_boxes[keep], all_scores[keep], all_labels[keep]

# ----------------------------
# Inference + NMS + mAP(0.5)
# ----------------------------
def decode_predictions(
    p_raw, num_classes, stride=8, conf_thres=0.25, max_det=100,
    topk_per_level=1000, anchors_px=None
):
    device = p_raw.device
    B, na, H, W, D = p_raw.shape
    if anchors_px is None:
        anchors_px = torch.tensor([(10,13),(16,30),(33,23)], dtype=torch.float32, device=device)
    else:
        anchors_px = torch.tensor(anchors_px, dtype=torch.float32, device=device)
    anchors_grid = anchors_px / stride

    logits_x = p_raw[...,0]; logits_y = p_raw[...,1]
    logits_w = p_raw[...,2]; logits_h = p_raw[...,3]
    logits_obj = p_raw[...,4]; logits_cls = p_raw[...,5:]

    px = torch.sigmoid(logits_x)
    py = torch.sigmoid(logits_y)
    pw = torch.exp(logits_w) * anchors_grid[:,0].view(na,1,1)
    ph = torch.exp(logits_h) * anchors_grid[:,1].view(na,1,1)
    pobj = torch.sigmoid(logits_obj)
    pcls = torch.sigmoid(logits_cls)

    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )
    gx = (grid_x.unsqueeze(0).unsqueeze(0) + px) * stride
    gy = (grid_y.unsqueeze(0).unsqueeze(0) + py) * stride
    gw = pw * stride
    gh = ph * stride

    disable_topk = (topk_per_level <= 0)
    eff_topk = max(50, topk_per_level // na)  # 未使用でもOK

    outs = []
    for b in range(B):
        boxes_list = []
        scores_list = []
        labels_list = []

        for a in range(na):
            # 信頼度 = obj * max_class_conf
            cls_max, _ = pcls[b, a].max(-1)        # [H, W]
            conf = pobj[b, a] * cls_max            # [H, W]
            conf_flat = conf.flatten()

            # ★ 分岐：Top-K を使う / 使わない
            if disable_topk:
                # しきい値で直接抽出
                idxs = torch.nonzero(conf_flat >= conf_thres, as_tuple=False).squeeze(1)
            else:
                k = min(eff_topk, conf_flat.numel())
                if k > 0:
                    topk_vals, topk_idx = torch.topk(conf_flat, k)
                    # topk からもしきい値で絞る（戻すのはインデックスに統一）
                    mask = (topk_vals >= conf_thres)
                    if mask.any():
                        idxs = topk_idx[mask]
                    else:
                        idxs = conf_flat.new_empty((0,), dtype=torch.long)
                else:
                    idxs = conf_flat.new_empty((0,), dtype=torch.long)

            # 何もなければ次のアンカーへ
            if idxs.numel() == 0:
                continue

            y = (idxs // W)
            x = (idxs %  W)

            # 座標変換
            x_c = gx[b, a, y, x];  y_c = gy[b, a, y, x]
            w   = gw[b, a, y, x];  h   = gh[b, a, y, x]
            x1 = x_c - w * 0.5; y1 = y_c - h * 0.5
            x2 = x_c + w * 0.5; y2 = y_c + h * 0.5
            boxes = torch.stack([x1, y1, x2, y2], dim=1)

            # ★ スコアは分岐に関わらず conf_flat[idxs] を使えば整合が取れる
            scores = conf_flat[idxs]

            # ラベルは最大クラスのインデックス
            labels = pcls[b, a, y, x].argmax(-1).long()

            boxes_list.append(boxes)
            scores_list.append(scores)
            labels_list.append(labels)

        if len(boxes_list) == 0:
            outs.append((
                torch.zeros((0,4), device=device),
                torch.zeros((0,),   device=device),
                torch.zeros((0,),   device=device).long()
            ))
            continue

        boxes  = torch.cat(boxes_list,  dim=0)
        scores = torch.cat(scores_list, dim=0)
        labels = torch.cat(labels_list, dim=0)

        # ★ クラス別しきい値(thr_map)は撤去済み。ここで再しきい値は不要。
        # if you really want a final pass, keep it identical to conf_thres:
        # m = scores >= conf_thres
        # boxes, scores, labels = boxes[m], scores[m], labels[m]

        # max_det 制限
        if boxes.size(0) > max_det:
            topv, topi = torch.topk(scores, max_det)
            boxes = boxes[topi]; scores = topv; labels = labels[topi]

        keep = batched_nms(boxes, scores, labels, 0.5)
        outs.append((boxes[keep], scores[keep], labels[keep]))

    return outs

def compute_map50(preds, gts, img_size=640):
    aps=[]
    for (p_boxes,p_scores,p_labels), gt in zip(preds, gts):
        gt_boxes = xywhn_to_xyxy(gt[:,1:], img_size, img_size) if gt.numel() else torch.zeros((0,4),device=p_boxes.device)
        gt_cls = gt[:,0].long() if gt.numel() else torch.zeros((0,),dtype=torch.long,device=p_boxes.device)
        matched = torch.zeros((gt_boxes.size(0),), dtype=torch.bool, device=p_boxes.device)
        order = torch.argsort(p_scores, descending=True)
        tp=[]; fp=[]
        for idx in order:
            if p_boxes.size(0)==0: break
            pb = p_boxes[idx:idx+1]
            pl = int(p_labels[idx].item())
            if gt_boxes.size(0)==0:
                tp.append(0); fp.append(1); continue
            ious = box_iou(pb, gt_boxes).squeeze(0)
            iou, j = (ious.max(0))
            if iou >= 0.5 and (not matched[j]) and (pl==int(gt_cls[j].item())):
                tp.append(1); fp.append(0); matched[j]=True
            else:
                fp.append(1); tp.append(0)
        if not tp and gt_boxes.size(0)==0:
            aps.append(1.0); continue
        if not tp and tp!=[]:
            aps.append(0.0); continue
        tp = torch.tensor(tp, dtype=torch.float32, device=p_boxes.device)
        fp = torch.tensor(fp, dtype=torch.float32, device=p_boxes.device)
        cum_tp = torch.cumsum(tp,0); cum_fp = torch.cumsum(fp,0)
        recalls = cum_tp / max(1, gt_boxes.size(0))
        precis = cum_tp / torch.clamp(cum_tp+cum_fp, min=1)
        ps=[]
        for r in torch.linspace(0,1,11,device=p_boxes.device):
            mask = recalls>=r
            p = precis[mask].max() if mask.any() else torch.tensor(0.,device=p_boxes.device)
            ps.append(p)
        aps.append(torch.stack(ps).mean().item())
    return float(np.mean(aps)) if aps else 0.0

def compute_map50_per_class(preds, gts, img_size=640, num_classes=3):
    # ざっくり per-class：各クラスのみを対象に評価
    out = {}
    for c in range(num_classes):
        sub_preds = []
        sub_gts = []
        for (p_boxes,p_scores,p_labels), gt in zip(preds, gts):
            m = (p_labels.long()==c)
            sub_preds.append((p_boxes[m], p_scores[m], p_labels[m]))
            if gt.numel():
                mgt = (gt[:,0].long()==c)
                sub_gts.append(gt[mgt])
            else:
                sub_gts.append(torch.zeros((0,5), device=p_boxes.device))
        out[c] = compute_map50(sub_preds, sub_gts, img_size=img_size)
    return out

def _ap_at_iou50(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels):
    # Dataset-level AP@0.5 (11-point interpolation). 前提：pred/gtは既に必要クラスでフィルタ済み
    device = pred_scores.device if hasattr(pred_scores, "device") else torch.device("cpu")

    # 例外系
    if gt_boxes.numel() == 0 and pred_boxes.numel() == 0:
        return 1.0
    if gt_boxes.numel() == 0 and pred_boxes.numel() > 0:
        return 0.0
    if pred_boxes.numel() == 0:
        return 0.0

    # スコア降順
    order = torch.argsort(pred_scores, descending=True)
    pb = pred_boxes[order]
    pl = pred_labels[order].long()

    matched = torch.zeros((gt_boxes.size(0),), dtype=torch.bool, device=pb.device)
    tps, fps = [], []
    for i in range(pb.size(0)):
        box = pb[i:i+1]
        if gt_boxes.size(0) == 0:
            tps.append(0); fps.append(1); continue
        ious = box_iou(box, gt_boxes).squeeze(0)
        iou, j = (ious.max(0))
        # IoU>=0.5 かつ まだ未マッチ かつ クラス一致 でTP
        if iou >= 0.5 and (not matched[j]) and (pl[i] == gt_labels[j].long()):
            matched[j] = True
            tps.append(1); fps.append(0)
        else:
            tps.append(0); fps.append(1)

    tps = torch.tensor(tps, dtype=torch.float32, device=pb.device)
    fps = torch.tensor(fps, dtype=torch.float32, device=pb.device)
    cum_tp = torch.cumsum(tps, 0)
    cum_fp = torch.cumsum(fps, 0)
    precision = cum_tp / (cum_tp + cum_fp).clamp(min=1e-9)
    recall = cum_tp / max(1, int(gt_boxes.size(0)))

    ap = 0.0
    for t in torch.linspace(0, 1, steps=11, device=pb.device):
        mask = (recall >= t)
        ap += float(precision[mask].max().item() if mask.any() else 0.0)
    return ap / 11.0

def compute_map50_dataset_level(preds, gts, num_classes, img_size=640):
    # preds: List[(boxes, scores, labels)], gts: List[tensor[N,5]]
    # 1) 連結
    all_boxes = torch.cat([b for b,_,_ in preds], 0)
    all_scores= torch.cat([s for _,s,_ in preds], 0)
    all_labels= torch.cat([l.long() for _,_,l in preds], 0)
    all_gt_boxes=[]; all_gt_cls=[]
    for gt in gts:
        if gt.numel():
            all_gt_boxes.append(xywhn_to_xyxy(gt[:,1:], img_size, img_size))
            all_gt_cls.append(gt[:,0].long())
    all_gt_boxes = torch.cat(all_gt_boxes, 0) if all_gt_boxes else all_boxes.new_zeros((0,4))
    all_gt_cls   = torch.cat(all_gt_cls, 0)   if all_gt_cls else all_labels.new_zeros((0,), dtype=torch.long)

    # 2) overall（全クラスまとめて）
    overall_ap = _ap_at_iou50(all_boxes, all_scores, all_labels, all_gt_boxes, all_gt_cls)

    # 3) per-class
    per_class = {}
    for c in range(num_classes):
        m_pred = (all_labels==c)
        m_gt   = (all_gt_cls==c)
        if m_gt.sum()==0:
            per_class[c] = float('nan')
            continue
        per_class[c] = _ap_at_iou50(all_boxes[m_pred], all_scores[m_pred], all_labels[m_pred],
                                    all_gt_boxes[m_gt], all_gt_cls[m_gt])
    return overall_ap, per_class

# ----------------------------
# K-means anchors (IoU distance + log space + stratified)
# ----------------------------
def kmeans_anchors_wh(dataset_root: str, run_name: str, img_size: int,
                      k: int = 3, iters: int = 1000, seed: int = 0,
                      use_log=True, per_class_balance=True, max_per_class=1000):
    rng = np.random.default_rng(seed)
    root = Path(dataset_root)/"runs"/run_name
    lbl_dir = root/"labels"

    by_cls = {0:[],1:[],2:[]}
    for p in lbl_dir.glob("*.txt"):
        txt = p.read_text().strip().splitlines()
        for line in (txt or []):
            ss = line.strip().split()
            if len(ss) != 5: continue
            c, _, _, w, h = map(float, ss)
            by_cls[int(c)].append([w*img_size, h*img_size])

    if per_class_balance:
        X = []
        for c in by_cls:
            arr = np.array(by_cls[c], dtype=np.float32)
            if arr.size == 0: continue
            if arr.shape[0] > max_per_class:
                idx = rng.choice(arr.shape[0], size=max_per_class, replace=False)
                arr = arr[idx]
            X.append(arr)
        if not X: raise RuntimeError("no labels found to compute anchors")
        X = np.concatenate(X, axis=0)
    else:
        X = np.array(sum(by_cls.values(), []), dtype=np.float32)
        if X.size == 0: raise RuntimeError("no labels found to compute anchors")

    Y = np.log(X + 1e-9) if use_log else X

    def iou_wh(a, b):
        a_wh = np.exp(a) if use_log else a
        b_wh = np.exp(b) if use_log else b
        inter = np.minimum(a_wh[:, None, 0], b_wh[None, :, 0]) * np.minimum(a_wh[:, None, 1], b_wh[None, :, 1])
        area_a = a_wh[:, 0] * a_wh[:, 1]
        area_b = b_wh[:, 0] * b_wh[:, 1]
        union = area_a[:, None] + area_b[None, :] - inter + 1e-9
        return inter / union

    centroids = []
    idx0 = rng.integers(0, len(Y))
    centroids.append(Y[idx0])
    for _ in range(1, k):
        ious = iou_wh(Y, np.stack(centroids, axis=0))
        dist = 1.0 - ious.max(axis=1)
        probs = dist / (dist.sum() + 1e-9)
        idx = rng.choice(len(Y), p=probs)
        centroids.append(Y[idx])
    C = np.stack(centroids, axis=0)

    for _ in range(iters):
        ious = iou_wh(Y, C)
        r = np.argmax(ious, axis=1)
        newC = []
        for i in range(k):
            sel = (r == i)
            newC.append(Y[sel].mean(axis=0) if np.any(sel) else C[i])
        newC = np.stack(newC, axis=0)
        if np.allclose(newC, C, atol=1e-5):
            C = newC; break
        C = newC

    C_px = np.exp(C) if use_log else C
    areas = C_px[:,0]*C_px[:,1]
    order = np.argsort(areas)
    C_px = C_px[order]
    C_px[:,0] = np.clip(C_px[:,0], 6, img_size*0.9)
    C_px[:,1] = np.clip(C_px[:,1], 6, img_size*0.9)
    anchors = [(float(round(w)), float(round(h))) for w,h in C_px]
    return anchors

def analyze_anchor_iou(dataset, anchors_px, img_size):
    all_ious = []
    for i in range(len(dataset)):
        _, labels, _ = dataset[i]  # labels: [N,5] (cls,cx,cy,w,h in 0-1)
        if labels.numel() == 0:
            continue
        w = labels[:, 3] * img_size
        h = labels[:, 4] * img_size
        wh = torch.stack([w, h], dim=1).cpu().numpy()  # [N,2]

        for bw, bh in wh:
            best = 0.0
            for aw, ah in anchors_px:
                inter = min(bw, aw) * min(bh, ah)
                union = bw * bh + aw * ah - inter
                iou = inter / union
                if iou > best:
                    best = iou
            all_ious.append(best)

    all_ious = np.array(all_ious)
    print(f"[anchor-iou] num={len(all_ious)} "
          f"mean={all_ious.mean():.3f} "
          f"median={np.median(all_ious):.3f} "
          f"ratio_iou<0.2={np.mean(all_ious < 0.2):.3f} "
          f"ratio_iou<0.1={np.mean(all_ious < 0.1):.3f}")

# ----------------------------
# EMA
# ----------------------------
class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.9998):
        self.ema = deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.copy_(v * d + msd[k] * (1.0 - d))
            else:
                v.copy_(msd[k])


class HumanFavoredDataset(torch.utils.data.Dataset):
    def __init__(self, ds, idx_pos_by_cls, idx_empty, p_empty=0.2):
        self.ds = ds
        self.idx_human = idx_pos_by_cls[2]  # human を含む index
        self.idx_ship  = idx_pos_by_cls[1]
        self.idx_air   = idx_pos_by_cls[0]
        self.idx_empty = idx_empty
        self.p_empty = p_empty

    def __len__(self):
        # 適当に元の train サイズに合わせる
        return len(self.idx_ship) + len(self.idx_human) + len(self.idx_air)

    def __getitem__(self, _):
        r = random.random()
        if r < 0.4 and self.idx_human:          # 40%: human
            i = random.choice(self.idx_human)
        elif r < 0.7 and self.idx_air:          # 30%: aircraft
            i = random.choice(self.idx_air)
        elif r < 0.9 and self.idx_ship:         # 20%: ship
            i = random.choice(self.idx_ship)
        else:                                   # 10%: empty
            i = random.choice(self.idx_empty) if self.idx_empty else random.choice(self.idx_ship)
        return self.ds[i]

# ----------------------------
# Train / Eval
# ----------------------------
def train(args):
    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # data
    ds = SonarYOLODataset(args.dataset_root, args.run_name, args.img_size, augment=False)
    idx_pos_by_cls, idx_empty = build_indices_by_class(ds)
    print({k:len(v) for k,v in idx_pos_by_cls.items()}, "empty:", len(idx_empty))

    pos_all = sorted(set().union(*idx_pos_by_cls.values())) if len(idx_pos_by_cls) > 0 else []

    val_ratio = args.val_split  # 例: 0.20
    rnd_pos   = random.Random(args.seed)
    rnd_emp   = random.Random(args.seed + 1)

    # 正例を含む画像の 8:2 分割
    pos_all_shuf = list(pos_all)          # コピー
    rnd_pos.shuffle(pos_all_shuf)
    nva_pos = max(1, int(val_ratio * len(pos_all_shuf))) if len(pos_all_shuf) > 0 else 0
    va_pos = set(pos_all_shuf[:nva_pos])
    tr_pos = set(pos_all_shuf[nva_pos:])

    # 空画像（アノテ無し）も同率で 8:2
    nva_empty = int(val_ratio * len(idx_empty))
    if nva_empty > 0:
        va_empty = set(rnd_emp.sample(idx_empty, nva_empty))
    else:
        va_empty = set()
    tr_empty = set([i for i in idx_empty if i not in va_empty])

    # 学習/検証の最終インデックス
    tr_idx = sorted(tr_pos.union(tr_empty))
    va_idx = sorted(va_pos.union(va_empty))

    print(f"[split] train={len(tr_idx)}  val={len(va_idx)}  "
        f"(pos_tr={len(tr_pos)}, pos_val={len(va_pos)}, "
        f"empty_tr={len(tr_empty)}, empty_val={len(va_empty)})")

    ds_tr = torch.utils.data.Subset(ds, tr_idx)
    ds_va = torch.utils.data.Subset(ds, va_idx)

    def _worker_init_fn(worker_id):
        try:
            import cv2; cv2.setNumThreads(0)
        except Exception:
            pass
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        torch.set_num_threads(1)

    # curriculum: 空画像20%混ぜる
    ds_tr_cur = HumanFavoredDataset(ds, idx_pos_by_cls, list(tr_empty), p_empty=0.2)

    dl_tr = DataLoader(
        ds_tr_cur,
        batch_size=args.batch_size, num_workers=args.workers,
        shuffle=True, sampler=None, collate_fn=collate_fn, pin_memory=True,
        persistent_workers=True, prefetch_factor=4, worker_init_fn=_worker_init_fn,
    )
    dl_va = DataLoader(
        ds_va, batch_size=args.batch_size, num_workers=args.workers,
        shuffle=False, collate_fn=collate_fn, pin_memory=True,
        persistent_workers=True, prefetch_factor=4, worker_init_fn=_worker_init_fn,
    )

    # anchors
    if args.anchors_kmeans_only or args.auto_anchors:
        anchors = kmeans_anchors_wh(
            args.dataset_root, args.run_name, args.img_size,
            k=args.anchors_k, iters=args.anchors_iters, seed=args.seed,
            use_log=args.anchors_log, per_class_balance=args.anchors_stratified, max_per_class=args.anchors_max_per_class
        )
        print(f"[auto-anchors] suggested anchors (px): {anchors}")
        if args.anchors_kmeans_only:
            return
    else:
        anchors = [(32, 40), (77, 73), (105, 212), (467, 382)]

    # model / loss / opt / ema
    model = DBNet(num_classes=args.num_classes, anchors=anchors).to(device)
    criterion = DBLoss(
        num_classes=args.num_classes, img_size=args.img_size, anchors=anchors,
        focal_obj=args.focal_obj, focal_cls=args.focal_cls,
        focal_gamma=args.focal_gamma, focal_alpha=args.focal_alpha
    ).to(device)
    optim = torch.optim.Adam(
        model.parameters(), 
        lr=args.init_lr, 
        betas=(0.937, 0.999),
        weight_decay=5e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)
    ema = ModelEMA(model, decay=args.ema_decay) if args.ema else None

    # logging dir
    exp_dir = Path("runs")/time.strftime("exp_%Y%m%d_%H%M%S")
    exp_dir.mkdir(parents=True, exist_ok=True)

    if args.save_config:
        try:
            resolved = vars(args).copy()
            (exp_dir / "config_resolved.yaml").write_text(
                yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True) if yaml else json.dumps(resolved, indent=2, ensure_ascii=False)
            )
        except Exception as e:
            print(f"[warn] failed to save resolved config: {e}")

    csv_path = exp_dir/"metrics.csv"
    logs_root = Path("runs_logs"); logs_root.mkdir(parents=True, exist_ok=True)
    csv_path_mirror = logs_root/f"{exp_dir.name}.csv"
    header = ["epoch","map50","l_box","l_obj","l_cls","n_pos","val_preds_kept","val_gt_images","val_gt_boxes","ap_aircraft","ap_ship","ap_human"]
    for pth in [csv_path, csv_path_mirror]:
        with open(pth, "w", newline="") as f:
            csv.writer(f).writerow(header)

    # 1バッチだけ sanity check（学習ループの前で1回だけ）
    imgs, lbls_list, _ = next(iter(dl_tr))
    tot = sum(int(l.size(0)) for l in lbls_list)
    if tot > 0:
        all_lbls = torch.cat([l for l in lbls_list if l.numel()], 0)
        w, h = all_lbls[:,3], all_lbls[:,4]
        print(f"[sanity] labels: {tot}  w_med={w.median():.3f}  h_med={h.median():.3f} "
            f"min/max cx={all_lbls[:,1].min():.3f}/{all_lbls[:,1].max():.3f} "
            f"cy={all_lbls[:,2].min():.3f}/{all_lbls[:,2].max():.3f}")

    for ep in range(args.epochs):
        if ep < args.warmup_epochs:
            # 0 → init_lr を線形
            lr_now = args.init_lr * float(ep + 1) / float(max(1, args.warmup_epochs))
        else:
            # 昇圧後は final_lr を維持
            lr_now = args.final_lr
        print(f"[ep {ep+1}/{args.epochs}] current_lr={lr_now:.6f}")
        _set_lr(optim, lr_now)
        model.train()
        m_logs = []
        for imgs, labels_list, _ in dl_tr:
            imgs = imgs.to(device)
            labels_list = [l.to(device) for l in labels_list]
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=args.amp):
                p = model(imgs)
                loss, logs = criterion(p, labels_list)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            if ema is not None:
                ema.update(model)
            m_logs.append(logs)

        avg = {k: float(np.mean([x[k] for x in m_logs])) for k in m_logs[0].keys()} if m_logs else {"l_box":0,"l_obj":0,"l_cls":0,"n_pos":0}
        print(f"[ep {ep+1}/{args.epochs}] loss: box={avg['l_box']:.3f} obj={avg['l_obj']:.3f} cls={avg['l_cls']:.3f} npos={avg['n_pos']:.1f}")

        eval_conf = args.conf_thres
        preds = decode_predictions(
            p, args.num_classes, stride=8,
            conf_thres=eval_conf, max_det=args.max_det, topk_per_level=args.topk_per_level,
            anchors_px=anchors
        )

        # eval (EMA優先)
        eval_model = ema.ema if ema is not None else model
        eval_model.eval()
        all_preds=[]; all_gts=[]
        with torch.no_grad():
            for imgs, labels_list, _ in dl_va:
                imgs = imgs.to(device)
                p = eval_model(imgs)
                eval_conf = args.conf_thres
                preds = decode_predictions(
                    p, args.num_classes, stride=8,
                    conf_thres=eval_conf, max_det=args.max_det, topk_per_level=args.topk_per_level,
                    anchors_px=anchors
                )
                all_preds += preds
                all_gts   += [l.to(device) for l in labels_list]

        gt_counts = {0:0,1:0,2:0}
        for gt in all_gts:
            if gt.numel():
                for c in [0,1,2]:
                    gt_counts[c] += int((gt[:,0].long()==c).sum())

        # 予測カウント
        pred_counts = {0:0,1:0,2:0}
        for (b,s,l) in all_preds:
            for c in [0,1,2]:
                pred_counts[c] += int((l.long()==c).sum())

        print("  GT counts (val):", gt_counts)
        print("  Pred counts (after NMS):", pred_counts)
        
        map50, pc = compute_map50_dataset_level(
            all_preds, all_gts,
            num_classes=args.num_classes,
            img_size=args.img_size
        )

        # 統計
        val_gt_imgs  = sum(1 for l in all_gts if l.numel() > 0)
        val_gt_boxes = sum(int(l.size(0)) for l in all_gts if l.numel() > 0)
        val_preds_kept = sum(int(x[0].size(0)) for x in all_preds)

        # （必要ならログ用に取り出し）
        ap_air  = pc.get(0, float('nan'))
        ap_ship = pc.get(1, float('nan'))
        ap_human= pc.get(2, float('nan'))

        print(f"  val mAP@0.5: {map50:.4f}")
        print(f"  val gt images: {val_gt_imgs}  boxes: {val_gt_boxes}")
        print(f"  val preds kept: {val_preds_kept} (<= {len(dl_va.dataset)*args.max_det})")
        def _fmt_ap(x):
            return "NA" if (isinstance(x,float) and (x!=x)) else f"{x:.6f}"  # NaN判定

        print("  per-class AP@0.5:", {
            0: _fmt_ap(pc.get(0, float('nan'))),
            1: _fmt_ap(pc.get(1, float('nan'))),
            2: _fmt_ap(pc.get(2, float('nan'))),
        })

        # save
        ckpt = {"model": model.state_dict(), "epoch": ep, "args": vars(args)}
        torch.save(ckpt, exp_dir/f"ckpt_ep{ep+1}.pt")
        if ema is not None:
            ckpt_ema = {"model": eval_model.state_dict(), "epoch": ep, "args": vars(args), "ema_decay": args.ema_decay}
            torch.save(ckpt_ema, exp_dir/f"ckpt_ep{ep+1}_ema.pt")
        row = [ep+1, float(map50), avg["l_box"], avg["l_obj"], avg["l_cls"], avg["n_pos"],
               int(val_preds_kept), int(val_gt_imgs), int(val_gt_boxes),
               float(ap_air), float(ap_ship), float(ap_human)]
        for pth in [csv_path, csv_path_mirror]:
            with open(pth, "a", newline="") as f:
                csv.writer(f).writerow(row)

    print(f"Done. Artifacts -> {exp_dir}")

# ----------------------------
# CLI / YAML
# ----------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    # core
    ap.add_argument("--dataset-root", type=str, required=False, default="./dataset/SCTD_yolo")
    ap.add_argument("--run-name", type=str, required=False, default="SCTD_train")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--img-size", type=int, default=640)
    ap.add_argument("--num-classes", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    #ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--seed", type=int, default=0)

    # eval filtering
    ap.add_argument("--conf-thres", type=float, default=0.30)
    ap.add_argument("--max-det", type=int, default=60)
    ap.add_argument("--topk-per-level", type=int, default=300)

    # focal
    ap.add_argument("--focal-obj", action="store_true")
    ap.add_argument("--focal-cls", action="store_true")
    ap.add_argument("--focal-gamma", type=float, default=1.5)
    ap.add_argument("--focal-alpha", type=float, default=0.25)

    # anchors
    ap.add_argument("--auto-anchors", action="store_true")
    ap.add_argument("--anchors-kmeans-only", action="store_true")
    ap.add_argument("--analyze-anchors-only", action="store_true", help="GTと現在のanchorsのIoU分布を解析し終了")
    ap.add_argument("--anchors-k", type=int, default=5)
    ap.add_argument("--anchors-iters", type=int, default=1000)
    ap.add_argument("--anchors-log", action="store_true", help="k-means in log-space")
    ap.add_argument("--anchors-stratified", action="store_true", help="class-balanced sampling for k-means")
    ap.add_argument("--anchors-max-per-class", type=int, default=1000)

    # EMA
    ap.add_argument("--ema", action="store_true")
    ap.add_argument("--ema-decay", type=float, default=0.9998)

    ap.add_argument("--warmup-epochs", type=int, default=3)            # 3ep warmup
    ap.add_argument("--init-lr", type=float, default=1e-3)             # 0.001
    ap.add_argument("--final-lr", type=float, default=1e-2)            # 0.01
    ap.add_argument("--val-split", type=float, default=0.20)

    # SAHI
    ap.add_argument("--sahi-infer", action="store_true", help="大きなSSS滝画像に対してSAHI推論を行うモード")
    ap.add_argument("--sahi-image", type=str, default="", help="SAHI推論をかけたい元画像パス")

    # logging
    ap.add_argument("--save-config", action="store_true")
    ap.add_argument("--config", type=str, default=None, help="yaml config to load first")
    args = ap.parse_args()

    # load YAML (if provided)
    if args.config:
        if yaml is None:
            raise RuntimeError("PyYAML is not installed, cannot load --config")
        cfg = yaml.safe_load(Path(args.config).read_text())
        for k, v in cfg.items():
            if hasattr(args, k):
                setattr(args, k, v)

    return args

if __name__ == "__main__":
    args = parse_args()
    # sensible defaults for anchors KMeans
    if args.auto_anchors or args.anchors_kmeans_only:
        if not args.anchors_log: args.anchors_log = True
        if not args.anchors_stratified: args.anchors_stratified = True
    if args.analyze_anchors_only:
        dataset = SonarYOLODataset(args.dataset_root, args.run_name, args.img_size, augment=False)
        anchors_px = [
            (93, 198),   # human / 小さめターゲット（縦長）
            (299, 133),  # 中規模 ship（横長）
            (504, 477),  # 画面の大半を占める巨大 ship / aircraft
        ]  # or kmeansで出したもの
        analyze_anchor_iou(dataset, anchors_px, img_size=args.img_size)
        import sys
        sys.exit(0)
     # SAHI推論だけしたい場合（学習せずに終了）
    if args.sahi_infer:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        anchors = [(32, 40), (77, 73), (105, 212), (467, 382)]  # 学習時と揃える
        model = DBNet(num_classes=args.num_classes, anchors=anchors).to(device)
        # ここで学習済み重みをロード
        ckpt = torch.load(args.weights, map_location=device)
        model.load_state_dict(ckpt["model"])

        boxes, scores, labels = sahi_infer_on_image(
            model,
            args.sahi_image,
            device,
            num_classes=args.num_classes,
            img_size=args.img_size,
            patch_size=args.img_size,
            overlap=0.2,
            conf_thres=args.conf_thres,
            max_det=args.max_det,
            topk_per_level=args.topk_per_level,
        )

        print("SAHI detections:", len(boxes))
        # ここで可視化やCSV保存など、好きな処理を追加
        import sys
        sys.exit(0)
    train(args)
