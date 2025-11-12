#!/usr/bin/env python3
import argparse, sys
from pathlib import Path
import csv
import statistics as stats

def parse_line(line):
    # YOLO: cls cx cy w h (normalized)
    p = line.strip().split()
    if len(p) != 5 and len(p) != 6:
        return None
    if len(p) == 6:
        # "filename cls cx cy w h" 形式に対処したい場合（保険）
        p = p[1:]
    try:
        cls = int(p[0]); cx=float(p[1]); cy=float(p[2]); w=float(p[3]); h=float(p[4])
    except:
        return None
    # 正当性チェック
    if not (0<=cx<=1 and 0<=cy<=1 and 0<w<=1 and 0<h<=1):
        # 一部フォーマット違いはここで弾く
        return None
    return cls, cx, cy, w, h

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-dir", required=True, help="YOLO label txt dir")
    ap.add_argument("--img-size", type=int, default=640, help="training input size (for px conversion)")
    ap.add_argument("--out-csv", default="labels_concat.csv")
    ap.add_argument("--tiny_wh_thresh", type=float, default=0.02, help="tiny threshold for w or h (normalized)")
    ap.add_argument("--tiny_area_thresh", type=float, default=0.002, help="tiny threshold for area (normalized)")
    ap.add_argument("--edge_margin", type=float, default=0.05, help="edge if center within this margin to border")
    args = ap.parse_args()

    lbl_dir = Path(args.labels_dir)
    if not lbl_dir.exists():
        print(f"[ERR] labels dir not found: {lbl_dir}", file=sys.stderr); sys.exit(1)

    rows = []  # filename, cls, cx, cy, w, h, area, aspect, w_px, h_px
    cls_counts = {}
    files = sorted(lbl_dir.glob("*.txt"))
    if not files:
        print(f"[ERR] no txt files in {lbl_dir}", file=sys.stderr); sys.exit(1)

    for f in files:
        with open(f, "r", encoding="utf-8") as fp:
            for line in fp:
                rec = parse_line(line)
                if rec is None: 
                    continue
                c, cx, cy, w, h = rec
                area = w*h
                aspect = (w/h) if h>0 else 0.0
                rows.append([f.name, c, cx, cy, w, h, area, aspect, w*args.img_size, h*args.img_size])
                cls_counts[c] = cls_counts.get(c, 0) + 1

    # 書き出し
    with open(args.out_csv, "w", newline="", encoding="utf-8") as fo:
        cw = csv.writer(fo)
        cw.writerow(["file","cls","cx","cy","w","h","area","aspect","w_px","h_px"])
        cw.writerows(rows)

    # サマリ
    n_boxes = len(rows)
    uniq_classes = sorted(cls_counts.keys())
    print("=== Summary ===")
    print(f"labels dir     : {lbl_dir}")
    print(f"img size       : {args.img_size}")
    print(f"files (txt)    : {len(files)}")
    print(f"boxes (total)  : {n_boxes}")
    print(f"classes found  : {uniq_classes}  (counts: {cls_counts})")

    # クラス別統計
    def perc(x, tot): 
        return 0.0 if tot==0 else 100.0*float(x)/float(tot)

    # 便利なビュー
    by_cls = {c: [] for c in uniq_classes}
    for r in rows:
        _, c, cx, cy, w, h, area, aspect, wpx, hpx = r
        by_cls[c].append((cx,cy,w,h,area,aspect,wpx,hpx))

    print("\n--- Per-class stats ---")
    for c in uniq_classes:
        L = by_cls[c]
        n = len(L)
        if n == 0: 
            continue
        ws = [x[2] for x in L]; hs=[x[3] for x in L]; areas=[x[4] for x in L]; aspects=[x[5] for x in L]
        wpxs=[x[6] for x in L]; hpxs=[x[7] for x in L]
        tiny_wh = sum(1 for (cx,cy,w,h,_,_,_,_) in L if (w<args.tiny_wh_thresh or h<args.tiny_wh_thresh))
        tiny_area = sum(1 for (cx,cy,w,h,area,_,_,_) in L if (area<args.tiny_area_thresh))
        edge = sum(1 for (cx,cy,_,_,_,_,_,_) in L if (cx<args.edge_margin or cx>1-args.edge_margin or cy<args.edge_margin or cy>1-args.edge_margin))
        print(f"cls={c:>2}  n={n:>4} | "
              f"w_px median={stats.median(wpxs):5.1f}  h_px median={stats.median(hpxs):5.1f}  "
              f"area_med={stats.median(areas):.4f}  asp_med={stats.median(aspects):.2f} | "
              f"tiny_wh<{args.tiny_wh_thresh}: {tiny_wh} ({perc(tiny_wh,n):.1f}%)  "
              f"tiny_area<{args.tiny_area_thresh}: {tiny_area} ({perc(tiny_area,n):.1f}%)  "
              f"edge<{args.edge_margin}: {edge} ({perc(edge,n):.1f}%)")

    # 追加チェック：IDズレの可能性
    # コード側の想定: 0: aircraft, 1: ship, 2: human
    expected_ids = {0,1,2}
    found_ids = set(uniq_classes)
    if not found_ids.issubset(expected_ids):
        extra = sorted(list(found_ids - expected_ids))
        print(f"\n[WARN] unexpected class ids in labels: {extra}  (code expects 0,1,2)")
    if len(found_ids) < 3:
        missing = sorted(list(expected_ids - found_ids))
        print(f"[WARN] missing class ids in labels: {missing}")

    # 極端アスペクト比の警告
    high_asp = {c: sum(1 for (_,_,_,_,_,asp,_,_) in L if asp>5 or asp<0.2) for c,L in by_cls.items()}
    for c in uniq_classes:
        if high_asp[c] > 0:
            print(f"[NOTE] cls={c}: extreme aspect boxes: {high_asp[c]}")

    print(f"\nCSV written: {args.out_csv}")

if __name__ == "__main__":
    main()