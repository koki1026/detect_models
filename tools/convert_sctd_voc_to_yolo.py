import os, argparse, xml.etree.ElementTree as ET
from pathlib import Path
from shutil import copy2

def parse_class_map(pairs): #クラス名のエイリアスを解析する
    m = {}
    for kv in pairs:
        if "=" not in kv: 
            raise ValueError(f"--class-map '{kv}' should be like 'aircraft=plane'")
        k,v = kv.split("=",1)
        m[k.strip()] = v.strip()
    return m

def voc_to_yolo(size_wh, box_xyxy): # VOC形式のバウンディングボックスをYOLO形式に変換する
    W,H = size_wh
    xmin,ymin,xmax,ymax = box_xyxy
    xc = (xmin + xmax) / (2.0 * W)
    yc = (ymin + ymax) / (2.0 * H)
    bw = (xmax - xmin) / W
    bh = (ymax - ymin) / H
    return xc, yc, bw, bh

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voc-root", required=True, help="SCTDのルート（Annotations/, JPEGImages/, ImageSets/...）")
    ap.add_argument("--dataset-root", required=True, help="あなたの dataset_root")
    ap.add_argument("--run-name", default="SCTD_train", help="runs/<RUN_NAME>")
    ap.add_argument("--split-file", default="ImageSets/Main/trainval.txt", help="相対パス or 絶対パス")
    ap.add_argument("--classes", nargs="+", required=True, help="YOLOクラス順（例: plane shipwreck drowning-victim）")
    ap.add_argument("--class-map", nargs="*", default=[], help="別名→正規名の対応（例: aircraft=plane drowning_victim=drowning-victim）")
    args = ap.parse_args()

    voc = Path(args.voc_root)
    ann_dir = voc / "Annotations"
    img_dir = voc / "JPEGImages"
    split_path = Path(args.split_file)
    if not split_path.is_absolute():
        split_path = voc / args.split_file

    cls_list = args.classes
    name_to_id = {n:i for i,n in enumerate(cls_list)}
    alias = parse_class_map(args.class_map)

    out_run = Path(args.dataset_root) / "runs" / args.run_name
    out_img = out_run / "sonar"
    out_lbl = out_run / "labels"
    out_run.mkdir(parents=True, exist_ok=True)
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    # 集計
    n_img = n_with_gt = n_box = n_skip_xml = n_skip_cls = 0

    with open(split_path, "r") as f:
        ids = [ln.strip() for ln in f if ln.strip()]

    for id_ in ids:
        xml_path = ann_dir / f"{id_}.xml"
        jpg_path = img_dir / f"{id_}.jpg"
        png_path = img_dir / f"{id_}.png"
        if not xml_path.exists():
            n_skip_xml += 1
            continue
        img_path = jpg_path if jpg_path.exists() else (png_path if png_path.exists() else None)
        if img_path is None:
            # 画像拡張子が違う場合は必要に応じて追加
            continue

        root = ET.parse(xml_path).getroot()
        size = root.find("size")
        W = int(size.find("width").text); H = int(size.find("height").text)

        lines = []
        for obj in root.findall("object"):
            raw = obj.find("name").text.strip()
            cls_name = alias.get(raw, raw)  # マッピング適用
            if cls_name not in name_to_id:
                n_skip_cls += 1
                continue
            bb = obj.find("bndbox")
            xmin = float(bb.find("xmin").text); ymin = float(bb.find("ymin").text)
            xmax = float(bb.find("xmax").text); ymax = float(bb.find("ymax").text)
            # 範囲クリップ＆最低サイズチェック
            xmin = max(0.0, min(xmin, W-1)); xmax = max(0.0, min(xmax, W-1))
            ymin = max(0.0, min(ymin, H-1)); ymax = max(0.0, min(ymax, H-1))
            if xmax <= xmin or ymax <= ymin: 
                continue
            xc,yc,bw,bh = voc_to_yolo((W,H),(xmin,ymin,xmax,ymax))
            if bw <= 1e-6 or bh <= 1e-6: 
                continue
            cid = name_to_id[cls_name]
            lines.append(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        # 出力
        copy2(img_path, out_img / f"{id_}{img_path.suffix.lower()}")
        (out_lbl / f"{id_}.txt").write_text("\n".join(lines))

        n_img += 1
        if lines:
            n_with_gt += 1
            n_box += len(lines)

    # meta.json（雛形）
    meta = {
        "run_name": args.run_name,
        "dataset_source": "SCTD(VOC)",
        "classes": cls_list,
        "notes": "Converted from VOC. Control/motion metadata not available.",
        "sonar": {"frequency_khz": None, "water_temp_c": None, "platform": None},
        "motion": {"speed_mps": None, "accel_mps2": None, "attitude_rpy_deg": None}
    }
    (out_run/"meta.json").write_text(
        __import__("json").dumps(meta, indent=2, ensure_ascii=False)
    )

    print(f"[DONE] images={n_img} with_gt={n_with_gt} boxes={n_box} skip_xml={n_skip_xml} skip_cls={n_skip_cls}")
    print(f"Run dir : {out_run}")
    print(f"  sonar : {out_img}")
    print(f"  labels: {out_lbl}")
    print(f"  meta  : {out_run/'meta.json'}")

if __name__ == "__main__":
    main()