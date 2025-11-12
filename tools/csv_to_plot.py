#!/usr/bin/env python3
"""
Plot training/validation metrics from a CSV exported by train_dbnet_sctd.py.

- Creates runs_logs/<stamp>/ and moves the CSV into that folder.
- Saves all plots inside runs_logs/<stamp>/ using the same stamp.
- Stamp is inferred from filename like 'exp_20251110_175814.csv' (or falls back to stem).

Usage:
  python plot_metrics.py --csv ./runs/exp_20251110_175814/metrics.csv
  # (optional) override logs root:
  python plot_metrics.py --csv ./runs_logs/exp_20251110_175814.csv --logs-root ./runs_logs
"""

import argparse
from pathlib import Path
import re
import sys
import shutil

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless save
import matplotlib.pyplot as plt


def detect_stamp(csv_path: Path) -> str:
    m = re.search(r"(exp_\d{8}_\d{6})", csv_path.stem)
    return m.group(1) if m else csv_path.stem


def read_csv_auto(csv_path: Path) -> pd.DataFrame:
    # Try auto sep
    try:
        df = pd.read_csv(csv_path, sep=None, engine="python")
        if df.shape[1] == 1:
            raise ValueError("single-column; retry tab")
        return df
    except Exception:
        # Try tab
        try:
            df = pd.read_csv(csv_path, sep="\t")
            if df.shape[1] == 1:
                raise ValueError("tab failed; retry whitespace")
            return df
        except Exception:
            # Fallback: any whitespace
            return pd.read_csv(csv_path, delim_whitespace=True)


def ensure_numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def plot_line(x, ys, labels, title, xlabel, ylabel, out_path: Path):
    plt.figure()
    for y, lab in zip(ys, labels):
        plt.plot(x, y, label=lab, marker="o")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if len(labels) > 1:
        plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="path to metrics.csv")
    ap.add_argument("--logs-root", type=str, default="runs_logs", help="root folder to store plots & moved csv")
    args = ap.parse_args()

    src_csv = Path(args.csv).expanduser().resolve()
    if not src_csv.exists():
        print(f"[error] CSV not found: {src_csv}", file=sys.stderr)
        sys.exit(1)

    stamp = detect_stamp(src_csv)
    logs_root = Path(args.logs_root).expanduser().resolve()
    outdir = logs_root / stamp
    outdir.mkdir(parents=True, exist_ok=True)

    # Move CSV into runs_logs/<stamp> (skip if already there)
    dst_csv = outdir / src_csv.name
    try:
        if src_csv.resolve() != dst_csv.resolve():
            # If a file already exists at destination, overwrite it
            if dst_csv.exists():
                dst_csv.unlink()
            shutil.move(str(src_csv), str(dst_csv))
            print(f"[ok] Moved CSV -> {dst_csv}")
        else:
            print(f"[info] CSV already in target folder: {dst_csv}")
    except Exception as e:
        print(f"[warn] failed to move CSV: {e}\n[info] will read from original path.")
        dst_csv = src_csv  # fall back

    # Read CSV
    df = read_csv_auto(dst_csv)
    df.columns = [c.strip() for c in df.columns]
    df = ensure_numeric(df, [
        "epoch","map50","l_box","l_obj","l_cls",
        "val_preds_kept","ap_aircraft","ap_ship","ap_human"
    ])
    if "epoch" in df.columns:
        df = df.sort_values("epoch")

    # 1) mAP@0.5
    if {"epoch","map50"}.issubset(df.columns):
        plot_line(
            df["epoch"], [df["map50"]],
            ["mAP@0.5"], "Validation mAP@0.5 over Epochs",
            "Epoch", "mAP@0.5",
            outdir / f"{stamp}_map.png"
        )

    # 2) Loss curves
    if {"epoch","l_box","l_obj","l_cls"}.issubset(df.columns):
        plot_line(
            df["epoch"], [df["l_box"], df["l_obj"], df["l_cls"]],
            ["l_box","l_obj","l_cls"], "Losses over Epochs",
            "Epoch","Loss",
            outdir / f"{stamp}_losses.png"
        )

    # 3) Per-class AP
    ap_cols = [("ap_aircraft","aircraft"), ("ap_ship","ship"), ("ap_human","human")]
    ys, labels = [], []
    for c, lab in ap_cols:
        if c in df.columns:
            ys.append(df[c]); labels.append(lab)
    if "epoch" in df.columns and ys:
        plot_line(
            df["epoch"], ys, labels,
            "Per-class AP@0.5 over Epochs",
            "Epoch", "AP@0.5",
            outdir / f"{stamp}_ap.png"
        )

    # 4) Predictions kept
    if {"epoch","val_preds_kept"}.issubset(df.columns):
        plot_line(
            df["epoch"], [df["val_preds_kept"]],
            ["val_preds_kept"], "Predictions Kept per Epoch",
            "Epoch", "Count",
            outdir / f"{stamp}_preds_kept.png"
        )

    print(f"[ok] Saved outputs in: {outdir}")
    for name in ["map","losses","ap","preds_kept"]:
        p = outdir / f"{stamp}_{name}.png"
        if p.exists():
            print(" -", p)


if __name__ == "__main__":
    main()
