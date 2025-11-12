#!/usr/bin/env python3
from pathlib import Path
import subprocess
import argparse
import sys
import re
import time
import os
import json
import pandas as pd

def run_cmd(cmd, cwd=None):
    print("[cmd]", " ".join(cmd), flush=True)
    p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out_lines = []
    while True:
        line = p.stdout.readline()
        if not line and p.poll() is not None:
            break
        if line:
            out_lines.append(line)
            print(line, end="")  # live echo
    code = p.wait()
    return code, "".join(out_lines)

def find_artifacts_path(stdout_text: str, runs_dir: Path) -> Path | None:
    m = re.search(r"Artifacts\s*->\s*(runs/exp_\d{8}_\d{6})", stdout_text)
    if m:
        # normalize to absolute path under runs_dir's parent
        exp_rel = Path(m.group(1))
        cand = runs_dir.parent / exp_rel
        return cand
    if runs_dir.exists():
        runs = sorted(runs_dir.glob("exp_*"), key=lambda p: p.stat().st_mtime, reverse=True)
        return runs[0] if runs else None
    return None

def ensure_labels_symlink(dataset_root: Path, run_name: str):
    runs_dir = dataset_root / "runs" / run_name
    lbl_target = dataset_root / "labels"
    lbl_link = runs_dir / "labels"
    runs_dir.mkdir(parents=True, exist_ok=True)
    if not lbl_link.exists():
        try:
            if lbl_link.is_symlink() or lbl_link.exists():
                lbl_link.unlink()
        except Exception:
            pass
        try:
            os.symlink(os.path.relpath(lbl_target, runs_dir), lbl_link)
            print(f"[prep] symlinked {lbl_link} -> {lbl_target}")
        except Exception as e:
            print(f"[prep-warn] symlink failed ({e}); trying copy...")
            if lbl_target.exists():
                import shutil
                shutil.copytree(lbl_target, lbl_link)
                print(f"[prep] copied labels to {lbl_link}")

def parse_metrics_any(repo_root: Path, exp_dir: Path):
    """Return (df,last,best_map,macro_last) from either runs_logs CSV or runs JSON."""
    runs_logs = repo_root / "runs_logs"
    exp_name = exp_dir.name  # e.g., exp_20251110_223426

    # 1) Preferred: runs_logs/exp_.../exp_....csv
    csv1 = runs_logs / exp_name / f"{exp_name}.csv"
    csv2 = runs_logs / f"{exp_name}.csv"
    if csv1.exists():
        df = pd.read_csv(csv1)
    elif csv2.exists():
        df = pd.read_csv(csv2)
    else:
        # 2) Fallback: runs/exp_.../metrics.json
        json_path = exp_dir / "metrics.json"
        if not json_path.exists():
            raise FileNotFoundError(f"No metrics found for {exp_name}")
        with open(json_path, "r") as f:
            data = json.load(f)
        # expect a list of dicts or a dict of lists; normalize
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = pd.DataFrame(data)
        # also save a convenience CSV under runs_logs
        (runs_logs / exp_name).mkdir(parents=True, exist_ok=True)
        out_csv = runs_logs / exp_name / f"{exp_name}.csv"
        df.to_csv(out_csv, index=False)
        print(f"[info] wrote metrics CSV -> {out_csv}")

    df.columns = [c.strip() for c in df.columns]
    # numeric coercion
    for c in ["epoch","map50","ap_aircraft","ap_ship","ap_human","val_preds_kept"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    last = df.iloc[-1].to_dict()
    best_map = df["map50"].max() if "map50" in df.columns else None
    macro_last = None
    if all(col in df.columns for col in ["ap_aircraft","ap_ship","ap_human"]):
        macro_last = float((last.get("ap_aircraft",0)+last.get("ap_ship",0)+last.get("ap_human",0))/3.0)
    return df, last, best_map, macro_last

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=str, default=".", help="path to detect_models root")
    ap.add_argument("--dataset-root", type=str, default="./dataset/SCTD_yolo")
    ap.add_argument("--labels-run-name", type=str, default="SCTD_train", help="run_name passed to training (used by auto-anchors to find dataset_root/runs/<run_name>/labels)")
    ap.add_argument("--train-script", type=str, default="./dbnet/train_dbnet_sctd.py")
    ap.add_argument("--plot-script", type=str, default="./tools/csv_to_plot.py")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--img-size", type=int, default=640)
    ap.add_argument("--num-classes", type=int, default=3)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--topk-per-level", type=int, default=1000)
    ap.add_argument("--max-det", type=int, default=100)
    ap.add_argument("--ks", type=str, default="3,4,5", help="comma-separated k values")
    ap.add_argument("--confs", type=str, default="0.25,0.20,0.15", help="comma-separated conf thresholds")
    ap.add_argument("--extra-flags", type=str, default="--focal-obj --focal-cls --focal-gamma 2.0 --focal-alpha 0.25 --ema")
    ap.add_argument("--dry-run", action="store_true", help="don't actually train; only print commands")
    ap.add_argument("--summary-out", type=str, default="./runs_logs/grid_summary.csv")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    dataset_root = (repo_root / args.dataset_root).resolve()
    train_script = (repo_root / args.train_script).resolve()
    plot_script = (repo_root / args.plot_script).resolve()
    runs_dir = repo_root / "runs"

    ks = [int(x.strip()) for x in args.ks.split(",") if x.strip()]
    confs = [float(x.strip()) for x in args.confs.split(",") if x.strip()]

    summary_rows = []
    (repo_root / Path(args.summary_out)).parent.mkdir(parents=True, exist_ok=True)

    print("[info] repo_root :", repo_root)
    print("[info] dataset   :", dataset_root)
    print("[info] train     :", train_script)
    print("[info] runs dir  :", runs_dir)
    print("[info] runs_logs :", repo_root / "runs_logs")
    print()

    for k in ks:
        for conf in confs:
            run_name = args.labels_run_name
            ensure_labels_symlink(dataset_root, run_name)

            train_cmd = [
                sys.executable, str(train_script),
                "--dataset-root", str(dataset_root),
                "--run-name", run_name,
                "--epochs", str(args.epochs),
                "--batch-size", str(args.batch_size),
                "--img-size", str(args.img_size),
                "--num-classes", str(args.num_classes),
                "--workers", str(args.workers),
                "--conf-thres", str(conf),
                "--topk-per-level", str(args.topk_per_level),
                "--max-det", str(args.max_det),
                "--auto-anchors", "--anchors-k", str(k),
            ] + args.extra_flags.split()

            print("\n================ GRID JOB =================")
            print(f"k={k}  conf={conf}  run-name={run_name}")
            print("==========================================\n")

            if args.dry_run:
                print("[dry-run] would run:", " ".join(train_cmd))
                artifacts = None
                last = {}
                best_map = None
                macro_last = None
            else:
                code, out = run_cmd(train_cmd, cwd=str(repo_root))
                if code != 0:
                    print(f"[error] training failed for k={k}, conf={conf} (exit={code})")
                    summary_rows.append(dict(
                        k=k, conf=conf, run_name=run_name,
                        status="failed", exp_path="",
                        map50_last=None, map50_best=None, map50_macro_last=None,
                        ap_aircraft=None, ap_ship=None, ap_human=None,
                        val_preds_kept=None
                    ))
                    continue

                artifacts = find_artifacts_path(out, runs_dir)
                if not artifacts:
                    print("[warn] could not detect exp dir; using latest under runs/")
                    artifacts = find_artifacts_path("", runs_dir)

                if not artifacts or not artifacts.exists():
                    print("[warn] exp dir does not exist; skipping metrics parse")
                    summary_rows.append(dict(
                        k=k, conf=conf, run_name=run_name,
                        status="ok_no_exp", exp_path=str(artifacts) if artifacts else "",
                        map50_last=None, map50_best=None, map50_macro_last=None,
                        ap_aircraft=None, ap_ship=None, ap_human=None,
                        val_preds_kept=None
                    ))
                    continue

                try:
                    df, last, best_map, macro_last = parse_metrics_any(repo_root, artifacts)
                except Exception as e:
                    print("[warn] failed to parse metrics:", e)
                    last = {}
                    best_map = None
                    macro_last = None

                if plot_script.exists():
                    print("[info] plotting via", plot_script)
                    # Most likely this script is called like: python tools/csv_to_plot.py --csv <csv>
                    # Use the CSV we just found/created:
                    target_csv = (repo_root / "runs_logs" / artifacts.name / f"{artifacts.name}.csv")
                    if not target_csv.exists():
                        # try flat location
                        alt_csv = (repo_root / "runs_logs" / f"{artifacts.name}.csv")
                        target_csv = alt_csv if alt_csv.exists() else target_csv
                    if target_csv.exists():
                        _ = run_cmd([sys.executable, str(plot_script), "--csv", str(target_csv)], cwd=str(repo_root))[0]
                    else:
                        print("[info] plotting skipped; csv not found at expected path")
                else:
                    print("[info] plotting script not found, skipped:", plot_script)

            summary_rows.append(dict(
                k=k, conf=conf, run_name=run_name,
                status="ok" if not args.dry_run else "dry",
                exp_path=str(artifacts) if artifacts else "",
                map50_last=last.get("map50") if last else None,
                map50_best=best_map,
                map50_macro_last=macro_last,
                ap_aircraft=last.get("ap_aircraft") if last else None,
                ap_ship=last.get("ap_ship") if last else None,
                ap_human=last.get("ap_human") if last else None,
                val_preds_kept=last.get("val_preds_kept") if last else None,
            ))

            time.sleep(1)

    df_sum = pd.DataFrame(summary_rows)
    out_csv = repo_root / Path(args.summary_out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_sum.to_csv(out_csv, index=False)
    print("\n=== Grid Summary ===")
    print(df_sum)
    print("\n[ok] Wrote summary CSV ->", out_csv)

    # Helpful echo for user's tree:
    print("\nExample command for your tree:")
    print(f"python3 tools/run_grid_search.py --repo-root . "
          f"--dataset-root ./dataset/SCTD_yolo "
          f"--train-script ./dbnet/train_dbnet_sctd.py "
          f"--plot-script ./tools/csv_to_plot.py "
          f"--epochs 10 --batch-size 32 --img-size 640 "
          f"--num-classes 3 --workers 16 "
          f"--ks 3,4,5 --confs 0.25,0.20,0.15 "
          f"--topk-per-level 1000 --max-det 100 "
          f"--extra-flags \"--focal-obj --focal-cls --focal-gamma 2.0 --focal-alpha 0.25 --ema\"")

if __name__ == "__main__":
    main()