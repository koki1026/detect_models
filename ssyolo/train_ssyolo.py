# ssyolo/train_ssyolo.py
from pathlib import Path
import sys

from ultralytics import YOLO
import ultralytics.nn.modules as um
import ultralytics.nn.tasks as ut

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # ~/detect_mdoels を指す

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from custom_layers.fastc2f import FastC2f


# ★ Ultralytics の名前空間に FastC2f を登録
um.FastC2f = FastC2f
ut.FastC2f = FastC2f


def main():
    model_cfg = ROOT / "ssyolo" / "ssyolo_sctd.yaml"
    data_cfg = ROOT / "data_sctd.yaml"

    model = YOLO(str(model_cfg))
    model.train(
        data=str(data_cfg),
        epochs=1,
        batch=8,
        imgsz=640,
        optimizer="SGD",
        lr0=0.01,
        lrf=0.1,
        momentum=0.937,
        weight_decay=0.0005,
        project="runs_ssyolo",
        name="exp1_fastc2f_smoke",
    )


if __name__ == "__main__":
    main()