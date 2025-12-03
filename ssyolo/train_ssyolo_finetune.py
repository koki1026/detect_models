# ssyolo/train_ssyolo_finetune.py
from pathlib import Path
import sys

from ultralytics import YOLO

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # ~/detect_mdoels

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# FastC2f の登録（元の train_ssyolo.py と同じ）
import ultralytics.nn.modules as um
import ultralytics.nn.tasks as ut
from custom_layers.fastc2f import FastC2f
um.FastC2f = FastC2f
ut.FastC2f = FastC2f


def main():
    model_cfg = ROOT / "runs_ssyolo" / "exp2_fastc2f_200ep3" / "weights" / "best.pt"
    data_cfg = ROOT / "data_sctd.yaml"

    # best.pt からロードして微調整
    model = YOLO(str(model_cfg))

    model.train(
        data=str(data_cfg),
        epochs=50,          # 追加で 50epoch だけ
        imgsz=640,          # 解像度はそのまま（ASSF 実験と比較しやすくする）
        batch=8,
        optimizer="SGD",
        lr0=0.003,          # ★ 低めの学習率で fine-tune
        lrf=0.1,
        momentum=0.937,
        weight_decay=0.0005,
        patience=20,        # 20epoch 改善なければ早期終了
        project="runs_ssyolo",
        name="exp2b_fastc2f_finetune",
        plots=True,
    )

if __name__ == "__main__":
    main()