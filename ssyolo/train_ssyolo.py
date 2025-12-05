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
from custom_layers.assf import ASSF
from custom_layers.detect_assf import DetectASSF

# ★ Ultralytics の名前空間に FastC2f を登録
um.FastC2f = FastC2f
ut.FastC2f = FastC2f

# ★ Ultralytics の名前空間に ASSF を登録
um.ASSF = ASSF
ut.ASSF = ASSF

um.Detect = DetectASSF
ut.Detect = DetectASSF


def main():
    model_cfg = ROOT / "ssyolo" / "ssyolo_sctd.yaml"
    data_cfg = ROOT / "data_sctd.yaml"

    model = YOLO(str(model_cfg))

    model.train(
        data=str(data_cfg),
        epochs=200,          # ★ 本気学習
        imgsz=640,           # 最初は 640 のままでOK（960 は後で）
<<<<<<< Updated upstream
        batch=8,             # 8GB なので 8 が安全圏。16 はたぶんアウト
=======
        batch=2,             # 8GB なので 8 が安全圏。16 はたぶんアウト
>>>>>>> Stashed changes
        optimizer="SGD",     # 論文準拠
        lr0=0.01,            # 初期学習率 (Table 2)
        lrf=0.1,             # 最終 lr = lr0 * lrf
        momentum=0.937,
        weight_decay=0.0005,
        #cosine=True,         # コサインスケジュール（好みだけど有り）
        patience=30,         # 早期終了。mAPが全然伸びなくなったら止めてくれる
        project="runs_ssyolo",
        name="exp4_fastc2f_fix_200ep",
        # amp=False, 
        plots=True,        # （今のまま SciPy エラー出ても気にしないならそのまま）
        # device=0,          # 明示してもOK
        workers=8,         # デフォルトのままでもOK
    )

if __name__ == "__main__":
    main()