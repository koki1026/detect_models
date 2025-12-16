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
from custom_layers.detectsa import DetectSA

# ★ Ultralytics の名前空間に FastC2f を登録
um.FastC2f = FastC2f
ut.FastC2f = FastC2f

# ★ Ultralytics の名前空間に ASSF を登録
um.ASSF = ASSF
ut.ASSF = ASSF

um.Detect = DetectSA
ut.Detect = DetectSA

um.DetectSA = DetectSA
ut.DetectSA = DetectSA


def main():
    model_cfg = ROOT / "ssyolo" / "ssyolo_sctd.yaml"
    data_cfg = ROOT / "data_sctd.yaml"

    model = YOLO(str(model_cfg), task="detect")
    
    model.model.info(imgsz=640, verbose=True)

    model.train(
        data=str(data_cfg),
        epochs=1,          # ★ 本気学習
        imgsz=960,           # 最初は 640 のままでOK（960 は後で）
        batch=8,             # 8GB なので 8 が安全圏。16 はたぶんアウト
        optimizer="SGD",     # 論文準拠
        lr0=0.01,            # 初期学習率 (Table 2)
        lrf=0.1,             # 最終 lr = lr0 * lrf
        momentum=0.937,
        weight_decay=0.0005,
        #cosine=True,         # コサインスケジュール（好みだけど有り）
        patience=300,         # 早期終了。mAPが全然伸びなくなったら止めてくれる
        project="runs_ssyolo",
        name="exp5_detectSA_ASSF_v203_300ep_960_8",
        amp=False, 
        plots=True,        # （今のまま SciPy エラー出ても気にしないならそのまま）
        workers=8,         # デフォルトのままでもOK
        # nbs=8, 
    )

if __name__ == "__main__":
    main()