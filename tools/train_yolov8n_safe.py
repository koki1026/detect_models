from torch.serialization import add_safe_globals
from ultralytics.nn.tasks import DetectionModel

# PyTorch に「このクラスは安全だからOKだよ」と教える
add_safe_globals([DetectionModel])

from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")  # ここでのtorch.loadがsafe_globalsの設定を使う
    model.train(
        data="data_sctd.yaml",
        epochs=200,
        imgsz=640,
        batch=8,
        device=0,
    )

if __name__ == "__main__":
    main()