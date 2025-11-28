import os
import numpy as np
from collections import Counter

labels_dir = "./dataset/SCTD_yolo/runs/SCTD_train/labels"  # ★適宜書き換え
class_names = ["aircraft", "ship", "human"]  # ★現在のクラス定義

counts = Counter()

for file in os.listdir(labels_dir):
    if file.endswith(".txt"):
        labels = np.loadtxt(os.path.join(labels_dir, file), ndmin=2)
        if labels.size > 0:
            cls_ids = labels[:, 0].astype(int)
            counts.update(cls_ids)

total = sum(counts.values())

print("\n===== Class Distribution =====")
for cid, name in enumerate(class_names):
    print(f"{name:8s}: {counts[cid]:4d} ({counts[cid]/total*100:.2f}%)")
print(f"Total: {total}")