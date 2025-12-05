## install
### 仮想環境を構築
```
./setup_env.sh
```
### venvを有効化
```
source .venv/bin/activate
```
### 依存ライブラリをインストール
```
pip install -r requirements.txt
```

## 実行
```
source .venv/bin/activate  # まだなら
python3 ssyolo/train_ssyolo.py
```
### ハイパーパラメータ
```
data=str(data_cfg), 
epochs=200,          # ★ 本気学習
imgsz=640,           # 最初は 640 のままでOK（960 は後で）
batch=8,             # 8GB なので 8 が安全圏。16 はたぶんアウト
optimizer="SGD",     # 論文準拠
lr0=0.01,            # 初期学習率 (Table 2)
lrf=0.1,             # 最終 lr = lr0 * lrf
momentum=0.937,
weight_decay=0.0005,
#cosine=True,         # コサインスケジュール（好みだけど有り）
patience=30,         # 早期終了。mAPが全然伸びなくなったら止めてくれる
project="runs_ssyolo",
name="exp3_fastc2f_assf_50ep",
plots=True,        # （今のまま SciPy エラー出ても気にしないならそのまま）
#device=0,          # 明示してもOK
workers=8,         # デフォルトのままでもOK
```
