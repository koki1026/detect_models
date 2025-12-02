# detect_mdoels/custom_layers/fastc2f.py

import torch
import torch.nn as nn


class PConv(nn.Module):
    """
    Partial Convolution:
      入力チャネルの一部だけ 3x3 Conv をかけ、残りはそのままスルーして結合する。
    """

    def __init__(self, c: int, k: int = 3, s: int = 1, ratio: float = 0.25, act: bool = True):
        super().__init__()
        c = int(c)

        # 部分畳み込みに使うチャネル数
        c_partial = max(int(round(c * ratio)), 1)
        c_skip = c - c_partial

        self.c = c
        self.c_partial = c_partial
        self.c_skip = max(c_skip, 0)

        padding = k // 2
        self.conv = nn.Conv2d(c_partial, c_partial, k, s, padding, bias=False)
        self.bn = nn.BatchNorm2d(c_partial)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        if self.c_skip > 0:
            x_p, x_s = torch.split(x, [self.c_partial, self.c_skip], dim=1)
        else:
            x_p, x_s = x, None

        y_p = self.act(self.bn(self.conv(x_p)))

        if x_s is not None:
            return torch.cat([y_p, x_s], dim=1)
        else:
            return y_p


class PWConv(nn.Module):
    """
    Pointwise Conv (1x1 Conv) + BN + 活性化
    """

    def __init__(self, c1: int, c2: int, act: bool = True):
        super().__init__()
        c1, c2 = int(c1), int(c2)
        self.conv = nn.Conv2d(c1, c2, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class FasterBlock(nn.Module):
    """
    SS-YOLO の Fast-C2f 内部ブロック:
      PConv → PWConv(拡張) → PWConv(圧縮) + 残差
    """

    def __init__(self, c: int, expansion: float = 2.0, ratio: float = 0.25):
        super().__init__()
        c = int(c)
        c_mid = max(int(round(c * expansion)), 1)

        self.pconv = PConv(c, k=3, s=1, ratio=ratio, act=True)
        self.pw1 = PWConv(c, c_mid, act=True)
        self.pw2 = PWConv(c_mid, c, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pconv(x)
        y = self.pw1(y)
        y = self.pw2(y)
        return x + y


class FastC2f(nn.Module):
    """
    YOLOv8 の C2f 互換インタフェースを持つ Fast-C2f 実装。

    Ultralytics からは YAML により

      - [-1, 3, FastC2f, [128, True]]

    という形で呼ばれるので、コンストラクタは

      FastC2f(c2, shortcut=False, g=1, e=0.5, ratio=0.25)

    という形に合わせる。入力チャネル c1 は forward 時に x.shape[1] から取得し、
    最初の forward で畳み込み層を lazy に構築する。
    """

    def __init__(
        self,
        c2: int,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
        ratio: float = 0.25,
    ):
        super().__init__()

        # YAML から list/tuple で来たときの保険
        if isinstance(c2, (list, tuple)):
            c2 = c2[0]

        self.c2 = int(c2)
        self.e = float(e)
        self.ratio = ratio
        self.shortcut = shortcut

        self._built = False  # lazy build フラグ

    def _build(self, c1: int):
        """最初の forward で呼ばれて，実際の入力チャネル数 c1 に合わせて層を構築する。"""
        c1 = int(c1)
        hidden = max(int(self.c2 * self.e), 1)
        self.hidden = hidden

        self.cv1 = nn.Sequential(
            nn.Conv2d(c1, 2 * hidden, 1, 1, 0, bias=False),
            nn.BatchNorm2d(2 * hidden),
            nn.SiLU(),
        )

        # 内部ブロックはとりあえず 1 個（外側の n=3 でスタックされる）
        self.blocks = nn.ModuleList(
            [FasterBlock(hidden, expansion=2.0, ratio=self.ratio)]
        )

        self.cv2 = nn.Sequential(
            nn.Conv2d((2 + len(self.blocks)) * hidden, self.c2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.c2),
            nn.SiLU(),
        )

        self._built = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 最初の forward でだけ層を構築する
        if not self._built:
            self._build(x.shape[1])

        # C2f と同じ forward 形式
        y0, y1 = self.cv1(x).chunk(2, 1)
        ys = [y0, y1]
        for m in self.blocks:
            ys.append(m(ys[-1]))
        out = torch.cat(ys, dim=1)
        return self.cv2(out)