# detect_mdoels/custom_layers/fastc2f.py
import torch
import torch.nn as nn

from ultralytics.nn.modules import Conv  # pip版のConvを使う


class PConv(nn.Module):
    """
    FasterNet 由来の部分畳み込み (Partial Convolution).
    入力チャンネルの一部だけに 3x3 Conv をかけ、残りはスキップして結合する。
    """

    def __init__(self, c: int, k: int = 3, s: int = 1, ratio: float = 0.25, act: bool = True):
        super().__init__()
        self.c = c
        self.ratio = ratio
        c_p = max(1, int(c * ratio))  # 畳み込むチャンネル数
        self.c_p = c_p

        p = k // 2
        self.conv = nn.Conv2d(c_p, c_p, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(c_p)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        c_p = self.c_p
        x_p, x_skip = x[:, :c_p], x[:, c_p:]
        y = self.act(self.bn(self.conv(x_p)))
        return torch.cat((y, x_skip), 1)


class PWConv(nn.Module):
    """
    Point-wise (1x1) Convolution の薄いラッパー。
    """
    def __init__(self, c1: int, c2: int, act: bool = True):
        super().__init__()
        self.conv = Conv(c1, c2, k=1, s=1, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class FasterBlock(nn.Module):
    """
    SS-YOLO 論文の Fast-C2f 内で使われる基本ブロック。
    PConv → PWConv(拡張) → PWConv(圧縮) + 入力の残差。
    """
    def __init__(self, c: int, expansion: float = 2.0, ratio: float = 0.25):
        super().__init__()
        c_mid = int(c * expansion)
        self.pconv = PConv(c, k=3, s=1, ratio=ratio, act=True)
        self.pw1 = PWConv(c, c_mid, act=True)
        self.pw2 = PWConv(c_mid, c, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pconv(x)
        y = self.pw1(y)
        y = self.pw2(y)
        return y + x  # 残差接続


class FastC2f(nn.Module):
    """
    SS-YOLO の Fast-C2f モジュール。
    Ultralytics の C2f と同じインターフェースで、内部の Bottleneck を FasterBlock に差し替えた版。
    """
    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = False,  # 互換性のため残しているだけ
        g: int = 1,
        e: float = 0.5,
        ratio: float = 0.25,
    ):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1, 1)
        self.m = nn.ModuleList(FasterBlock(self.c, ratio=ratio) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        for m in self.m:
            y.append(m(y[-1]))
        return self.cv2(torch.cat(y, 1))