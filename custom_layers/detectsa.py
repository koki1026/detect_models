# custom_layers/detectsa.py

import torch
import torch.nn as nn

import ultralytics.nn.modules as um
from ultralytics.nn.modules import Conv  # YOLOv8 と同じ Conv を使う


class MHSA2D(nn.Module):
    """
    2D feature map 用の Multi-Head Self-Attention.
    入力: (B, C, H, W)
    出力: 同じ shape
    """
    def __init__(self, c: int, num_heads: int = 4):
        super().__init__()
        self.c = c
        self.num_heads = num_heads
        # batch_first=True で (B, N, C)
        self.attn = nn.MultiheadAttention(embed_dim=c,
                                          num_heads=num_heads,
                                          batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        # (B, C, H, W) -> (B, N, C)
        x_flat = x.view(b, c, h * w).permute(0, 2, 1)
        y, _ = self.attn(x_flat, x_flat, x_flat)  # self-attention
        # (B, N, C) -> (B, C, H, W)
        y = y.permute(0, 2, 1).view(b, c, h, w)
        return y


class DetectSA(um.Detect):
    """
    論文 Figure 6 の「Memory-efficient self-attention」を
    YOLOv8 Detect の前段に入れたヘッド。

    - 1x1 Conv で C1 -> C2 に圧縮
    - MHSA2D で自己注意
    - 1x1 Conv で C2 -> C1 に戻す
    - 1x1 Conv で bypass を作って residual add
    - その出力を親クラス Detect に渡す
    """

    def __init__(
        self,
        nc: int = 80,
        ch=(),
        inplace: bool = True,
        num_heads: int = 4,
        attn_ratio: float = 0.25,
    ):
        c3, c4, c5 = ch
        # まず通常の Detect を初期化
        super().__init__(nc=nc, ch=ch)

        self.nl = len(ch)
        self.ch = list(ch)

        # 各スケールごとのチャンネル圧縮 C1 -> C2
        c2_list = [max(8, int(c * attn_ratio)) for c in self.ch]

        self.reduce = nn.ModuleList(
            [Conv(c1, c2, 1, 1) for c1, c2 in zip(self.ch, c2_list)]
        )
        self.mhsa = nn.ModuleList(
            [MHSA2D(c2, num_heads=num_heads) for c2 in c2_list]
        )
        self.expand = nn.ModuleList(
            [Conv(c2, c1, 1, 1) for c2, c1 in zip(c2_list, self.ch)]
        )
        # bypass 用 1x1 conv
        self.short = nn.ModuleList(
            [Conv(c1, c1, 1, 1) for c1 in self.ch]
        )

    def forward(self, x):
        """
        x: list[Tensor] = [P3, P4, P5], 各 (B, C, H, W)
        """
        assert len(x) == self.nl, f"Expected {self.nl} feature maps, got {len(x)}"

        attn_feats = []
        for i, xi in enumerate(x):
            # 1x1 conv で圧縮
            y = self.reduce[i](xi)
            # MHSA
            y = self.mhsa[i](y)
            # 1x1 conv で元のチャネル数へ
            y = self.expand[i](y)
            # bypass + residual
            skip = self.short[i](xi)
            y = y + skip
            attn_feats.append(y)

        # あとは YOLOv8 の Detect に丸投げ
        return super().forward(attn_feats)