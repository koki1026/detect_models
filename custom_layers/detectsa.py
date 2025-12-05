# custom_layers/detectsa.py

import torch
import torch.nn as nn

import ultralytics.nn.modules as um
from ultralytics.nn.modules import Conv  # YOLOv8 と同じ Conv を使う

from custom_layers.assf import ASSF


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
        # ch は backbone/neck からの (C3, C4, C5) = (256, 512, 1024) 想定
        assert len(ch) == 3, f"DetectSA expects 3 feature maps, got {ch}"
        c3, c4, c5 = ch

        # ASSF で揃える出力チャネル数（ここでは P3 と同じ 256）
        c_out = c3

        # ★ Detect 側には「ASSF 後」のチャネル (256, 256, 256) を渡す
        ch_assf = (c_out, c_out, c_out)
        super().__init__(nc=nc, ch=ch_assf)

        # ★ ASSF 本体（入力は元の 256, 512, 1024）
        self.assf = ASSF(c3=c3, c4=c4, c5=c5, c_out=c_out)

        self.nl = len(ch_assf)
        self.ch = list(ch_assf)   # = [256, 256, 256]

        # ---- 以下は既存の SA 部分を「ASSF 後の 3スケール」に対して適用 ----
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
        self.short = nn.ModuleList(
            [Conv(c1, c1, 1, 1) for c1 in self.ch]
        )

    def forward(self, x):
        """
        x: [P3, P4, P5] with channels (256, 512, 1024) from the neck
        """
        assert isinstance(x, (list, tuple)) and len(x) == self.nl, \
            f"Expected {self.nl} feature maps, got {len(x)}"

        # ★ まず ASSF で multi-scale fusion & channel unify (→ 3 枚とも 256ch)
        p3, p4, p5 = self.assf(x)
        x_assf = [p3, p4, p5]

        # ★ そのあと Self-Attention ブロック
        attn_feats = []
        for i, xi in enumerate(x_assf):
            y = self.reduce[i](xi)
            y = self.mhsa[i](y)
            y = self.expand[i](y)
            skip = self.short[i](xi)
            y = y + skip
            attn_feats.append(y)

        # ★ 最後に YOLO Detect へ
        return super().forward(attn_feats)