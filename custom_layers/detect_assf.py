# custom_layers/detect_assf.py

import torch
import torch.nn as nn

from ultralytics.nn.modules.head import Detect as DetectBase
from custom_layers.assf import ASSF


class DetectASSF(DetectBase):
    """
    Detect ヘッドの中に ASSF を統合したバージョン。

    入力:
        x: [P3, P4, P5] の list を想定
            P3: [B, C3, H3, W3]
            P4: [B, C4, H4, W4]
            P5: [B, C5, H5, W5]

    やること:
        1. ASSF で multi-scale feature fusion
        2. 出力3枚 (P3', P4', P5') を DetectBase に渡して通常の YOLO head を適用
    """

    def __init__(self, nc=80, ch=(), **kwargs):
        """
        nc: クラス数
        ch: backbone/neck から入ってくるチャネル数タプル (C3, C4, C5) 例: (256, 512, 1024)
        """
        assert len(ch) == 3, f"DetectASSF expects 3 input channels, got {ch}"
        c3, c4, c5 = ch

        # ASSF で最終的に揃えるチャネル数
        c_out = c3  # ここでは P3 のチャネル数 256 に統一

        # DetectBase 側には「ASSF後」のチャネル情報を渡す
        super().__init__(nc=nc, ch=(c_out, c_out, c_out), **kwargs)

        # ASSF 本体（元の P3/P4/P5 チャネルを使う）
        self.assf = ASSF(c3=c3, c4=c4, c5=c5, c_out=c_out)

    def forward(self, x):
        """
        x: list of feature maps [P3, P4, P5]
        """
        # ASSF で融合した特徴を作る
        p3, p4, p5 = self.assf(x)

        # DetectBase の forward に渡す
        return super().forward([p3, p4, p5])