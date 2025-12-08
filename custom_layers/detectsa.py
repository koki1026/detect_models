import torch
import torch.nn as nn
import torch.nn.functional as F

import ultralytics.nn.modules as um
from ultralytics.nn.modules import Conv  # YOLOv8 と同じ Conv を使う

from custom_layers.assf import ASSF


class GlobalMHSA2D(nn.Module):
    """
    Global Multi-Head Self-Attention for 2D feature maps.

    - 入力: (B, C, H, W)
    - 出力: (B, C, H, W) 同じ shape
    - HxW 全体を 1 シーケンス (N = H*W) として扱う
    """

    def __init__(self, c: int, num_heads: int = 4):
        super().__init__()
        self.c = c
        self.num_heads = num_heads

        # batch_first=True で (B, N, C)
        self.attn = nn.MultiheadAttention(
            embed_dim=c,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W]
        """
        b, c, h, w = x.shape
        # (B, C, H, W) -> (B, N, C)
        x_seq = x.view(b, c, h * w).permute(0, 2, 1).contiguous()  # (B, N, C)

        # Global MHSA
        y, _ = self.attn(x_seq, x_seq, x_seq)  # (B, N, C)

        # (B, N, C) -> (B, C, H, W)
        y = y.permute(0, 2, 1).contiguous().view(b, c, h, w)
        return y


class ConvFFN(nn.Module):
    """
    Conv ベースの FFN
      x -> Conv1x1(C, hidden) -> SiLU -> Conv1x1(hidden, C) -> x + ...
    """

    def __init__(self, c: int, expansion: float = 2.0):
        super().__init__()
        hidden = max(int(c * expansion), 1)
        self.fc1 = Conv(c, hidden, 1, 1)          # Conv + BN + SiLU
        self.fc2 = Conv(hidden, c, 1, 1, act=False)  # Conv + BN のみ

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fc1(x)
        y = self.fc2(y)
        return x + y


# custom_layers/detectsa.py の DetectSA 部分だけ差し替え

class DetectSA(um.Detect):
    """
    ASSF → (Conv1x1 -> MHSA -> Conv1x1 + shortcut) だけを挟んだ「置き換え head」に近い Detect。

    - ASSF で 3スケール特徴を融合して (C, C, C) に統一
    - 各スケールごとに:
        1) 1x1 Conv で C1 -> C2 に圧縮
        2) LocalMHSA2D で (ローカル) self-attention
        3) 1x1 Conv で C2 -> C1 に戻す
        4) 1x1 Conv による shortcut と residual add
    - ここでは FFN は入れず、論文 Figure 6 の構造に寄せる
    """

    def __init__(
        self,
        nc: int = 80,
        ch=(),
        inplace: bool = True,
        num_heads: int = 2,
        attn_ratio: float = 0.125,
    ):
        # ch は backbone/neck からの (C3, C4, C5) を想定
        assert len(ch) == 3, f"DetectSA expects 3 feature maps, got {ch}"
        c3, c4, c5 = ch

        # ASSF で揃える出力チャネル数（ここでは P3 と同じ C3）
        c_out = c3

        # Detect 側には「ASSF 後」のチャネル (c_out, c_out, c_out) を渡す
        ch_assf = (c_out, c_out, c_out)
        super().__init__(nc=nc, ch=ch_assf)

        # ASSF 本体（入力は元の c3, c4, c5）
        self.assf = ASSF(c3=c3, c4=c4, c5=c5, c_out=c_out)

        self.nl = len(ch_assf)
        self.ch = list(ch_assf)  # = [c_out, c_out, c_out]

        # ---- Self-Attention 用のチャネル圧縮設定 ----
        # C2 = max(8, int(C1 * attn_ratio))
        c2_list = [max(8, int(c * attn_ratio)) for c in self.ch]

        # 1x1 Conv で C1 -> C2
        self.reduce = nn.ModuleList(
            [Conv(c1, c2, 1, 1) for c1, c2 in zip(self.ch, c2_list)]
        )

        # Local Window MHSA
        self.mhsa = nn.ModuleList(
            [GlobalMHSA2D(c2, num_heads=num_heads)
             for c2 in c2_list]
        )

        # 1x1 Conv で C2 -> C1
        self.expand = nn.ModuleList(
            [Conv(c2, c1, 1, 1) for c2, c1 in zip(c2_list, self.ch)]
        )

        # shortcut 用の 1x1 Conv (shape を合わせる)
        # act=False にしておくと、完全に恒等射に近い振る舞いになる
        self.short = nn.ModuleList(
            [nn.Identity() for _ in self.ch]
        )

    def forward(self, x):
        """
        x: [P3, P4, P5] with channels (C3, C4, C5) from the neck
        """
        assert isinstance(x, (list, tuple)) and len(x) == self.nl, \
            f"Expected {self.nl} feature maps, got {len(x)}"

        # 1) ASSF で multi-scale fusion & channel unify (→ 3 枚とも C_out ch)
        p3, p4, p5 = self.assf(x)
        x_assf = [p3, p4, p5]

        # 2) Self-Attention ブロック
        attn_feats = []
        for i, xi in enumerate(x_assf):
            # Conv1x1 でチャンネル圧縮 C1 -> C2
            y = self.reduce[i](xi)

            # Local Window MHSA
            y = self.mhsa[i](y)

            # Conv1x1 で C2 -> C1
            y = self.expand[i](y)

            # shortcut + residual (Conv1x1 で shape を合わせたうえで add)
            skip = self.short[i](xi)
            y = y + skip

            attn_feats.append(y)

        # 3) 最後に YOLO Detect へ
        #   → 親クラス Detect の中身は「最終予測用の Conv 群」のみを使うイメージ
        return super().forward(attn_feats)
