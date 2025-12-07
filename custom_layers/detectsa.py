import torch
import torch.nn as nn
import torch.nn.functional as F

import ultralytics.nn.modules as um
from ultralytics.nn.modules import Conv  # YOLOv8 と同じ Conv を使う

from custom_layers.assf import ASSF


class LocalMHSA2D(nn.Module):
    """
    メモリ効率を意識した 2D feature map 用 Multi-Head Self-Attention.
    - 入力: (B, C, H, W)
    - 出力: 同じ shape
    - グローバル全画素ではなく、小さなウィンドウごとに MHSA をかける
    """

    def __init__(self, c: int, num_heads: int = 4, window_size: int = 8):
        super().__init__()
        self.c = c
        self.num_heads = num_heads
        self.window_size = window_size

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
        ws = self.window_size

        # H, W を window_size の倍数にパディング
        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws
        if pad_h or pad_w:
            # F.pad の引数順: (left, right, top, bottom)
            x = F.pad(x, (0, pad_w, 0, pad_h))
        b, c, h2, w2 = x.shape

        # (B, C, H2, W2) -> (B, C, nH, ws, nW, ws)
        n_h = h2 // ws
        n_w = w2 // ws
        x = x.view(b, c, n_h, ws, n_w, ws)

        # -> (B, nH, nW, ws, ws, C)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()

        # 各 window を1シーケンスにまとめて MultiheadAttention
        # (B * nH * nW, ws*ws, C)
        x = x.view(b * n_h * n_w, ws * ws, c)

        y, _ = self.attn(x, x, x)

        # もとの window 形状に戻す
        y = y.view(b, n_h, n_w, ws, ws, c)
        y = y.permute(0, 5, 1, 3, 2, 4).contiguous()
        y = y.view(b, c, h2, w2)

        # パディング部分を除去
        if pad_h or pad_w:
            y = y[:, :, :h, :w]

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


class DetectSA(um.Detect):
    """
    論文 Figure 6 の「Memory-efficient self-attention」に近づけた Detect ヘッド。

    - ASSF で 3スケール特徴を融合
    - 各スケールごとに:
        1) 1x1 Conv で C1 -> C2 に圧縮
        2) LocalMHSA2D で window attention
        3) 1x1 Conv で C2 -> C1 に拡張
        4) shortcut Conv との residual
        5) ConvFFN による FFN + residual
    - その出力を親クラス Detect に渡す
    """

    def __init__(
        self,
        nc: int = 80,
        ch=(),
        inplace: bool = True,
        num_heads: int = 4,
        attn_ratio: float = 0.25,
        window_size: int = 8,
        ffn_ratio: float = 2.0,
    ):
        # ch は backbone/neck からの (C3, C4, C5) = (256, 512, 1024) 想定
        assert len(ch) == 3, f"DetectSA expects 3 feature maps, got {ch}"
        c3, c4, c5 = ch

        # ASSF で揃える出力チャネル数（ここでは P3 と同じ 256）
        c_out = c3

        # Detect 側には「ASSF 後」のチャネル (256, 256, 256) を渡す
        ch_assf = (c_out, c_out, c_out)
        super().__init__(nc=nc, ch=ch_assf)

        # ASSF 本体（入力は元の 256, 512, 1024）
        self.assf = ASSF(c3=c3, c4=c4, c5=c5, c_out=c_out)

        self.nl = len(ch_assf)
        self.ch = list(ch_assf)  # = [256, 256, 256]

        # ---- Self-Attention 用のチャネル圧縮設定 ----
        # C2 = max(8, int(C1 * attn_ratio))
        c2_list = [max(8, int(c * attn_ratio)) for c in self.ch]

        # 1x1 Conv で C1 -> C2
        self.reduce = nn.ModuleList(
            [Conv(c1, c2, 1, 1) for c1, c2 in zip(self.ch, c2_list)]
        )

        # Local Window MHSA
        self.mhsa = nn.ModuleList(
            [LocalMHSA2D(c2, num_heads=num_heads, window_size=window_size)
             for c2 in c2_list]
        )

        # 1x1 Conv で C2 -> C1
        self.expand = nn.ModuleList(
            [Conv(c2, c1, 1, 1) for c2, c1 in zip(c2_list, self.ch)]
        )

        # shortcut 用の 1x1 Conv (shape を合わせる)
        self.short = nn.ModuleList(
            [Conv(c1, c1, 1, 1) for c1 in self.ch]
        )

        # FFN
        self.ffn = nn.ModuleList(
            [ConvFFN(c1, expansion=ffn_ratio) for c1 in self.ch]
        )

    def forward(self, x):
        """
        x: [P3, P4, P5] with channels (256, 512, 1024) from the neck
        """
        assert isinstance(x, (list, tuple)) and len(x) == self.nl, \
            f"Expected {self.nl} feature maps, got {len(x)}"

        # 1) ASSF で multi-scale fusion & channel unify (→ 3 枚とも 256ch)
        p3, p4, p5 = self.assf(x)
        x_assf = [p3, p4, p5]

        # 2) Self-Attention + 3) FFN
        attn_feats = []
        for i, xi in enumerate(x_assf):
            # SA ブロック
            y = self.reduce[i](xi)       # C1 -> C2
            y = self.mhsa[i](y)         # local window attention
            y = self.expand[i](y)       # C2 -> C1

            # shortcut + residual
            skip = self.short[i](xi)
            y = y + skip

            # FFN ブロック (residual 内部で add)
            y = self.ffn[i](y)

            attn_feats.append(y)

        # 3) 最後に YOLO Detect へ
        return super().forward(attn_feats)
