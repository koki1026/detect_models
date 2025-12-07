# custom_layers/assf.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianSmoothing(nn.Module):
    """
    各チャネル独立の depthwise 3x3 Conv で Gaussian 風の平滑化を行うモジュール。
    論文中の Gaussian smoothing を、学習可能なフィルタとして近似するイメージ。
    """

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.channels = channels
        self.kernel_size = kernel_size
        padding = kernel_size // 2

        # depthwise conv
        self.conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            groups=channels,
            bias=False,
        )

        # 初期値として単純に平均フィルタを入れておく（Gaussian に近い滑らかさ）
        with torch.no_grad():
            k = torch.ones((channels, 1, kernel_size, kernel_size))
            k = k / (kernel_size * kernel_size)
            self.conv.weight.copy_(k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ASSF(nn.Module):
    """
    ASSF (TFE + SSFF) に近づけた実装。

    入出力の前提:
        - 入力: list/tuple of 3 tensors [P3, P4, P5]
            P3: [B, C3, H3, W3]  (高解像度)
            P4: [B, C4, H4, W4]  (中解像度, 基準スケール)
            P5: [B, C5, H5, W5]  (低解像度)

        - 出力: (P3_out, P4_out, P5_out)
            それぞれ元と同じ解像度だが，チャネル数は c_out に統一
    """

    def __init__(self, c3: int, c4: int, c5: int, c_out: int):
        super().__init__()

        # ---- TFE: チャネルを c_out に揃える 1x1 Conv + BN + SiLU ----
        self.lateral3 = nn.Conv2d(c3, c_out, kernel_size=1, stride=1, padding=0, bias=False)
        self.lateral4 = nn.Conv2d(c4, c_out, kernel_size=1, stride=1, padding=0, bias=False)
        self.lateral5 = nn.Conv2d(c5, c_out, kernel_size=1, stride=1, padding=0, bias=False)

        self.bn3 = nn.BatchNorm2d(c_out)
        self.bn4 = nn.BatchNorm2d(c_out)
        self.bn5 = nn.BatchNorm2d(c_out)

        self.act = nn.SiLU()

        # ---- Gaussian smoothing 風フィルタ（各スケールごと） ----
        self.smooth3 = GaussianSmoothing(c_out, kernel_size=3)
        self.smooth4 = GaussianSmoothing(c_out, kernel_size=3)
        self.smooth5 = GaussianSmoothing(c_out, kernel_size=3)

        # ---- SSFF: scale-sequence を 3D Conv で融合 ----
        # 入力 shape: [B, C, D=3, H4, W4] を想定
        self.ssff = nn.Sequential(
            nn.Conv3d(c_out, c_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(c_out),
            nn.SiLU(),
        )

        # ---- 各スケールへの再投影（2D Conv + BN + SiLU） ----
        self.out3 = nn.Conv2d(c_out, c_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.out4 = nn.Conv2d(c_out, c_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.out5 = nn.Conv2d(c_out, c_out, kernel_size=3, stride=1, padding=1, bias=False)

        self.out_bn3 = nn.BatchNorm2d(c_out)
        self.out_bn4 = nn.BatchNorm2d(c_out)
        self.out_bn5 = nn.BatchNorm2d(c_out)

    def forward(self, xs):
        """
        xs: [P3, P4, P5] の list or tuple を想定
        """

        if not isinstance(xs, (list, tuple)) or len(xs) != 3:
            raise ValueError(f"ASSF expects a list/tuple of 3 tensors [P3, P4, P5], got {type(xs)}")

        x3, x4, x5 = xs  # P3, P4, P5

        # ---- 1x1 Conv でチャネル揃え (TFE の前処理) ----
        p3 = self.act(self.bn3(self.lateral3(x3)))
        p4 = self.act(self.bn4(self.lateral4(x4)))
        p5 = self.act(self.bn5(self.lateral5(x5)))

        # ---- 解像度を P4 (中間スケール) に揃える ----
        B, C, H4, W4 = p4.shape

        # P3 は downsample, P5 は upsample
        p3_to_mid = F.interpolate(p3, size=(H4, W4), mode="nearest")
        p4_to_mid = p4
        p5_to_mid = F.interpolate(p5, size=(H4, W4), mode="nearest")

        # ---- Gaussian smoothing 風フィルタ ----
        g3 = self.smooth3(p3_to_mid)
        g4 = self.smooth4(p4_to_mid)
        g5 = self.smooth5(p5_to_mid)

        # ---- SSFF: scale-sequence を作成して 3D Conv ----
        # [B, C, H4, W4] * 3 → [B, C, D=3, H4, W4]
        seq = torch.stack([g3, g4, g5], dim=2)

        # 3D Conv によるスケール間融合
        fused_seq = self.ssff(seq)  # [B, C, 3, H4, W4]

        # D 次元を 3 つに分解
        f3_mid = fused_seq[:, :, 0, :, :]  # [B, C, H4, W4]
        f4_mid = fused_seq[:, :, 1, :, :]
        f5_mid = fused_seq[:, :, 2, :, :]

        # ---- 各スケール解像度に戻す ----
        f3 = F.interpolate(f3_mid, size=x3.shape[-2:], mode="nearest")
        f4 = F.interpolate(f4_mid, size=x4.shape[-2:], mode="nearest")
        f5 = F.interpolate(f5_mid, size=x5.shape[-2:], mode="nearest")

        # ---- 最終整形 Conv + BN + SiLU ----
        p3_out = self.act(self.out_bn3(self.out3(f3)) + p3)
        p4_out = self.act(self.out_bn4(self.out4(f4)) + p4)
        p5_out = self.act(self.out_bn5(self.out5(f5)) + p5)

        return p3_out, p4_out, p5_out