# custom_layers/assf.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianSmoothing(nn.Module):
    """
    各チャネル独立の depthwise 3x3 Conv で Gaussian 風の平滑化を行うモジュール。
    """
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            groups=channels,
            bias=False,
        )

        # とりあえず「平均フィルタ」で初期化 → 学習で Gaussian に寄せるイメージ
        with torch.no_grad():
            k = torch.ones((channels, 1, kernel_size, kernel_size))
            k = k / (kernel_size * kernel_size)
            self.conv.weight.copy_(k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ASSF(nn.Module):
    """
    ASSF (TFE + SSFF) に近づけた実装（TFE のチャネル比を反映した版）。

    - 入力: [P3, P4, P5]
        P3: 高解像度
        P4: 中解像度 (基準スケール)
        P5: 低解像度

    - TFE 部:
        大スケール (P3): ch = 0.5C, Max+Avg Pool で P4 と同じ解像度へ
        中スケール (P4): ch = C, そのまま
        小スケール (P5): ch = 2C, 最近傍補間で P4 解像度へ

      その後、3つをそれぞれ 1x1 Conv で C チャネルに揃え、Gaussian smoothing → SSFF に渡す。
    """

    def __init__(self, c3: int, c4: int, c5: int, c_out: int):
        super().__init__()

        # TFE のチャネル比:
        #   P3: 0.5C, P4: C, P5: 2C
        c3_mid = max(c_out // 2, 1)
        c4_mid = c_out
        c5_mid = c_out * 2

        # ---- TFE: 各スケールで一次エンコード ----
        # P3 -> 0.5C
        self.lateral3 = nn.Conv2d(c3, c3_mid, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(c3_mid)

        # P4 -> C
        self.lateral4 = nn.Conv2d(c4, c4_mid, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn4 = nn.BatchNorm2d(c4_mid)

        # P5 -> 2C
        self.lateral5 = nn.Conv2d(c5, c5_mid, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn5 = nn.BatchNorm2d(c5_mid)

        self.act = nn.SiLU()

        # ---- TFE 後に C チャネルへ揃える 1x1 Conv ----
        self.to_c3 = nn.Conv2d(c3_mid, c_out, kernel_size=1, stride=1, padding=0, bias=False)
        self.to_c4 = nn.Conv2d(c4_mid, c_out, kernel_size=1, stride=1, padding=0, bias=False)
        self.to_c5 = nn.Conv2d(c5_mid, c_out, kernel_size=1, stride=1, padding=0, bias=False)

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

    def forward(self, x):
        """
        x: [P3, P4, P5]
        """
        assert isinstance(x, (list, tuple)) and len(x) == 3, \
            f"ASSF expects 3 feature maps, got {len(x)}"

        x3, x4, x5 = x

        # ---- TFE: 各スケールで一次エンコード ----
        # P3: [B, c3_mid, H3, W3]
        p3_enc = self.act(self.bn3(self.lateral3(x3)))
        # P4: [B, c4_mid=C, H4, W4]
        p4_enc = self.act(self.bn4(self.lateral4(x4)))
        # P5: [B, c5_mid=2C, H5, W5]
        p5_enc = self.act(self.bn5(self.lateral5(x5)))

        # ---- 解像度を P4 (中間スケール) に揃える ----
        B, C4, H4, W4 = p4_enc.shape

        # 大スケール (P3): Max + Avg Pool で downsample
        p3_max = F.adaptive_max_pool2d(p3_enc, (H4, W4))
        p3_avg = F.adaptive_avg_pool2d(p3_enc, (H4, W4))
        p3_mid = 0.5 * (p3_max + p3_avg)

        # 中スケール (P4): そのまま
        p4_mid = p4_enc

        # 小スケール (P5): 最近傍補間で upsample
        p5_mid = F.interpolate(p5_enc, size=(H4, W4), mode="nearest")

        # ---- C チャネルに揃える ----
        t3 = self.to_c3(p3_mid)  # [B, C, H4, W4]
        t4 = self.to_c4(p4_mid)
        t5 = self.to_c5(p5_mid)

        # ---- Gaussian smoothing 風フィルタ ----
        g3 = self.smooth3(t3)
        g4 = self.smooth4(t4)
        g5 = self.smooth5(t5)

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

        # ★ residual に使う t3,t4,t5 も、各スケールの解像度に揃える
        r3 = F.interpolate(t3, size=x3.shape[-2:], mode="nearest")  # [B, C, H3, W3]
        r4 = F.interpolate(t4, size=x4.shape[-2:], mode="nearest")  # [B, C, H4, W4]
        r5 = F.interpolate(t5, size=x5.shape[-2:], mode="nearest")  # [B, C, H5, W5]

        # ---- 最終整形 Conv + BN + SiLU + 残差 ----
        p3_out = self.act(self.out_bn3(self.out3(f3)) + r3)
        p4_out = self.act(self.out_bn4(self.out4(f4)) + r4)
        p5_out = self.act(self.out_bn5(self.out5(f5)) + r5)


        return p3_out, p4_out, p5_out