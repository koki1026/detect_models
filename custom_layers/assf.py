# custom_layers/assf.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class ASSF(nn.Module):
    """
    Simple ASSF block (TFE + SSFF の簡略版)

    入出力の前提:
        - 入力: list/tuple of 3 tensors [P3, P4, P5]
            P3: [B, C3, H3, W3]  (高解像度)
            P4: [B, C4, H4, W4]
            P5: [B, C5, H5, W5]  (低解像度)

        - 出力: (P3_out, P4_out, P5_out)
            それぞれ元と同じ解像度だが，チャネル数は c_out に統一
    """

    def __init__(self, c3: int, c4: int, c5: int, c_out: int):
        super().__init__()

        # ---- チャネルを c_out に揃える 1x1 Conv (TFE の前処理) ----
        self.lateral3 = nn.Conv2d(c3, c_out, kernel_size=1, stride=1, padding=0, bias=False)
        self.lateral4 = nn.Conv2d(c4, c_out, kernel_size=1, stride=1, padding=0, bias=False)
        self.lateral5 = nn.Conv2d(c5, c_out, kernel_size=1, stride=1, padding=0, bias=False)

        self.bn3 = nn.BatchNorm2d(c_out)
        self.bn4 = nn.BatchNorm2d(c_out)
        self.bn5 = nn.BatchNorm2d(c_out)

        self.act = nn.SiLU()

        # ---- TFE: 3スケールを P3 解像度に合わせて concat → Conv ----
        self.tfe_conv = nn.Sequential(
            nn.Conv2d(c_out * 3, c_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.SiLU(),
        )

        # ---- SSFF 簡略版: fused を各スケールに再投影して Conv ----
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

        # 1x1 Conv でチャネル揃え
        p3 = self.act(self.bn3(self.lateral3(x3)))
        p4 = self.act(self.bn4(self.lateral4(x4)))
        p5 = self.act(self.bn5(self.lateral5(x5)))

        # すべて P3 解像度 (H3, W3) にアップ／ダウンサンプル
        B, C, H3, W3 = p3.shape
        p4_up = F.interpolate(p4, size=(H3, W3), mode="nearest")
        p5_up = F.interpolate(p5, size=(H3, W3), mode="nearest")

        # ---- TFE: concat → conv で 3スケール融合 ----
        tfe = torch.cat([p3, p4_up, p5_up], dim=1)  # [B, 3*C, H3, W3]
        fused = self.tfe_conv(tfe)                  # [B, C, H3, W3]

        # ---- SSFF (簡略版): fused を各スケール解像度に戻して Conv ----
        # P3 用
        p3_out = self.act(self.out_bn3(self.out3(fused)))

        # P4 用: 解像度を P4 に合わせてから conv
        p4_resized = F.interpolate(fused, size=x4.shape[-2:], mode="nearest")
        p4_out = self.act(self.out_bn4(self.out4(p4_resized)))

        # P5 用
        p5_resized = F.interpolate(fused, size=x5.shape[-2:], mode="nearest")
        p5_out = self.act(self.out_bn5(self.out5(p5_resized)))

        # 将来的に:
        #  - ここに Gaussian smoothing
        #  - 3D Conv (scale dimension を追加) などを追加して SSFF を本家に近づける

        return p3_out, p4_out, p5_out