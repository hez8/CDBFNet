

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple






class _DecoderBlock(nn.Module):


    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(32, out_ch) if out_ch >= 8 else out_ch, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(32, out_ch) if out_ch >= 8 else out_ch, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))






class DepthDecoder(nn.Module):


    def __init__(self, pretrained_depth: Optional[str] = None):
        super().__init__()
        self.block3 = _DecoderBlock(1024, 512, 256)
        self.block2 = _DecoderBlock(256, 256, 128)
        self.block1 = _DecoderBlock(128, 64, 64)
        self.head = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid(),
        )

        if pretrained_depth is not None:
            self._load_pretrained(pretrained_depth)

    def _load_pretrained(self, path: str):
        state = torch.load(path, map_location="cpu")
        if "state_dict" in state:
            state = state["state_dict"]

        if any(k.startswith("module.") for k in state.keys()):
            from collections import OrderedDict
            state = OrderedDict(
                (k.replace("module.", "", 1), v) for k, v in state.items()
            )
        self.load_state_dict(state, strict=False)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:

        f1, f2, f3, f4 = features
        d3 = self.block3(f4, f3)
        d2 = self.block2(d3, f2)
        d1 = self.block1(d2, f1)
        depth_lr = self.head(d1)

        H_in = f1.shape[2] * 4
        W_in = f1.shape[3] * 4
        depth = F.interpolate(depth_lr, size=(H_in, W_in), mode="bilinear", align_corners=False)
        return depth






class BasisDecoder(nn.Module):


    def __init__(self, in_channels_list: List[int], k_bases: int):
        super().__init__()
        self.k_bases = k_bases


        self.block3 = _DecoderBlock(1024, 512, 256)
        self.block2 = _DecoderBlock(256, 256, 128)
        self.block1 = _DecoderBlock(128, 64, 64)
        self.head_C = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, k_bases, 3, padding=1),
        )


        self.roughness_head = nn.Sequential(
            nn.Conv2d(1024, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Softplus(),
        )

    def forward(
        self, features: List[torch.Tensor], temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        f1, f2, f3, f4 = features


        d3 = self.block3(f4, f3)
        d2 = self.block2(d3, f2)
        d1 = self.block1(d2, f1)
        C_lr = self.head_C(d1)
        C_lr = F.interpolate(C_lr, scale_factor=4, mode="bilinear", align_corners=False)


        C = F.softmax(C_lr / temperature, dim=1)


        n_map_lr = self.roughness_head(f4)
        n_map = F.interpolate(n_map_lr, scale_factor=16, mode="bilinear", align_corners=False)
        n_map = n_map + 2.0

        return C, n_map






class ShadowDecoder(nn.Module):


    def __init__(self):
        super().__init__()
        self.block3 = _DecoderBlock(1024, 512, 256)
        self.block2 = _DecoderBlock(256, 256, 128)
        self.block1 = _DecoderBlock(128, 64, 64)
        self.head = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:

        f1, f2, f3, f4 = features
        d3 = self.block3(f4, f3)
        d2 = self.block2(d3, f2)
        d1 = self.block1(d2, f1)
        L_S_P_lr = self.head(d1)
        return F.interpolate(L_S_P_lr, scale_factor=4, mode="bilinear", align_corners=False)


class DirectPredictionDecoder(nn.Module):


    def __init__(self, out_channels: int):
        super().__init__()
        self.block3 = _DecoderBlock(1024, 512, 256)
        self.block2 = _DecoderBlock(256, 256, 128)
        self.block1 = _DecoderBlock(128, 64, 64)
        self.head = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        f1, f2, f3, f4 = features
        decoded = self.block3(f4, f3)
        decoded = self.block2(decoded, f2)
        decoded = self.block1(decoded, f1)
        prediction = self.head(decoded)
        output_size = (f1.shape[2] * 4, f1.shape[3] * 4)
        return F.interpolate(
            prediction, size=output_size, mode="bilinear", align_corners=False
        )
