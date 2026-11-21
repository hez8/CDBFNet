

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.amp import autocast
from typing import List, Tuple, Optional






def _replace_bn_with_gn(module: nn.Module, num_groups: int = 32) -> nn.Module:

    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            gn = nn.GroupNorm(
                num_groups=min(num_groups, child.num_features),
                num_channels=child.num_features,
            )
            setattr(module, name, gn)
        else:
            _replace_bn_with_gn(child, num_groups)
    return module






class _FeatureAdapter(nn.Module):


    def __init__(self, in_channels: List[int]):
        super().__init__()
        target = SharedEncoder.OUT_CHANNELS
        self.projections = nn.ModuleList()
        self.norms = nn.ModuleList()
        for ic, tc in zip(in_channels, target):
            if ic != tc:
                self.projections.append(nn.Conv2d(ic, tc, 1, bias=False))

                num_groups = min(32, tc) if tc >= 32 else tc
                self.norms.append(nn.GroupNorm(num_groups, tc))
            else:
                self.projections.append(nn.Identity())
                self.norms.append(nn.Identity())

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        return [norm(proj(f)) for proj, norm, f in
                zip(self.projections, self.norms, features)]






class SharedEncoder(nn.Module):


    OUT_CHANNELS = [64, 256, 512, 1024]


    _BACKBONE_CHANNELS = {
        "resnet50":       [64, 256, 512, 1024],

        "swin_t":         [96, 96, 192, 1152],

        "convnext_tiny":  [96, 96, 192, 1152],
    }

    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        n_channels: int = 3,
    ):
        super().__init__()

        if backbone == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.resnet50(weights=weights)
            model = _replace_bn_with_gn(model)
            self._build_resnet(model, n_channels)

        elif backbone == "swin_t":
            weights = models.Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.swin_t(weights=weights)
            self._build_swin(model, n_channels)

        elif backbone == "convnext_tiny":
            weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.convnext_tiny(weights=weights)
            self._build_convnext(model, n_channels)

        else:
            raise ValueError(
                f"Unknown backbone: {backbone}. "
                f"Supported: resnet50, swin_t, convnext_tiny"
            )


        raw_channels = self._BACKBONE_CHANNELS[backbone]
        self.adapter = _FeatureAdapter(raw_channels)
        self._backbone_name = backbone


    def _build_resnet(self, model: nn.Module, n_channels: int):
        if n_channels != 3:
            model.conv1 = nn.Conv2d(
                n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        self.stem = nn.Sequential(
            model.conv1, model.bn1, model.relu, model.maxpool
        )
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3

    def _forward_resnet(self, x: torch.Tensor) -> List[torch.Tensor]:
        s0 = self.stem(x)
        f1 = s0
        f2 = self.layer1(s0)
        f3 = self.layer2(f2)
        f4 = self.layer3(f3)
        return [f1, f2, f3, f4]


    def _build_swin(self, model: nn.Module, n_channels: int):
        if n_channels != 3:

            old_conv = model.features[0][0]
            model.features[0][0] = nn.Conv2d(
                n_channels, old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )
        self.swin_features = model.features



        self._swin_stage_names = ["0", "2", "4", "6"]

    def _forward_swin(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []

        for i, layer in enumerate(self.swin_features):
            x = layer(x)

            if i == 0:
                features.append(x.permute(0, 3, 1, 2))
            elif i in (1, 3, 5, 7):
                features.append(x.permute(0, 3, 1, 2))


        f1 = features[0]
        f2 = features[1]
        f3 = features[2]
        f4_raw = features[3]

        s4_up = F.interpolate(features[4], size=f4_raw.shape[2:],
                              mode="bilinear", align_corners=False)
        f4 = torch.cat([f4_raw, s4_up], dim=1)

        return [f1, f2, f3, f4]


    def _build_convnext(self, model: nn.Module, n_channels: int):
        if n_channels != 3:
            old_conv = model.features[0][0]
            model.features[0][0] = nn.Conv2d(
                n_channels, old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )
        self.cn_features = model.features

    def _forward_convnext(self, x: torch.Tensor) -> List[torch.Tensor]:



        feats = []
        for i, layer in enumerate(self.cn_features):
            x = layer(x)
            if i in (0, 1, 3, 5, 7):
                feats.append(x)

        f1 = feats[0]
        f2 = feats[1]
        f3 = feats[2]
        f4_raw = feats[3]
        s4_up = F.interpolate(feats[4], size=f4_raw.shape[2:],
                              mode="bilinear", align_corners=False)
        f4 = torch.cat([f4_raw, s4_up], dim=1)
        return [f1, f2, f3, f4]


    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:

        if self._backbone_name in ("swin_t", "convnext_tiny"):
            with autocast('cuda', enabled=False):
                if self._backbone_name == "swin_t":
                    raw = self._forward_swin(x)
                else:
                    raw = self._forward_convnext(x)
        elif self._backbone_name == "resnet50":
            raw = self._forward_resnet(x)
        else:
            raw = self._forward_resnet(x)

        return self.adapter(raw)






class LightingEncoder(nn.Module):


    def __init__(self, n_channels: int = 3):
        super().__init__()
        C_amb = n_channels


        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_light = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 3 + C_amb * 2),
        )


        self.shadow_decoder = nn.Sequential(
            nn.Conv2d(1024, 256, 3, padding=1),
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 3, padding=1),
        )


        self.focal_base = nn.Parameter(torch.tensor(1.0))

    def forward(
        self, x: torch.Tensor, f4: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        B = x.shape[0]

        g = self.global_pool(f4).view(B, -1)
        light_params = self.fc_light(g).unsqueeze(-1).unsqueeze(-1)

        L_S_P = self.shadow_decoder(f4)

        focal_length = self.focal_base.view(1, 1, 1, 1).expand(B, -1, -1, -1)

        return light_params, L_S_P, focal_length
