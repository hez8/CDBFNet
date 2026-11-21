

from typing import Dict

import torch
import torch.nn as nn


class WHDRCalibrationHead(nn.Module):


    def __init__(
        self,
        n_channels: int = 3,
        hidden_channels: int = 16,
        residual_scale: float = 0.1,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.residual_scale = residual_scale
        self.eps = eps

        self.net = nn.Sequential(
            nn.Conv2d(n_channels * 2, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, n_channels + 1, kernel_size=1),
        )
        final = self.net[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, reflectance: torch.Tensor) -> Dict[str, torch.Tensor]:
        refl = reflectance.clamp(min=self.eps, max=1.0)
        features = torch.cat([refl, refl.log()], dim=1)
        raw = self.net(features)

        residual = torch.tanh(raw[:, :self.n_channels]) * self.residual_scale
        calibrated = (refl * residual.exp()).clamp(0.0, 1.0)

        score_base = calibrated.mean(dim=1, keepdim=True).clamp(min=self.eps).log()
        score = score_base + raw[:, self.n_channels:self.n_channels + 1]

        return {
            "reflectance_calibrated": calibrated,
            "whdr_score": score,
        }
