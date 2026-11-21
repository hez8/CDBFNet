

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from .renderer import UnifiedLightingModule, multi_scale_normals
from .encoders import SharedEncoder, LightingEncoder
from .decoders import (
    BasisDecoder,
    DepthDecoder,
    DirectPredictionDecoder,
    ShadowDecoder,
)
from .whdr_head import WHDRCalibrationHead


class CDBFNet(nn.Module):


    def __init__(self, config: dict):
        super().__init__()
        n_channels = config["model"]["n_channels"]
        k_bases = config["model"]["k_bases"]
        self.ablation = {
            "independent_coefficients": False,
            "direct_reflectance": False,
            "single_lambertian": False,
            "without_normals": False,
            "without_visibility": False,
            "direct_intrinsic": False,
            **config["model"].get("ablation", {}),
        }


        self.encoder = SharedEncoder(
            backbone=config["model"]["encoder"],
            pretrained=config["model"]["pretrained_encoder"],
            n_channels=n_channels,
        )


        self.depth_decoder = DepthDecoder(
            pretrained_depth=config["model"].get("pretrained_depth", None)
        )
        self.basis_decoder = BasisDecoder(
            in_channels_list=SharedEncoder.OUT_CHANNELS,
            k_bases=k_bases,
        )
        self.shadow_decoder = ShadowDecoder()
        self.lighting_encoder = LightingEncoder(n_channels=n_channels)
        self.specular_basis_decoder = None
        if self.ablation["independent_coefficients"]:
            self.specular_basis_decoder = BasisDecoder(
                in_channels_list=SharedEncoder.OUT_CHANNELS,
                k_bases=k_bases,
            )
        self.direct_reflectance_decoder = None
        if self.ablation["direct_reflectance"]:
            self.direct_reflectance_decoder = DirectPredictionDecoder(2 * n_channels)
        self.direct_intrinsic_decoder = None
        if self.ablation["direct_intrinsic"]:
            self.direct_intrinsic_decoder = DirectPredictionDecoder(2 * n_channels)






        self.P1_raw = nn.Parameter(torch.randn(n_channels, k_bases, 1, 1) * 0.3 - 1.5)
        self.P2_raw = nn.Parameter(torch.randn(n_channels, k_bases, 1, 1) * 2.0)


        renderer_config = config["renderer"].copy()
        renderer_config["n_channels"] = n_channels
        renderer_config["single_lambertian"] = self.ablation["single_lambertian"]
        self.renderer = UnifiedLightingModule(renderer_config)

        whdr_cfg = config.get("whdr_head", {})
        self.whdr_head = None
        if whdr_cfg.get("enabled", False):
            self.whdr_head = WHDRCalibrationHead(
                n_channels=n_channels,
                hidden_channels=whdr_cfg.get("hidden_channels", 16),
                residual_scale=whdr_cfg.get("residual_scale", 0.1),
            )

        if self.ablation["direct_intrinsic"]:
            unused_modules = (
                self.depth_decoder,
                self.basis_decoder,
                self.shadow_decoder,
                self.lighting_encoder,
                self.renderer,
            )
            for module in unused_modules:
                for parameter in module.parameters():
                    parameter.requires_grad = False
            self.P1_raw.requires_grad = False
            self.P2_raw.requires_grad = False

        self._temperature = 1.0
        self._k_bases = k_bases





    @property
    def temperature(self) -> float:
        return self._temperature

    @temperature.setter
    def temperature(self, val: float):
        self._temperature = val





    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:


        features = self.encoder(x)

        if self.direct_intrinsic_decoder is not None:
            direct_layers = self.direct_intrinsic_decoder(features)
            direct_layers = F.interpolate(
                direct_layers, size=x.shape[2:], mode="bilinear", align_corners=False
            )
            reflectance, shading = direct_layers.chunk(2, dim=1)
            batch_size, _, height, width = x.shape
            zero_rgb = torch.zeros_like(reflectance)
            zero_scalar = torch.zeros(
                batch_size, 1, height, width, device=x.device, dtype=x.dtype
            )
            normal = torch.zeros(
                batch_size, 3, height, width, device=x.device, dtype=x.dtype
            )
            normal[:, 2:3] = 1.0
            coefficients = torch.full(
                (batch_size, self._k_bases, height, width),
                1.0 / self._k_bases,
                device=x.device,
                dtype=x.dtype,
            )
            P1, P2 = self._normalize_P()
            outputs = {
                "X_recon": reflectance * shading,
                "R1": zero_rgb,
                "R2": reflectance,
                "reflectance": reflectance,
                "C": coefficients,
                "Shading": shading,
                "N_vec": normal,
                "n_map": torch.ones_like(zero_scalar),
                "light_params": zero_scalar[:, :, :1, :1],
                "L_S_P": torch.ones_like(zero_scalar),
                "H_x": zero_scalar,
                "R_C_CosT_n": zero_scalar,
                "LS_CosT": zero_scalar,
                "focal_length": torch.ones_like(zero_scalar[:, :, :1, :1]),
                "P1": P1,
                "P2": P2,
            }
            if self.whdr_head is not None:
                outputs.update(self.whdr_head(reflectance))
            return outputs


        depth = self.depth_decoder(features)
        C, n_map = self.basis_decoder(features, self._temperature)
        C_specular = None
        if self.specular_basis_decoder is not None:
            C_specular, _ = self.specular_basis_decoder(features, self._temperature)
        L_S_P_shadow = self.shadow_decoder(features)
        light_params, L_S_P_light, focal_length = self.lighting_encoder(x, features[-1])



        input_size = x.shape[2:]
        depth = F.interpolate(depth, size=input_size,
                              mode="bilinear", align_corners=False)
        C = F.interpolate(C, size=input_size,
                          mode="bilinear", align_corners=False)
        if C_specular is not None:
            C_specular = F.interpolate(
                C_specular, size=input_size, mode="bilinear", align_corners=False
            )
        n_map = F.interpolate(n_map, size=input_size,
                              mode="bilinear", align_corners=False)
        L_S_P_shadow = F.interpolate(L_S_P_shadow, size=input_size,
                                     mode="bilinear", align_corners=False)


        img_grad = self._compute_image_gradient(x)
        N_vec = multi_scale_normals(depth, img_grad)
        if self.ablation["without_normals"]:
            N_vec = torch.zeros_like(N_vec)
            N_vec[:, 2:3] = 1.0


        P1, P2 = self._normalize_P()


        L_S_P_full = F.interpolate(L_S_P_light, size=x.shape[2:],
                                   mode="bilinear", align_corners=False)


        L_S_P = L_S_P_full * L_S_P_shadow
        if self.ablation["without_visibility"]:
            L_S_P = torch.ones_like(L_S_P)


        direct_r1 = direct_r2 = None
        if self.direct_reflectance_decoder is not None:
            direct_layers = self.direct_reflectance_decoder(features)
            direct_layers = F.interpolate(
                direct_layers, size=input_size, mode="bilinear", align_corners=False
            )
            direct_r1, direct_r2 = direct_layers.chunk(2, dim=1)
        rendered = self.renderer(
            N_vec, C, P1, P2, n_map, light_params, L_S_P, focal_length,
            C_specular=C_specular,
            R1_override=direct_r1,
            R2_override=direct_r2,
        )

        reflectance = rendered["R1"] + rendered["R2"]
        outputs = {
            "X_recon":      rendered["X_final"],
            "R1":           rendered["R1"],
            "R2":           rendered["R2"],
            "reflectance":  reflectance,
            "C":            C,
            "Shading":      rendered["Shading"],
            "N_vec":        N_vec,
            "n_map":        rendered["n_map"],
            "light_params": light_params,
            "L_S_P":        L_S_P,
            "H_x":          rendered["H_x"],
            "R_C_CosT_n":   rendered["R_C_CosT_n"],
            "LS_CosT":      rendered["LS_CosT"],
            "focal_length": focal_length,
            "P1":           P1,
            "P2":           P2,
        }
        if C_specular is not None:
            outputs["C_specular"] = C_specular
        if self.whdr_head is not None:
            outputs.update(self.whdr_head(reflectance))
        return outputs



    def _compute_image_gradient(self, x: torch.Tensor) -> torch.Tensor:

        dx = x[:, :, :, 1:] - x[:, :, :, :-1]
        dy = x[:, :, 1:, :] - x[:, :, :-1, :]

        mag_x = dx[:, :, :-1, :].pow(2).mean(dim=1, keepdim=True)
        mag_y = dy[:, :, :, :-1].pow(2).mean(dim=1, keepdim=True)
        mag = mag_x + mag_y
        mag = F.pad(mag, (1, 0, 1, 0))
        return mag

    def _normalize_P(self) -> tuple:

        return torch.sigmoid(self.P1_raw), torch.sigmoid(self.P2_raw)





    def base_parameters(self):

        return [self.P1_raw, self.P2_raw]

    def decoder_parameters(self):

        base_ids = {id(p) for p in self.base_parameters()}
        return [p for p in self.parameters() if id(p) not in base_ids]
