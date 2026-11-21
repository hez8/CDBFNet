

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, Dict






class HxFunc(nn.Module):


    def __init__(self, pretrained_path: Optional[str] = None, freeze: bool = False):
        super().__init__()
        self.fc1 = nn.Linear(2, 100)
        self.fc2 = nn.Linear(100, 1)

        if pretrained_path is not None:
            state = torch.load(pretrained_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]

            if any(k.startswith("module.") for k in state.keys()):
                from collections import OrderedDict
                state = OrderedDict(
                    (k.replace("module.", "", 1), v) for k, v in state.items()
                )
            self.load_state_dict(state, strict=False)

        if freeze:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, combined_input: torch.Tensor) -> torch.Tensor:

        x = torch.sigmoid(self.fc1(combined_input))
        return F.softplus(self.fc2(x))






def _sobel_kernel(size: int, direction: str) -> torch.Tensor:

    if size == 3:
        if direction == "x":
            return torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32) / 8.0
        else:
            return torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32) / 8.0
    elif size == 5:
        if direction == "x":
            k = torch.tensor([
                [-1, -2, 0, 2, 1],
                [-2, -3, 0, 3, 2],
                [-3, -5, 0, 5, 3],
                [-2, -3, 0, 3, 2],
                [-1, -2, 0, 2, 1],
            ], dtype=torch.float32) / 48.0
        else:
            k = torch.tensor([
                [-1, -2, -3, -2, -1],
                [-2, -3, -5, -3, -2],
                [0, 0, 0, 0, 0],
                [2, 3, 5, 3, 2],
                [1, 2, 3, 2, 1],
            ], dtype=torch.float32) / 48.0
        return k
    raise ValueError(f"Unsupported kernel size: {size}")


def compute_normals(depth: torch.Tensor) -> torch.Tensor:

    sx = _sobel_kernel(3, "x").view(1, 1, 3, 3).to(depth.device)
    sy = _sobel_kernel(3, "y").view(1, 1, 3, 3).to(depth.device)

    grad_x = F.conv2d(depth, sx, padding=1)
    grad_y = F.conv2d(depth, sy, padding=1)

    normal = torch.cat([-grad_x, -grad_y, torch.ones_like(grad_x)], dim=1)
    norm = normal.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8)
    return normal / norm


def multi_scale_normals(
    depth: torch.Tensor,
    image_gradient: Optional[torch.Tensor] = None,
) -> torch.Tensor:

    n3 = compute_normals(depth)

    sx5 = _sobel_kernel(5, "x").view(1, 1, 5, 5).to(depth.device)
    sy5 = _sobel_kernel(5, "y").view(1, 1, 5, 5).to(depth.device)
    gx5 = F.conv2d(depth, sx5, padding=2)
    gy5 = F.conv2d(depth, sy5, padding=2)
    n5_raw = torch.cat([-gx5, -gy5, torch.ones_like(gx5)], dim=1)
    n5 = n5_raw / n5_raw.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8)

    if image_gradient is not None:
        edge_confidence = torch.exp(-image_gradient / 0.1)
        alpha = edge_confidence

        if alpha.shape[2:] != n3.shape[2:]:
            alpha = F.interpolate(alpha, size=n3.shape[2:],
                                  mode="bilinear", align_corners=False)
    else:
        alpha = 0.5

    return F.normalize(alpha * n3 + (1 - alpha) * n5, p=2, dim=1)






class UnifiedLightingModule(nn.Module):


    def __init__(self, config: dict):
        super().__init__()
        self.dielectric = config.get("dielectric", True)
        self.spherical_light = config.get("spherical_light", True)
        self.single_lambertian = config.get("single_lambertian", False)
        self.n_channels = config.get("n_channels", 3)

        self.Hx_func = HxFunc(
            pretrained_path=config.get("pretrained_Hx", None),
            freeze=config.get("freeze_Hx", False),
        )


    def forward(
        self,
        N_vec: torch.Tensor,
        C: torch.Tensor,
        P1: torch.Tensor,
        P2: torch.Tensor,
        roughness: torch.Tensor,
        light_params: torch.Tensor,
        L_S_P: torch.Tensor,
        focal_length: torch.Tensor,
        C_specular: Optional[torch.Tensor] = None,
        R1_override: Optional[torch.Tensor] = None,
        R2_override: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:

        B, k = C.shape[0], C.shape[1]
        device = C.device



        ref_size = C.shape[2:]
        if N_vec.shape[2:] != ref_size:
            N_vec = F.interpolate(N_vec, size=ref_size,
                                  mode="bilinear", align_corners=False)
        n_map = roughness.clamp(0.5, 200.0)
        if n_map.shape[2:] != ref_size:
            n_map = F.interpolate(n_map, size=ref_size,
                                  mode="bilinear", align_corners=False)
        if L_S_P.shape[2:] != ref_size:
            L_S_P = F.interpolate(L_S_P, size=ref_size,
                                  mode="bilinear", align_corners=False)

        C_dim, H, W = self.n_channels, ref_size[0], ref_size[1]


        Camera_vec = self.compute_camera_vectors(B, H, W, focal_length)


        if C_specular is not None and C_specular.shape[2:] != ref_size:
            C_specular = F.interpolate(
                C_specular, size=ref_size, mode="bilinear", align_corners=False
            )
        specular_coefficients = C if C_specular is None else C_specular
        R1 = F.conv2d(specular_coefficients, P1, padding=0).clamp(0, 1)
        R2 = F.conv2d(C, P2, padding=0).clamp(0, 1)
        if R1_override is not None:
            R1 = F.interpolate(
                R1_override, size=ref_size, mode="bilinear", align_corners=False
            ).clamp(0, 1)
        if R2_override is not None:
            R2 = F.interpolate(
                R2_override, size=ref_size, mode="bilinear", align_corners=False
            ).clamp(0, 1)


        if self.dielectric:
            R1_gray = R1.mean(dim=1, keepdim=True)
            R1 = R1_gray.expand(-1, C_dim, -1, -1)
        if self.single_lambertian:
            R1 = torch.zeros_like(R1)


        d = light_params.shape[1]
        n_ambient_channels = C_dim

        if self.spherical_light:

            LS_dir, LS_intensity = self.parse_directional_light(
                light_params[:, :3, :, :]
            )
            LY_diffuse = self.parse_ambient_light(
                light_params[:, 3:3 + n_ambient_channels, :, :]
            )
            LY_specular = self.parse_ambient_light(
                light_params[:, 3 + n_ambient_channels:, :, :]
            )
        else:

            LS_dir = light_params[:, :3, :, :]
            LS_dir = LS_dir / (LS_dir.norm(p=2, dim=1, keepdim=True) + 1e-8)
            LS_intensity = light_params[:, 3:4, :, :].sigmoid() * math.pi * 2
            LY_diffuse = light_params[:, 4:4 + n_ambient_channels, :, :].sigmoid() / (4 * math.pi)
            LY_specular = LY_diffuse




        LS_CosT = (LS_dir * N_vec).sum(dim=1, keepdim=True).clamp(min=0.0)


        LS_R = LS_dir - (LS_dir * N_vec).sum(dim=1, keepdim=True) * 2 * N_vec
        LS_F_x_val = LS_R.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8)
        R_C_CosT = ((LS_R / LS_F_x_val) * Camera_vec).sum(dim=1, keepdim=True).clamp(min=0.0)

        log_val = (R_C_CosT + 1e-5).log().clamp(min=-30.0, max=30.0)
        R_C_CosT_n = (log_val * n_map.clamp(max=200.0)).exp().clamp(max=1e10)


        H_x = self.compute_Hx_wrapper(R_C_CosT, n_map)


        L_S_P_clamped = L_S_P.clamp(0, 1)
        LS_CosT_final = L_S_P_clamped * LS_CosT * LS_intensity
        LS_F_x_final = L_S_P_clamped * R_C_CosT_n * LS_F_x_val * LS_intensity


        diffuse_term = R2 * (LS_CosT_final + LY_diffuse * math.pi)
        specular_term = R1 * (LS_F_x_final + LY_specular * H_x)
        X_final = diffuse_term + specular_term


        alpha = (P2 / (P1 + 1e-5)).mean()
        diffuse_transport = LS_CosT_final + LY_diffuse * math.pi
        Shading_diffuse = diffuse_transport if self.single_lambertian else alpha * diffuse_transport
        Shading_specular = (
            torch.zeros_like(LS_F_x_final)
            if self.single_lambertian
            else LS_F_x_final + LY_specular * H_x
        )
        Shading = Shading_diffuse + Shading_specular

        return {
            "X_final":      X_final,
            "R1":           R1,
            "R2":           R2,
            "Shading":      Shading,
            "H_x":          H_x,
            "R_C_CosT_n":   R_C_CosT_n,
            "LS_CosT":      LS_CosT,
            "n_map":        n_map,
        }


    def compute_camera_vectors(
        self, B: int, H: int, W: int, focal_length: torch.Tensor
    ) -> torch.Tensor:

        y_range = torch.arange(H, dtype=torch.float32, device=focal_length.device)
        x_range = torch.arange(W, dtype=torch.float32, device=focal_length.device)
        y, x = torch.meshgrid(y_range, x_range, indexing="ij")

        f = focal_length.clamp(min=0.1)
        x_norm = (x - W / 2) / (f * W / 2)
        y_norm = (y - H / 2) / (f * H / 2)


        x_norm = x_norm.view(B, 1, H, W)
        y_norm = y_norm.view(B, 1, H, W)
        ones = torch.ones_like(x_norm)

        direction = torch.cat([x_norm, y_norm, ones], dim=1)
        return direction / direction.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8)


    def parse_directional_light(
        self, params: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        theta = params[:, 0:1, :, :] * math.pi
        phi   = params[:, 1:2, :, :] * 2 * math.pi
        intensity = params[:, 2:3, :, :].sigmoid() * math.pi * 2

        d_x = theta.sin() * phi.cos()
        d_y = theta.sin() * phi.sin()
        d_z = theta.cos()
        return torch.cat([d_x, d_y, d_z], dim=1), intensity


    def parse_ambient_light(self, params: torch.Tensor) -> torch.Tensor:

        return params.sigmoid() / (4 * math.pi)


    def compute_Hx_wrapper(
        self, R_C_CosT: torch.Tensor, n_map: torch.Tensor
    ) -> torch.Tensor:

        B, _, H, W = R_C_CosT.shape
        flat = torch.cat([R_C_CosT, n_map], dim=1)
        flat = flat.permute(0, 2, 3, 1).reshape(-1, 2)
        hx_flat = self.Hx_func(flat)
        return hx_flat.view(B, H, W, 1).permute(0, 3, 1, 2)
