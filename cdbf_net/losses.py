

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple

from .reflectance_source import select_reflectance_output




class LossMRAE(nn.Module):

    def __init__(self, eps: float = 1e-2):
        super().__init__()
        self.eps = eps

    def forward(self, output: torch.Tensor, label: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        error = (output - label).abs().div(label.abs() + self.eps)
        if mask is not None:
            if mask.shape[1] != output.shape[1]:
                mask = mask.expand(-1, output.shape[1], -1, -1)
            error = error * mask
            return error.sum() / (mask.sum() + self.eps)
        return error.mean()


class LossRMSE(nn.Module):

    def forward(self, output: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        return (output - label).pow(2).mean().sqrt()


class LossPSNR(nn.Module):

    def forward(self, output: torch.Tensor, label: torch.Tensor,
                data_range: float = 1.0) -> torch.Tensor:
        mse = (output - label).pow(2).mean()
        return 10.0 * torch.log10(data_range ** 2 / (mse + 1e-8))






class VGGPerceptualLoss(nn.Module):


    _VGG_LAYER_MAP = {
        "relu1_1": "4",  "relu1_2": "7",
        "relu2_1": "12", "relu2_2": "16",
        "relu3_1": "21", "relu3_2": "25", "relu3_3": "28",
        "relu4_1": "32", "relu4_2": "35",
    }

    def __init__(
        self,
        layers: Optional[List[str]] = None,
        layer_weights: Optional[List[float]] = None,
        input_channels: int = 3,
    ):
        super().__init__()
        from torchvision.models import vgg16, VGG16_Weights

        self.layers = layers or ["relu1_2", "relu2_2", "relu3_3"]
        self.layer_weights = layer_weights or [1.0, 0.5, 0.25]

        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features

        max_idx = max(int(self._VGG_LAYER_MAP[l]) for l in self.layers)
        self.slices = nn.ModuleList()
        prev_idx = 0
        for l in self.layers:
            idx = int(self._VGG_LAYER_MAP[l])
            self.slices.append(vgg[prev_idx:idx + 1])
            prev_idx = idx + 1


        if input_channels != 3:
            self.input_adapter = nn.Conv2d(input_channels, 3, 1, bias=False)
        else:
            self.input_adapter = nn.Identity()


        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))


        for p in self.parameters():
            p.requires_grad = False

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:


        pred = self.input_adapter(pred)
        target = self.input_adapter(target)
        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std

        loss = torch.tensor(0.0, device=pred.device)
        x_pred, x_target = pred, target
        for i, slc in enumerate(self.slices):
            x_pred = slc(x_pred)
            x_target = slc(x_target)
            loss = loss + self.layer_weights[i] * (x_pred - x_target).abs().mean()

        return loss






def entropy_regularization(
    C: torch.Tensor,
    min_entropy: float = 0.6,
    eps: float = 1e-8,
) -> torch.Tensor:

    per_pixel_entropy = -(C * (C.clamp(min=eps) + eps).log()).sum(dim=1)

    return F.softplus(min_entropy - per_pixel_entropy).mean()


def get_temperature(
    epoch: int, total_epochs: int, start: float = 1.0, end: float = 0.1
) -> float:

    if total_epochs <= 0:
        return end
    ratio = min(1.0, epoch / total_epochs)
    return start * (end / start) ** ratio






def prepare_saw_labels(labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

    values = labels.float()
    max_value = float(values.detach().max()) if values.numel() else 0.0
    integer_encoded = bool(
        ((values.detach() - values.detach().round()).abs() < 1e-4).all()
    )
    if not integer_encoded and max_value <= 1.1:
        if max_value < 0.5:
            is_div255 = max_value < 0.012 or max_value > 0.1
            values = values * (255.0 if is_div255 else 100.0)
        else:
            values = values * 100.0
    classes = values.round()
    valid = (classes <= 2.0).to(values.dtype)
    non_smooth = (((classes == 0.0) | (classes == 1.0)).to(values.dtype)
                  * valid)
    return non_smooth, valid


def normal_smoothness_weighted(
    N: torch.Tensor,
    image: torch.Tensor,
    sigma: float = 0.1,
    edge_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:


    img_dx = image[:, :, :, 1:] - image[:, :, :, :-1]
    img_dy = image[:, :, 1:, :] - image[:, :, :-1, :]
    img_mag_x = img_dx[:, :, :-1, :].pow(2).mean(dim=1, keepdim=True)
    img_mag_y = img_dy[:, :, :, :-1].pow(2).mean(dim=1, keepdim=True)
    img_mag = img_mag_x + img_mag_y
    weight = torch.exp(-img_mag / (2 * sigma ** 2))


    if edge_mask is not None:
        edge_mask, valid_mask = prepare_saw_labels(edge_mask)

        if edge_mask.shape[2:] != weight.shape[2:]:
            edge_mask = F.interpolate(edge_mask, size=weight.shape[2:],
                                       mode="nearest")
            valid_mask = F.interpolate(valid_mask, size=weight.shape[2:],
                                       mode="nearest")
        edge_weight = (1.0 - edge_mask.clamp(0, 1)) * valid_mask
        weight = weight * edge_weight


    n_dx = N[:, :, :, 1:] - N[:, :, :, :-1]
    n_dy = N[:, :, 1:, :] - N[:, :, :-1, :]
    loss_x = (weight * n_dx[:, :, :-1, :].pow(2).sum(dim=1)).mean()
    loss_y = (weight * n_dy[:, :, :, :-1].pow(2).sum(dim=1)).mean()
    return loss_x + loss_y






def center_loss_stable(
    P1: torch.Tensor,
    C: torch.Tensor,
    R1: torch.Tensor,
    min_support: float = 0.01,
) -> torch.Tensor:

    B, k, H, W = C.shape
    C_dim = P1.shape[0]
    min_mass = min_support * H * W

    C_prime = C.unsqueeze(1)
    R_prime = R1.unsqueeze(2)


    numerator = (C_prime * R_prime.detach()).sum(dim=(3, 4))
    denominator = C_prime.sum(dim=(3, 4))

    valid_mask = (denominator > min_mass).float()
    denominator = denominator.clamp(min=min_mass)
    P_prime = numerator / denominator

    P1_sq = P1.squeeze(-1).squeeze(-1)
    diff = (P1_sq.unsqueeze(0) - P_prime) * valid_mask
    return diff.abs().sum() / (valid_mask.sum() + 1e-8)






def ranking_loss_reflectance(
    R: torch.Tensor,
    judgements: List[List[dict]],
    H: int, W: int,
    delta: float = 0.10,
    delta_equal: float = 0.05,
    margin_eq: Optional[float] = None,
    margin_ineq: Optional[float] = None,
    ineq_weight: float = 1.0,
    ratio_delta: Optional[float] = None,
    loss_space: str = "absolute",
    hard_pair_weight: float = 0.0,
    boundary_weight: float = 0.0,
    max_pair_weight: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:

    if not judgements:
        return torch.tensor(0.0, device=R.device)

    margin_eq = delta_equal if margin_eq is None else margin_eq
    margin_ineq = delta if margin_ineq is None else margin_ineq
    ratio_delta = delta if ratio_delta is None else ratio_delta
    loss_space = loss_space.lower()
    if loss_space not in {"absolute", "log_ratio"}:
        raise ValueError("loss_space must be 'absolute' or 'log_ratio'")
    log_margin = torch.log(
        torch.tensor(1.0 + ratio_delta, device=R.device, dtype=R.dtype)
    )
    max_pair_weight = max(1.0, float(max_pair_weight))

    eq_loss = torch.tensor(0.0, device=R.device)
    ineq_loss = torch.tensor(0.0, device=R.device)
    eq_weight = torch.tensor(0.0, device=R.device)
    ineq_weight_sum = torch.tensor(0.0, device=R.device)

    B = R.shape[0]
    for b in range(B):
        if b >= len(judgements):
            break
        per_image_judgements = judgements[b]
        if not per_image_judgements:
            continue

        for j in per_image_judgements:
            w = j.get("weight", 0.0)
            if w <= 0:
                continue

            p1, p2 = j["point1"], j["point2"]
            y1 = min(max(int(p1["y"] * H), 0), H - 1)
            x1 = min(max(int(p1["x"] * W), 0), W - 1)
            y2 = min(max(int(p2["y"] * H), 0), H - 1)
            x2 = min(max(int(p2["x"] * W), 0), W - 1)
            r1_val = R[b, :, y1, x1].mean()
            r2_val = R[b, :, y2, x2].mean()

            darker = j.get("darker", "E")
            pair_loss = None
            boundary_distance = None

            if loss_space == "log_ratio":
                log_r1 = torch.log(r1_val.clamp(min=eps))
                log_r2 = torch.log(r2_val.clamp(min=eps))
                log_diff = log_r2 - log_r1
                if darker == "E":
                    pair_loss = F.relu(log_diff.abs() - log_margin)
                    boundary_distance = (log_diff.abs() - log_margin).abs()
                    pair_weight = _rank_pair_multiplier(
                        pair_loss, boundary_distance, log_margin,
                        hard_pair_weight, boundary_weight, max_pair_weight)
                    eq_loss = eq_loss + w * pair_weight * pair_loss
                    eq_weight = eq_weight + w
                elif darker == "1":
                    pair_loss = F.relu(log_margin - log_diff)
                    boundary_distance = (log_diff - log_margin).abs()
                    pair_weight = _rank_pair_multiplier(
                        pair_loss, boundary_distance, log_margin,
                        hard_pair_weight, boundary_weight, max_pair_weight)
                    ineq_loss = ineq_loss + w * pair_weight * pair_loss
                    ineq_weight_sum = ineq_weight_sum + w
                elif darker == "2":
                    pair_loss = F.relu(log_margin + log_diff)
                    boundary_distance = (-log_diff - log_margin).abs()
                    pair_weight = _rank_pair_multiplier(
                        pair_loss, boundary_distance, log_margin,
                        hard_pair_weight, boundary_weight, max_pair_weight)
                    ineq_loss = ineq_loss + w * pair_weight * pair_loss
                    ineq_weight_sum = ineq_weight_sum + w
            else:
                diff = r2_val - r1_val
                if darker == "E":
                    pair_loss = F.relu(diff.abs() - margin_eq)
                    boundary_distance = (diff.abs() - margin_eq).abs()
                    pair_weight = _rank_pair_multiplier(
                        pair_loss, boundary_distance,
                        torch.as_tensor(margin_eq, device=R.device, dtype=R.dtype),
                        hard_pair_weight, boundary_weight, max_pair_weight)
                    eq_loss = eq_loss + w * pair_weight * pair_loss
                    eq_weight = eq_weight + w
                elif darker == "1":

                    pair_loss = F.relu(margin_ineq - diff)
                    boundary_distance = (diff - margin_ineq).abs()
                    pair_weight = _rank_pair_multiplier(
                        pair_loss, boundary_distance,
                        torch.as_tensor(margin_ineq, device=R.device, dtype=R.dtype),
                        hard_pair_weight, boundary_weight, max_pair_weight)
                    ineq_loss = ineq_loss + w * pair_weight * pair_loss
                    ineq_weight_sum = ineq_weight_sum + w
                elif darker == "2":
                    pair_loss = F.relu(margin_ineq + diff)
                    boundary_distance = (-diff - margin_ineq).abs()
                    pair_weight = _rank_pair_multiplier(
                        pair_loss, boundary_distance,
                        torch.as_tensor(margin_ineq, device=R.device, dtype=R.dtype),
                        hard_pair_weight, boundary_weight, max_pair_weight)
                    ineq_loss = ineq_loss + w * pair_weight * pair_loss
                    ineq_weight_sum = ineq_weight_sum + w

    has_eq = eq_weight > 0
    has_ineq = ineq_weight_sum > 0
    if not has_eq and not has_ineq:
        return torch.tensor(0.0, device=R.device)

    if has_eq:
        eq_loss = eq_loss / eq_weight.clamp(min=1e-6)
    if has_ineq:
        ineq_loss = ineq_loss / ineq_weight_sum.clamp(min=1e-6)

    if has_eq and has_ineq:
        return (eq_loss + ineq_weight * ineq_loss) / (1.0 + ineq_weight)
    return eq_loss if has_eq else ineq_loss


def _rank_pair_multiplier(
    pair_loss: torch.Tensor,
    boundary_distance: torch.Tensor,
    margin: torch.Tensor,
    hard_pair_weight: float,
    boundary_weight: float,
    max_pair_weight: float,
) -> torch.Tensor:
    multiplier = torch.ones((), device=pair_loss.device, dtype=pair_loss.dtype)
    if hard_pair_weight > 0:
        hard = (pair_loss.detach() > 0).to(pair_loss.dtype)
        multiplier = multiplier + hard_pair_weight * hard
    if boundary_weight > 0:
        scale = margin.detach().abs().clamp(min=1e-6)
        boundary = 1.0 / (1.0 + boundary_distance.detach() / scale)
        multiplier = multiplier + boundary_weight * boundary
    return multiplier.clamp(max=max_pair_weight)


def whdr_pair_classification_loss(
    score: torch.Tensor,
    judgements: List[List[dict]],
    H: int,
    W: int,
    ratio_delta: float = 0.10,
) -> torch.Tensor:

    if not judgements:
        return torch.tensor(0.0, device=score.device)

    label_map = {"1": 0, "2": 1, "E": 2}
    margin = torch.log(
        torch.tensor(1.0 + ratio_delta, device=score.device, dtype=score.dtype)
    )
    total = torch.tensor(0.0, device=score.device)
    weight_sum = torch.tensor(0.0, device=score.device)

    B = score.shape[0]
    for b in range(B):
        if b >= len(judgements):
            break
        per_image_judgements = judgements[b]
        if not per_image_judgements:
            continue

        for j in per_image_judgements:
            darker = j.get("darker", "E")
            if darker not in label_map:
                continue
            w = j.get("weight", j.get("darker_score", 0.0))
            if w <= 0:
                continue

            p1, p2 = j["point1"], j["point2"]
            y1 = min(max(int(p1["y"] * H), 0), H - 1)
            x1 = min(max(int(p1["x"] * W), 0), W - 1)
            y2 = min(max(int(p2["y"] * H), 0), H - 1)
            x2 = min(max(int(p2["x"] * W), 0), W - 1)

            s1 = score[b, :, y1, x1].mean()
            s2 = score[b, :, y2, x2].mean()
            diff = s2 - s1
            logits = torch.stack([
                diff,
                -diff,
                margin - diff.abs(),
            ]).unsqueeze(0)
            target = torch.tensor([label_map[darker]], device=score.device)
            pair_loss = F.cross_entropy(logits, target, reduction="none").squeeze(0)
            total = total + float(w) * pair_loss
            weight_sum = weight_sum + float(w)

    if weight_sum <= 0:
        return torch.tensor(0.0, device=score.device)
    return total / weight_sum.clamp(min=1e-6)






def shading_prior_loss(
    Shading: torch.Tensor,
    upper_bound: float = 1.0,
    mean_target: float = 0.8,
    var_lower: float = 0.01,
    dark_pixel_ratio: float = 0.0,
    dark_pixel_threshold: float = 0.2,
) -> torch.Tensor:


    upper_violation = F.relu(Shading - upper_bound).pow(2).mean()

    mean_shading = Shading.mean()
    mean_penalty = F.relu(mean_shading - mean_target).pow(2)

    spatial_var = Shading.var()
    var_penalty = F.relu(var_lower - spatial_var)

    dark_mask = (Shading < dark_pixel_threshold).float()
    dark_ratio = dark_mask.mean()
    dark_penalty = F.relu(dark_pixel_ratio - dark_ratio).pow(2)

    return upper_violation + mean_penalty + var_penalty + dark_penalty






def shading_simse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:

    log_pred = torch.log(pred.clamp(min=eps))
    log_target = torch.log(target.clamp(min=eps))
    diff = log_pred - log_target

    diff = diff - diff.mean(dim=(2, 3), keepdim=True)
    return (diff ** 2).mean()






def shading_smoothness_loss(
    Shading: torch.Tensor,
    image: torch.Tensor,
    sigma: float = 0.1,
) -> torch.Tensor:


    img_dx = image[:, :, :, 1:] - image[:, :, :, :-1]
    img_dy = image[:, :, 1:, :] - image[:, :, :-1, :]
    img_mag_x = img_dx[:, :, :-1, :].pow(2).mean(dim=1, keepdim=True)
    img_mag_y = img_dy[:, :, :, :-1].pow(2).mean(dim=1, keepdim=True)
    img_mag = img_mag_x + img_mag_y
    edge_weight = torch.exp(-img_mag / (2 * sigma ** 2))


    S = Shading.mean(dim=1, keepdim=True)
    s_dx = S[:, :, :, 1:] - S[:, :, :, :-1]
    s_dy = S[:, :, 1:, :] - S[:, :, :-1, :]


    loss_x = (edge_weight * s_dx[:, :, :-1, :].pow(2)).mean()
    loss_y = (edge_weight * s_dy[:, :, :, :-1].pow(2)).mean()

    return loss_x + loss_y


def saw_shading_gradient_loss(
    shading: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.1,
    boundary_window: int = 5,
    eps: float = 1e-6,
) -> torch.Tensor:

    edge_mask, valid_mask = prepare_saw_labels(labels)
    if edge_mask.shape[2:] != shading.shape[2:]:
        size = shading.shape[2:]
        edge_mask = F.interpolate(edge_mask, size=size, mode="nearest")
        valid_mask = F.interpolate(valid_mask, size=size, mode="nearest")

    log_shading = shading.mean(dim=1, keepdim=True).clamp(min=eps).log()
    dx = log_shading[:, :, :, 1:] - log_shading[:, :, :, :-1]
    dy = log_shading[:, :, 1:, :] - log_shading[:, :, :-1, :]
    grad = torch.sqrt(
        dx[:, :, :-1, :].pow(2) + dy[:, :, :, :-1].pow(2) + eps
    )

    valid_cell = (
        valid_mask[:, :, :-1, :-1]
        * valid_mask[:, :, 1:, :-1]
        * valid_mask[:, :, :-1, 1:]
        * valid_mask[:, :, 1:, 1:]
    ) > 0.5
    edge_cell = torch.maximum(
        torch.maximum(edge_mask[:, :, :-1, :-1], edge_mask[:, :, 1:, :-1]),
        torch.maximum(edge_mask[:, :, :-1, 1:], edge_mask[:, :, 1:, 1:]),
    ) > 0.5
    smooth_cell = valid_cell & ~edge_cell
    nonsmooth_cell = valid_cell & edge_cell

    losses = []
    if smooth_cell.any():
        losses.append(grad[smooth_cell].pow(2).mean())
    if nonsmooth_cell.any():
        window = max(1, int(boundary_window))
        if window % 2 == 0:
            window += 1
        local_grad = F.max_pool2d(
            grad, kernel_size=window, stride=1, padding=window // 2
        )
        losses.append(F.relu(margin - local_grad[nonsmooth_cell]).pow(2).mean())
    if not losses:
        return shading.sum() * 0.0
    return torch.stack(losses).mean()






def compute_shading_gt_loss(
    s_pred: torch.Tensor,
    gt_shade: torch.Tensor,
    mask: Optional[torch.Tensor],
    gt_cfg: dict,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    mrae_ratio = gt_cfg.get("mrae_ratio", 1.0)
    simse_ratio = gt_cfg.get("simse_ratio", 0.0)
    mrae_eps = gt_cfg.get("mrae_eps", 0.1)

    loss_mrae = LossMRAE(eps=mrae_eps)(s_pred, gt_shade, mask=mask)
    loss_simse = shading_simse_loss(s_pred, gt_shade)

    total = mrae_ratio * loss_mrae + simse_ratio * loss_simse
    return total, loss_mrae.detach(), loss_simse.detach()


def compute_shading_regularization_loss(
    Shading: torch.Tensor,
    image: torch.Tensor,
    has_gt_shading: bool,
    reg_cfg: dict,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    prior_w = reg_cfg.get("prior_weight", 1.0)
    smooth_w = reg_cfg.get("smoothness_weight", 0.0)


    upper_bound = (reg_cfg.get("upper_bound_with_gt", 5.0) if has_gt_shading
                   else reg_cfg.get("upper_bound", 1.0))
    loss_prior = shading_prior_loss(
        Shading,
        upper_bound=upper_bound,
        mean_target=reg_cfg.get("mean_target", 0.8),
        var_lower=reg_cfg.get("var_lower", 0.01),
        dark_pixel_ratio=reg_cfg.get("dark_pixel_ratio", 0.0),
        dark_pixel_threshold=reg_cfg.get("dark_pixel_threshold", 0.2),
    )


    loss_smooth = torch.tensor(0.0, device=Shading.device)
    if smooth_w > 0:
        loss_smooth = shading_smoothness_loss(
            Shading, image,
            sigma=reg_cfg.get("smoothness_sigma", 0.1),
        )

    total = prior_w * loss_prior + smooth_w * loss_smooth
    return total, loss_prior.detach(), loss_smooth.detach()






def cycle_consistency_loss(
    C: torch.Tensor, C_recon: torch.Tensor,
    N: torch.Tensor, N_recon: torch.Tensor,
) -> torch.Tensor:

    return ((C - C_recon).pow(2).mean() +
            (N - N_recon).pow(2).mean())







_PHASE_LOSSES = {
    1: ["mrae"],
    2: ["mrae", "entropy", "normal", "shading_regularization"],
    3: ["mrae", "entropy", "normal", "shading_regularization", "center"],
    4: ["mrae", "entropy", "normal", "shading_regularization", "center", "rank", "whdr_ce", "cycle"],
    5: ["mrae", "entropy", "normal", "shading_regularization", "center", "rank", "whdr_ce", "cycle"],
}


def _reflectance_source_for_loss(config: dict, section: str) -> str:
    section_cfg = config.get(section, {})
    return section_cfg.get("reflectance_source", config.get("reflectance_source", "total"))


def compute_total_loss(
    outputs: Dict[str, torch.Tensor],
    batch: dict,
    phase: int,
    loss_weights: dict,
    config: dict,
    model: Optional[nn.Module] = None,
    active_losses: Optional[List[str]] = None,
) -> Dict[str, torch.Tensor]:

    if active_losses is not None:
        active = active_losses
    else:
        active = _PHASE_LOSSES.get(phase, _PHASE_LOSSES[1])
    components = {}
    total = torch.tensor(0.0, device=outputs["X_recon"].device)


    if "mrae" in active:
        gt_mask = batch.get("mask", None)
        loss_mrae = LossMRAE()(outputs["X_recon"], batch["image"], mask=gt_mask)
        components["mrae"] = loss_mrae.detach()
        total = total + loss_mrae * loss_weights.get("mrae", 1.0)


        gt_refl = batch.get("reflectance_gt", batch.get("reflectance"))
        if gt_refl is not None:
            R_pred = select_reflectance_output(
                outputs, _reflectance_source_for_loss(config, "reflectance_loss"))
            loss_refl = LossMRAE()(R_pred, gt_refl, mask=gt_mask)
            components["mrae_refl"] = loss_refl.detach()
            total = total + loss_refl * loss_weights.get("refl_mrae", 0.3)


        gt_shade = batch.get("shading_gt", batch.get("shading"))
        if gt_shade is not None:
            s_pred = outputs["Shading"]

            if s_pred.shape[1] != gt_shade.shape[1]:
                gt_shade = gt_shade.expand(-1, s_pred.shape[1], -1, -1)

            if s_pred.shape[2:] != gt_shade.shape[2:]:
                gt_shade = F.interpolate(
                    gt_shade, size=s_pred.shape[2:],
                    mode="bilinear", align_corners=False)


            sgt_cfg = config.get("shading_gt", {})
            loss_sgt, _, _ = compute_shading_gt_loss(
                s_pred, gt_shade, gt_mask, sgt_cfg)
            components["shading_gt"] = loss_sgt.detach()
            total = total + loss_sgt * loss_weights.get("shading_gt", 0.3)


    if "entropy" in active:
        ent_cfg = config.get("entropy", {})
        loss_ent = entropy_regularization(outputs["C"],
                                          min_entropy=ent_cfg.get("min_entropy", 0.6))
        components["entropy"] = loss_ent.detach()
        total = total + loss_ent * loss_weights.get("entropy", 0.01)


    if "normal" in active:
        nrm_cfg = config.get("normal_smoothness", {})
        saw_mask = batch.get("saw_mask", None)
        loss_norm = normal_smoothness_weighted(
            outputs["N_vec"], batch["image"],
            sigma=nrm_cfg.get("sigma", 0.1),
            edge_mask=saw_mask)
        components["normal"] = loss_norm.detach()
        total = total + loss_norm * loss_weights.get("normal", 0.001)

    if "saw_shading" in active and batch.get("saw_mask") is not None:
        saw_cfg = config.get("saw_shading", {})
        loss_saw = saw_shading_gradient_loss(
            outputs["Shading"],
            batch["saw_mask"],
            margin=saw_cfg.get("margin", 0.1),
            boundary_window=saw_cfg.get("boundary_window", 5),
        )
        components["saw_shading"] = loss_saw.detach()
        total = total + loss_saw * loss_weights.get("saw_shading", 0.2)


    if "center" in active:
        ctr_cfg = config.get("center_loss", {})
        loss_ctr = center_loss_stable(
            outputs["P1"], outputs["C"], outputs["R1"],
            min_support=ctr_cfg.get("min_support", 0.01))
        components["center"] = loss_ctr.detach()
        total = total + loss_ctr * loss_weights.get("center", 0.1)


    if "shading_regularization" in active:
        reg_cfg = config.get("shading_regularization", {})
        has_gt_shading = (batch.get("shading_gt") is not None or
                          batch.get("shading") is not None)
        loss_reg, _, _ = compute_shading_regularization_loss(
            outputs["Shading"], batch["image"], has_gt_shading, reg_cfg)
        components["shading_reg"] = loss_reg.detach()
        total = total + loss_reg * loss_weights.get("shading_regularization", 0.01)


    if "rank" in active and "judgements" in batch and batch["judgements"]:
        _, _, H, W = outputs["R1"].shape
        rank_cfg = config.get("rank_loss", {})
        R_rank = select_reflectance_output(
            outputs,
            rank_cfg.get("reflectance_source", config.get("reflectance_source", "total")),
        )
        loss_rank = ranking_loss_reflectance(
            R_rank,
            batch["judgements"],
            H,
            W,
            margin_eq=rank_cfg.get("margin_eq"),
            margin_ineq=rank_cfg.get("margin_ineq"),
            ineq_weight=rank_cfg.get("ineq_weight", 1.0),
            ratio_delta=rank_cfg.get("ratio_delta"),
            loss_space=rank_cfg.get("loss_space", "absolute"),
            hard_pair_weight=rank_cfg.get("hard_pair_weight", 0.0),
            boundary_weight=rank_cfg.get("boundary_weight", 0.0),
            max_pair_weight=rank_cfg.get("max_pair_weight", 1.0),
        )
        components["rank"] = loss_rank.detach()
        total = total + loss_rank * loss_weights.get("rank", 0.05)

    if "whdr_ce" in active and "judgements" in batch and batch["judgements"]:
        if "whdr_score" in outputs:
            _, _, H, W = outputs["whdr_score"].shape
            whdr_cfg = config.get("whdr_head", {})
            rank_cfg = config.get("rank_loss", {})
            loss_whdr_ce = whdr_pair_classification_loss(
                outputs["whdr_score"],
                batch["judgements"],
                H,
                W,
                ratio_delta=whdr_cfg.get(
                    "ratio_delta",
                    rank_cfg.get("ratio_delta", config.get("evaluation", {}).get("whdr_delta", 0.10)),
                ),
            )
            components["whdr_ce"] = loss_whdr_ce.detach()
            total = total + loss_whdr_ce * loss_weights.get("whdr_ce", 0.5)


    if "perceptual" in active:
        gt_refl = batch.get("reflectance_gt", batch.get("reflectance"))
        if gt_refl is not None:
            R_total = select_reflectance_output(
                outputs, _reflectance_source_for_loss(config, "reflectance_loss"))

            if R_total.shape[2:] != gt_refl.shape[2:]:
                gt_refl = F.interpolate(gt_refl, size=R_total.shape[2:],
                                        mode="bilinear", align_corners=False)
            loss_perc = _get_perceptual_loss().to(R_total.device)(R_total, gt_refl)
            components["perceptual"] = loss_perc.detach()
            total = total + loss_perc * loss_weights.get("perceptual", 0.1)


    if "cycle" in active and model is not None:


        if "C_cycle" in batch and "N_cycle" in batch:
            loss_cyc = cycle_consistency_loss(
                outputs["C"], batch["C_cycle"],
                outputs["N_vec"], batch["N_cycle"])
        else:

            with torch.no_grad():
                outputs2 = model(outputs["X_recon"].detach())
            loss_cyc = cycle_consistency_loss(
                outputs["C"], outputs2["C"],
                outputs["N_vec"], outputs2["N_vec"])
        components["cycle"] = loss_cyc.detach()
        total = total + loss_cyc * loss_weights.get("cycle", 0.05)

    components["total"] = total.detach()
    return {"loss": total, "components": components}







_perceptual_loss = None


def _get_perceptual_loss() -> VGGPerceptualLoss:
    global _perceptual_loss
    if _perceptual_loss is None:
        _perceptual_loss = VGGPerceptualLoss(
            layers=["relu1_2", "relu2_2", "relu3_3"],
            layer_weights=[1.0, 0.5, 0.25],
        )
    return _perceptual_loss

class AverageMeter:

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
