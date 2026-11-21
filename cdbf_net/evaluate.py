

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import math
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path
from .inference import inference_fast
from .dataset import create_dataloader
from .reflectance_source import select_reflectance_output


def _get_model_eval_image_size(model: nn.Module) -> Optional[int]:
    raw = model.module if isinstance(model, nn.DataParallel) else model
    return getattr(raw, "_infer_image_size", None)


def _prepare_model_input(
    model: nn.Module,
    image: torch.Tensor,
) -> Tuple[torch.Tensor, Tuple[int, int]]:

    output_size = image.shape[-2:]
    eval_size = _get_model_eval_image_size(model)
    if eval_size and output_size != (eval_size, eval_size):
        image = F.interpolate(
            image,
            size=(eval_size, eval_size),
            mode="bilinear",
            align_corners=False,
        )
    return image, output_size


def _select_reflectance_output(
    outputs: Dict[str, torch.Tensor],
    config: Optional[dict] = None,
    dataset_name: str = "default",
) -> torch.Tensor:
    sources = (config or {}).get("reflectance_sources", {})
    source = sources.get(dataset_name, sources.get("default", "R2"))
    return select_reflectance_output(outputs, source)






def compute_whdr(
    reflectance: np.ndarray,
    judgements,
    delta: float = 0.10,
) -> Tuple[float, float]:

    if not judgements:
        return 0.0, 0.0


    if isinstance(judgements, dict):
        points = {p["id"]: p for p in judgements.get("intrinsic_points", [])}
        comparisons = judgements.get("intrinsic_comparisons", [])
    else:
        points = {}
        comparisons = judgements if isinstance(judgements, list) else []

    if not comparisons:
        return 0.0, 0.0


    reflectance = np.clip(reflectance, 0, 1)

    H, W = reflectance.shape[:2]
    error_sum = 0.0
    weight_sum = 0.0

    for c in comparisons:
        if not isinstance(c, dict):
            continue
        darker = c.get("darker", "E")

        weight = c.get("darker_score", c.get("weight", 0))
        if not weight or weight <= 0:
            continue


        if "point1" in c:
            p1_raw = c["point1"]
            p2_raw = c["point2"]

            if isinstance(p1_raw, str) and points:
                p1_raw = points.get(p1_raw, {})
                p2_raw = points.get(p2_raw, {})
        else:
            continue

        if not (p1_raw.get("opaque", True) and p2_raw.get("opaque", True)):
            continue

        r1 = max(1e-10, reflectance[int(p1_raw["y"] * H), int(p1_raw["x"] * W)].mean())
        r2 = max(1e-10, reflectance[int(p2_raw["y"] * H), int(p2_raw["x"] * W)].mean())

        if r2 / r1 > 1.0 + delta:
            alg_darker = "1"
        elif r1 / r2 > 1.0 + delta:
            alg_darker = "2"
        else:
            alg_darker = "E"

        if darker != alg_darker:
            error_sum += weight
        weight_sum += weight

    return error_sum, weight_sum


def compute_whdr_dataset(
    predictions: Dict[str, np.ndarray],
    judgements_map: Dict[str, List[dict]],
    delta: float = 0.10,
) -> float:

    total_error, total_weight = 0.0, 0.0
    for fname, refl in predictions.items():
        if fname in judgements_map:
            e, w = compute_whdr(refl, judgements_map[fname], delta)
            total_error += e
            total_weight += w
    return total_error / total_weight if total_weight > 0 else float("nan")






def evaluate_whdr_bootstrap(
    model: nn.Module,
    dataloader: DataLoader,
    n_bootstrap: int = 1000,
    seed: int = 42,
    device: str = "cuda",
    mode: str = "fast",
    config: Optional[dict] = None,
) -> Dict[str, float]:

    rng = np.random.RandomState(seed)
    model.eval()


    eval_size = _get_model_eval_image_size(model)


    all_results = []
    for batch in dataloader:
        images = batch["image"].to(device)
        judgements = batch["judgements"]
        filenames = batch.get("filename", [f"{i:04d}" for i in range(len(images))])


        for b in range(images.shape[0]):
            img = images[b:b+1]
            model_input, output_size = _prepare_model_input(model, img)
            out = inference_fast(model, model_input)
            refl_tensor = _select_reflectance_output(out, config, "iiw").clamp(0, 1)
            if refl_tensor.shape[-2:] != output_size:
                refl_tensor = F.interpolate(
                    refl_tensor, size=output_size,
                    mode="bilinear", align_corners=False,
                )
            refl = refl_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

            js = judgements[b] if isinstance(judgements, list) and len(judgements) > b \
                else judgements
            e, w = compute_whdr(refl, js if isinstance(js, list) else [])
            if w > 0:
                all_results.append({"error": e, "weight": w, "filename": filenames[b]})

    if not all_results:
        return {"whdr_mean": float("nan"), "whdr_ci_low": float("nan"),
                "whdr_ci_high": float("nan"), "whdr_std": float("nan"), "n_valid": 0}


    n = len(all_results)
    whdr_samples = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        e_sum = sum(all_results[j]["error"] for j in idx)
        w_sum = sum(all_results[j]["weight"] for j in idx)
        whdr_samples[i] = e_sum / w_sum

    return {
        "whdr_mean":    float(whdr_samples.mean()),
        "whdr_ci_low":  float(np.percentile(whdr_samples, 2.5)),
        "whdr_ci_high": float(np.percentile(whdr_samples, 97.5)),
        "whdr_std":     float(whdr_samples.std()),
        "n_valid":      n,
    }






def _compute_lmse(pred: torch.Tensor, gt: torch.Tensor,
                  window_size: int = 10) -> float:

    if pred.shape != gt.shape:
        pred = F.interpolate(pred, size=gt.shape[-2:],
                             mode="bilinear", align_corners=False)


    K = window_size
    p_wins = F.unfold(pred, kernel_size=K)
    g_wins = F.unfold(gt,   kernel_size=K)


    num = (p_wins * g_wins).sum(dim=1, keepdim=True)
    den = (p_wins * p_wins).sum(dim=1, keepdim=True) + 1e-8
    alpha = num / den

    p_scaled = p_wins * alpha
    mse_wins = (p_scaled - g_wins).pow(2).mean(dim=1)

    return mse_wins.mean().item()


def _gaussian_window(size: int, sigma: float, device: torch.device) -> torch.Tensor:

    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    g /= g.sum()
    return g


def _compute_ssim(pred: torch.Tensor, gt: torch.Tensor,
                  window_size: int = 11, sigma: float = 1.5) -> float:

    if pred.shape != gt.shape:
        pred = F.interpolate(pred, size=gt.shape[-2:],
                             mode="bilinear", align_corners=False)

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2


    g1d = _gaussian_window(window_size, sigma, pred.device)
    window = g1d[:, None] @ g1d[None, :]
    window = window.view(1, 1, window_size, window_size)

    C = pred.shape[1]
    window = window.expand(C, 1, window_size, window_size)


    pred_groups = pred.reshape(-1, 1, pred.shape[2], pred.shape[3])
    gt_groups   = gt.reshape(-1, 1, gt.shape[2], gt.shape[3])

    mu1 = F.conv2d(pred_groups, window, padding=window_size//2, groups=1)
    mu2 = F.conv2d(gt_groups,   window, padding=window_size//2, groups=1)






    B, C_orig, H, W = pred.shape
    p_flat = pred.reshape(B * C_orig, 1, H, W)
    g_flat = gt.reshape(B * C_orig, 1, H, W)
    win = window[:1, :1, :, :]

    mu1 = F.conv2d(p_flat, win, padding=window_size // 2)
    mu2 = F.conv2d(g_flat, win, padding=window_size // 2)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(p_flat * p_flat, win, padding=window_size // 2) - mu1_sq
    sigma2_sq = F.conv2d(g_flat * g_flat, win, padding=window_size // 2) - mu2_sq
    sigma12   = F.conv2d(p_flat * g_flat,   win, padding=window_size // 2) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-8)

    return ssim_map.mean().item()




@torch.no_grad()
def evaluate_reconstruction(pred: torch.Tensor, gt: torch.Tensor) -> Dict[str, float]:

    if pred.shape != gt.shape:
        pred = F.interpolate(pred, size=gt.shape[-2:],
                             mode="bilinear", align_corners=False)


    p = pred.reshape(-1)
    g = gt.reshape(-1)
    alpha = (g * p).sum() / (p * p).sum().clamp(min=1e-8)
    si_mse = (p * alpha - g).pow(2).mean().item()

    return {
        "mse":  si_mse,
        "lmse": _compute_lmse(pred, gt, window_size=10),
        "ssim": _compute_ssim(pred, gt),
    }






@torch.no_grad()
def evaluate_cgintrinsics(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cuda",
    config: Optional[dict] = None,
) -> Dict[str, float]:

    from .inference import inference_fast

    eval_size = _get_model_eval_image_size(model)
    model.eval()
    mse_vals, lmse_vals, ssim_vals = [], [], []

    for batch in dataloader:
        images = batch["image"].to(device)
        gt_refl = batch.get("reflectance")
        if gt_refl is None:
            continue
        gt_refl = gt_refl.to(device)

        for b in range(images.shape[0]):
            img = images[b:b+1]
            gt = gt_refl[b:b+1]

            model_input, _ = _prepare_model_input(model, img)
            out = inference_fast(model, model_input)
            pred = _select_reflectance_output(
                out, config, "cgintrinsics"
            ).clamp(0, 1)

            if pred.shape != gt.shape:
                pred = F.interpolate(pred, size=gt.shape[-2:],
                                     mode="bilinear", align_corners=False)

            m = evaluate_reconstruction(pred, gt)
            mse_vals.append(m["mse"])
            lmse_vals.append(m["lmse"])
            ssim_vals.append(m["ssim"])

    n = len(mse_vals)
    if n == 0:
        return {k: float("nan") for k in
                ["mse_mean","mse_std","lmse_mean","lmse_std",
                 "ssim_mean","ssim_std","dssim_mean","n_samples"]}

    ssim_mean = float(np.mean(ssim_vals))
    return {
        "mse_mean":    float(np.mean(mse_vals)),
        "mse_std":     float(np.std(mse_vals)),
        "lmse_mean":   float(np.mean(lmse_vals)),
        "lmse_std":    float(np.std(lmse_vals)),
        "ssim_mean":   ssim_mean,
        "ssim_std":    float(np.std(ssim_vals)),
        "dssim_mean":  (1.0 - ssim_mean) / 2.0,
        "n_samples":   n,
    }






@torch.no_grad()
def evaluate_hypersim(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cuda",
    config: Optional[dict] = None,
) -> Dict[str, float]:

    from .inference import inference_fast

    eval_size = _get_model_eval_image_size(model)
    model.eval()
    refl_mse, refl_lmse, refl_ssim = [], [], []
    shd_mse,  shd_lmse,  shd_ssim  = [], [], []

    for batch in dataloader:
        images = batch["image"].to(device)
        gt_refl = batch.get("reflectance")
        gt_shd = batch.get("shading")

        for b in range(images.shape[0]):
            img = images[b:b+1]
            model_input, _ = _prepare_model_input(model, img)
            out = inference_fast(model, model_input)

            if gt_refl is not None:
                gt_r = gt_refl[b:b+1].to(device)
                m = evaluate_reconstruction(
                    _select_reflectance_output(
                        out, config, "hypersim"
                    ).clamp(0, 1),
                    gt_r,
                )
                refl_mse.append(m["mse"])
                refl_lmse.append(m["lmse"])
                refl_ssim.append(m["ssim"])

            if gt_shd is not None:
                gt_s = gt_shd[b:b+1].to(device)
                m = evaluate_reconstruction(out["shading"].clamp_min(0), gt_s)
                shd_mse.append(m["mse"])
                shd_lmse.append(m["lmse"])
                shd_ssim.append(m["ssim"])

    def _agg(mse_v, lmse_v, ssim_v):
        if not mse_v:
            return {}
        mse_mean = float(np.mean(mse_v))
        return {
            "mse_mean":  mse_mean,
            "mse_std":   float(np.std(mse_v)),
            "lmse_mean": float(np.mean(lmse_v)),
            "lmse_std":  float(np.std(lmse_v)),
            "psnr":      -10.0 * math.log10(max(mse_mean, 1e-8)),
            "ssim_mean": float(np.mean(ssim_v)),
            "ssim_std":  float(np.std(ssim_v)),
        }

    refl = _agg(refl_mse, refl_lmse, refl_ssim)
    shd  = _agg(shd_mse,  shd_lmse,  shd_ssim)
    n_samples = max(len(refl_mse), len(shd_mse))

    return {
        **{f"refl_{k}": v for k, v in refl.items()},
        **{f"shd_{k}":  v for k, v in shd.items()},
        "n_samples": n_samples,
    }






def _gen_saw_thresholds(count: int = 200) -> np.ndarray:

    zero_to_one = np.linspace(1.001, 1.5, num=count // 2)
    one_to_five = np.linspace(1.501, 64.0, num=count // 2)
    return np.log(np.sort(np.concatenate([zero_to_one, one_to_five])))


def compute_saw_edge_ap(
    shading: np.ndarray,
    saw_mask_gt: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
    bl_filter_size: int = 10,
) -> float:

    from scipy.ndimage import sobel, maximum_filter

    if thresholds is None:
        thresholds = _gen_saw_thresholds()

    eps = 1e-8


    S = np.asarray(shading, dtype=np.float32)
    if S.ndim == 3:
        S = S.mean(axis=2)
    S = np.maximum(S, 1e-4)
    S_log = np.log(S)


    gy = sobel(S_log, axis=0)
    gx = sobel(S_log, axis=1)
    gradmag = np.hypot(gx, gy)
    if bl_filter_size > 0:
        gradmag = maximum_filter(gradmag, size=bl_filter_size)
    gradmag_max = gradmag

    gradmag_flat = gradmag.ravel()
    gradmag_max_flat = gradmag_max.ravel()


    gt = np.asarray(saw_mask_gt, dtype=np.float32).squeeze()
    if gt.shape != gradmag.shape:
        from skimage.transform import resize as sk_resize
        gt = sk_resize(gt, gradmag.shape, order=0, preserve_range=True,
                       mode='constant', anti_aliasing=False)
    gt_flat = gt.ravel()

    gt_unique = set(np.unique(gt_flat))
    if gt_unique.issubset({0.0, 1.0, 2.0}) and 2.0 in gt_unique:
        gt_smooth = (gt_flat == 2.0).astype(np.float32)
        gt_nonsmooth = ((gt_flat == 0.0) | (gt_flat == 1.0)).astype(np.float32)
    elif gt_unique.issubset({0.0, 1.0, 2.0}) and 2.0 not in gt_unique:
        vals = sorted(gt_unique)
        smooth_val = max(vals)
        gt_smooth = (gt_flat == smooth_val).astype(np.float32)
        gt_nonsmooth = (gt_flat != smooth_val).astype(np.float32)
    else:
        gt_nonsmooth = (gt_flat > 0.5).astype(np.float32)
        gt_smooth = 1.0 - gt_nonsmooth

    if gt_smooth.sum() == 0:
        return float("nan")


    precisions, recalls = [], []
    for thres in thresholds:
        pred_smooth = (gradmag_flat < thres).astype(np.float32)
        pred_max_smooth = (gradmag_max_flat < thres).astype(np.float32)

        tp = (pred_smooth * gt_smooth).sum()
        fp = (pred_max_smooth * gt_nonsmooth).sum()
        fn = ((1 - pred_smooth) * gt_smooth).sum()

        precisions.append(tp / (tp + fp + eps))
        recalls.append(tp / (tp + fn + eps))


    order = np.argsort(recalls)
    r_sorted = np.array(recalls)[order]
    p_sorted = np.array(precisions)[order]
    for i in range(len(p_sorted) - 2, -1, -1):
        p_sorted[i] = max(p_sorted[i], p_sorted[i + 1])
    r_ext = np.concatenate([[0.0], r_sorted, [1.0]])
    p_ext = np.concatenate([[p_sorted[0]], p_sorted, [0.5]])

    return float(np.trapz(p_ext, r_ext))


@torch.no_grad()
def _evaluate_saw_ap_preprocessed(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cuda",
    config: Optional[dict] = None,
) -> Dict[str, float]:

    from .inference import inference_fast
    from scipy.ndimage import sobel, maximum_filter

    thresholds = _gen_saw_thresholds()
    bl_filter_size = 10
    class_weights = [1, 1, 2]
    eps = 1e-5

    model.eval()
    thres_count = len(thresholds)
    overall_cm = [np.zeros((3, 2), dtype=np.int64) for _ in range(thres_count)]
    n_valid, n_total = 0, 0

    for batch in dataloader:
        images = batch["image"].to(device)
        saw_masks = batch.get("saw_mask")

        for b in range(images.shape[0]):
            n_total += 1
            img = images[b:b+1]
            model_input, output_size = _prepare_model_input(model, img)
            out = inference_fast(model, model_input)

            S = out.get("shading")
            if S is None:
                continue
            if S.shape[-2:] != output_size:
                S = F.interpolate(
                    S, size=output_size,
                    mode="bilinear", align_corners=False,
                )
            S_np = S.squeeze(0).permute(1, 2, 0).cpu().numpy()
            if S_np.ndim == 3:
                S_np = S_np.mean(axis=2)
            S_np = np.maximum(S_np, 1e-4)
            S_log = np.log(S_np)
            gy = sobel(S_log, axis=0)
            gx = sobel(S_log, axis=1)
            gradmag = np.abs(np.hypot(gx, gy))
            gradmag_max = maximum_filter(gradmag, size=bl_filter_size) if bl_filter_size > 0 else gradmag

            if saw_masks is not None:
                gt = saw_masks[b].cpu().numpy().squeeze()
            else:
                continue
            if gt.shape != gradmag.shape:
                from skimage.transform import resize as sk_resize
                gt = sk_resize(gt, gradmag.shape, order=0, preserve_range=True,
                               mode='constant', anti_aliasing=False)


            gt_flat = gt.ravel().astype(np.float64)
            gt_max = gt_flat.max()
            if gt_max > 50:

                y_true = gt_flat.round().astype(np.int64)
            elif gt_max <= 1.1:

                y_true = (gt_flat * 100.0).round().astype(np.int64)
            else:

                y_true = gt_flat.round().astype(np.int64)


            ignored = y_true > 99
            if np.all(ignored):
                continue

            valid = ~ignored
            yt = y_true[valid]
            gm_flat = gradmag.ravel()[valid]
            gm_max_flat = gradmag_max.ravel()[valid]


            if ((yt == 2).sum() == 0 or
                ((yt == 0).sum() + (yt == 1).sum()) == 0):
                continue
            n_valid += 1


            for ti, thres in enumerate(thresholds):
                y_pred = (gm_flat < thres).astype(np.int64)
                y_pred_max = (gm_max_flat < thres).astype(np.int64)

                cm = np.zeros((3, 2), dtype=np.int64)
                for gt_label in range(3):
                    mask = yt == gt_label
                    if gt_label < 2:

                        for pred_label in range(2):
                            cm[gt_label, pred_label] = int(
                                (y_pred_max[mask] == pred_label).sum())
                    else:

                        for pred_label in range(2):
                            cm[gt_label, pred_label] = int(
                                (y_pred[mask] == pred_label).sum())

                overall_cm[ti] += cm


    precisions, recalls = [], []
    for cm in overall_cm:
        if cm.sum() == 0:
            precisions.append(0.0)
            recalls.append(0.0)
            continue

        cm_w = cm.astype(np.float64).copy()
        label_counts = cm_w.sum(axis=1)
        cm_w *= (np.array(class_weights, dtype=np.float64)
                 / (label_counts + eps))[:, np.newaxis]

        smooth_count_true = cm_w[2, :].sum()
        smooth_count_pred = cm_w[:, 1].sum()
        smooth_count_correct = float(cm_w[2, 1])

        prec = smooth_count_correct / (smooth_count_pred + eps)
        rec = smooth_count_correct / (smooth_count_true + eps)
        precisions.append(prec)
        recalls.append(rec)

    order = np.argsort(recalls)
    r_sorted = np.array(recalls)[order]
    p_sorted = np.array(precisions)[order]
    for i in range(len(p_sorted) - 2, -1, -1):
        p_sorted[i] = max(p_sorted[i], p_sorted[i + 1])
    r_ext = np.concatenate([[0.0], r_sorted, [1.0]])
    p_ext = np.concatenate([[p_sorted[0]], p_sorted, [0.5]])
    ap = float(np.trapz(p_ext, r_ext)) if n_valid > 0 else float("nan")

    return {
        "ap_mean": ap, "ap_std": 0.0,
        "n_valid": n_valid, "n_total": n_total,
    }


def _resolve_saw_path(
    base_dir: Path,
    photo_id,
    extension: str,
) -> Optional[Path]:
    if isinstance(photo_id, (int, np.integer)):
        candidates = (
            base_dir / f"{int(photo_id)}{extension}",
            base_dir / f"{int(photo_id):05d}{extension}",
            base_dir / f"{int(photo_id):06d}{extension}",
        )
    else:
        candidates = (base_dir / f"{photo_id}{extension}",)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = sorted(base_dir.rglob(f"{photo_id}{extension}"))
    return matches[0] if matches else None


@torch.no_grad()
def evaluate_saw_ap(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cuda",
    config: Optional[dict] = None,
) -> Dict[str, float]:

    from PIL import Image
    from scipy.ndimage import maximum_filter, sobel
    from skimage.transform import resize as sk_resize

    del dataloader
    saw_root = Path((config or {}).get("data", {}).get("saw_path", ""))
    ids_path = saw_root / "saw_splits" / "test_ids.npy"
    image_dir = saw_root / "saw_images_512"
    labels_root = saw_root / "saw_pixel_labels"
    if not ids_path.exists():
        raise FileNotFoundError(f"SAW split file not found: {ids_path}")

    label_files = sorted(labels_root.rglob("*.npy"))
    if not label_files:
        raise FileNotFoundError(f"No SAW labels found under {labels_root}")
    label_dir = label_files[0].parent

    photo_ids = []
    for value in np.load(ids_path).flat:
        try:
            photo_ids.append(int(value))
        except (TypeError, ValueError):
            photo_ids.append(str(value).rsplit(".", 1)[0])

    thresholds = _gen_saw_thresholds()
    confusion = [np.zeros((3, 2), dtype=np.int64) for _ in thresholds]
    class_weights = np.array([1.0, 1.0, 2.0], dtype=np.float64)
    n_total = 0
    n_valid = 0
    model.eval()

    for photo_id in photo_ids:
        n_total += 1
        image_path = _resolve_saw_path(image_dir, photo_id, ".png")
        label_path = _resolve_saw_path(label_dir, photo_id, ".npy")
        if image_path is None or label_path is None:
            continue

        image_np = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32)
        image = torch.from_numpy(image_np / 255.0).permute(2, 0, 1).unsqueeze(0)
        image = image.to(device)
        model_input, output_size = _prepare_model_input(model, image)
        shading = inference_fast(model, model_input).get("shading")
        if shading is None:
            continue
        if shading.shape[-2:] != output_size:
            shading = F.interpolate(
                shading, size=output_size,
                mode="bilinear", align_corners=False,
            )

        shading_np = shading.clamp_min(0).squeeze(0).permute(1, 2, 0).cpu().numpy()
        shading_gray = shading_np.mean(axis=2) if shading_np.ndim == 3 else shading_np
        log_shading = np.log(np.maximum(shading_gray, 1e-4))
        gradient = np.abs(
            np.hypot(sobel(log_shading, axis=1), sobel(log_shading, axis=0))
        )
        gradient_max = maximum_filter(gradient, size=10)

        labels = np.load(label_path)
        if labels.shape != gradient.shape:
            labels = sk_resize(
                labels.astype(np.float32), gradient.shape,
                order=0, preserve_range=True, mode="constant",
                anti_aliasing=False,
            ).astype(labels.dtype)
        labels = labels.ravel()
        valid = labels <= 99
        if not np.any(valid):
            continue
        n_valid += 1

        target = labels[valid]
        gradient_valid = gradient.ravel()[valid]
        gradient_max_valid = gradient_max.ravel()[valid]
        for index, threshold in enumerate(thresholds):
            smooth = (gradient_valid < threshold).astype(np.int64)
            smooth_max = (gradient_max_valid < threshold).astype(np.int64)
            matrix = np.zeros((3, 2), dtype=np.int64)
            for target_class in range(3):
                class_mask = target == target_class
                prediction = smooth_max if target_class < 2 else smooth
                for predicted_class in range(2):
                    matrix[target_class, predicted_class] = int(
                        (prediction[class_mask] == predicted_class).sum()
                    )
            confusion[index] += matrix

    precision = []
    recall = []
    for matrix in confusion:
        weighted = matrix.astype(np.float64)
        counts = weighted.sum(axis=1)
        weighted *= (class_weights / (counts + 1e-5))[:, None]
        true_smooth = weighted[2, :].sum()
        predicted_smooth = weighted[:, 1].sum()
        correct_smooth = weighted[2, 1]
        precision.append(correct_smooth / (predicted_smooth + 1e-5))
        recall.append(correct_smooth / (true_smooth + 1e-5))

    order = np.argsort(recall)
    recall_sorted = np.asarray(recall)[order]
    precision_sorted = np.asarray(precision)[order]
    for index in range(len(precision_sorted) - 2, -1, -1):
        precision_sorted[index] = max(
            precision_sorted[index], precision_sorted[index + 1]
        )
    recall_curve = np.concatenate(([0.0], recall_sorted, [1.0]))
    precision_curve = np.concatenate(([precision_sorted[0]], precision_sorted, [0.5]))
    ap = float(np.trapz(precision_curve, recall_curve))
    return {
        "ap_mean": ap,
        "ap_std": 0.0,
        "n_valid": n_valid,
        "n_total": n_total,
    }


