

import csv
import time
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any






_STEP_HEADER = [
    "timestamp", "epoch", "phase", "batch", "global_step",
    "has_gt_refl", "has_gt_shade", "has_judgements",
    "lr", "temperature",
    "loss_total",
    "loss_mrae", "loss_mrae_refl",
    "loss_entropy", "loss_normal", "loss_center",
    "loss_rank", "loss_shading_reg", "loss_shading_gt", "loss_cycle",
    "grad_norm_pre_clip", "grad_norm_post_clip",
    "C_entropy_mean", "C_max_mean",
    "Shading_mean", "Shading_std",
    "R1_mean", "R2_mean", "n_map_mean",
    "P1_norm", "P2_norm",
]


_EPOCH_HEADER = [
    "epoch", "phase", "n_batches", "duration_sec",
    "lr", "temperature", "val_whdr", "val_composite",
    "val_cgintrinsics_mse", "val_hypersim_refl_mse", "val_saw_ap",
    "loss_total",
    "loss_mrae", "loss_mrae_refl",
    "loss_entropy", "loss_normal", "loss_center",
    "loss_rank", "loss_shading_reg", "loss_shading_gt", "loss_cycle",
    "grad_norm_mean",
]


_DIAG_HEADER = [
    "timestamp", "epoch", "phase", "global_step",
    "C_entropy_mean", "C_entropy_std", "C_max_mean", "C_max_std",
    "Shading_mean", "Shading_std", "Shading_min", "Shading_max",
    "R1_mean", "R1_std", "R2_mean", "R2_std",
    "n_map_mean", "n_map_std",
    "P1_norm", "P2_norm", "P1_grad_norm", "P2_grad_norm",
    "N_vec_mean", "N_vec_std",
    "N_vec_x_mean", "N_vec_z_mean",
    "H_x_mean", "focal_length",
]






class TrainingLogger:


    def __init__(self, base_dir: str = "./log/", run_id: Optional[str] = None):
        base = Path(base_dir)

        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = base / self.run_id
        self.log_dir.mkdir(parents=True, exist_ok=True)


        self._step_path = self.log_dir / "step.csv"
        self._step_fh = open(str(self._step_path), "w", newline="", encoding="utf-8")
        self._step_writer = csv.writer(self._step_fh)
        self._step_writer.writerow(_STEP_HEADER)
        self._step_fh.flush()


        self._epoch_path = self.log_dir / "epoch_summary.csv"
        self._epoch_fh = open(str(self._epoch_path), "w", newline="", encoding="utf-8")
        self._epoch_writer = csv.writer(self._epoch_fh)
        self._epoch_writer.writerow(_EPOCH_HEADER)
        self._epoch_fh.flush()


        self._diag_path = self.log_dir / "diagnostics.csv"
        self._diag_fh = open(str(self._diag_path), "w", newline="", encoding="utf-8")
        self._diag_writer = csv.writer(self._diag_fh)
        self._diag_writer.writerow(_DIAG_HEADER)
        self._diag_fh.flush()

        self._current_phase = 1

        print(f"  [Log] {self.log_dir}")



    def set_phase(self, phase_id: int):

        self._current_phase = phase_id

    def close(self):
        for fh in [self._step_fh, self._epoch_fh, self._diag_fh]:
            if fh is not None:
                fh.close()



    def log_step(
        self,
        epoch: int,
        batch_idx: int,
        global_step: int,
        loss_dict: Dict[str, Any],
        lr: float,
        temperature: float,
        batch: dict,
        outputs: Optional[Dict[str, torch.Tensor]] = None,
        grad_norm_pre: Optional[float] = None,
        grad_norm_post: Optional[float] = None,
    ):

        comps = loss_dict.get("components", {})

        def _c(key: str, default: float = 0.0) -> float:
            v = comps.get(key, default)
            if isinstance(v, torch.Tensor):
                return float(v.detach().cpu())
            return float(v) if v is not None else default

        has_refl = 1 if batch.get("reflectance") is not None else 0
        has_shade = 1 if batch.get("shading") is not None else 0
        has_judge = 1 if batch.get("judgements") and any(
            js for js in batch["judgements"] if js
        ) else 0

        C_entropy = _quick_entropy(outputs.get("C")) if outputs is not None else -1.0
        C_max = _quick_max_mean(outputs.get("C")) if outputs is not None else -1.0
        Shading_mean_val = _safe_mean(outputs.get("Shading")) if outputs is not None else -1.0
        Shading_std_val = _safe_std(outputs.get("Shading")) if outputs is not None else -1.0
        R1_mean_val = _safe_mean(outputs.get("R1")) if outputs is not None else -1.0
        R2_mean_val = _safe_mean(outputs.get("R2")) if outputs is not None else -1.0
        n_map_mean_val = _safe_mean(outputs.get("n_map")) if outputs is not None else -1.0
        P1_norm_val = _safe_norm(outputs.get("P1")) if outputs is not None else -1.0
        P2_norm_val = _safe_norm(outputs.get("P2")) if outputs is not None else -1.0

        row = [
            datetime.now().isoformat(timespec="seconds"),
            epoch, self._current_phase, batch_idx, global_step,
            has_refl, has_shade, has_judge,
            f"{lr:.6e}", f"{temperature:.4f}",
            f"{loss_dict.get('loss', 0):.6f}",
            f"{_c('mrae'):.6f}", f"{_c('mrae_refl'):.6f}",
            f"{_c('entropy'):.6f}", f"{_c('normal'):.6f}", f"{_c('center'):.6f}",
            f"{_c('rank'):.6f}", f"{_c('shading_reg'):.6f}", f"{_c('shading_gt'):.6f}",
            f"{_c('cycle'):.6f}",
            f"{grad_norm_pre:.4f}" if grad_norm_pre is not None else "",
            f"{grad_norm_post:.4f}" if grad_norm_post is not None else "",
            f"{C_entropy:.4f}", f"{C_max:.4f}",
            f"{Shading_mean_val:.4f}", f"{Shading_std_val:.4f}",
            f"{R1_mean_val:.4f}", f"{R2_mean_val:.4f}", f"{n_map_mean_val:.2f}",
            f"{P1_norm_val:.4f}", f"{P2_norm_val:.4f}",
        ]
        self._step_writer.writerow(row)
        self._step_fh.flush()



    def log_epoch(
        self,
        epoch: int,
        epoch_losses: Dict[str, float],
        val_whdr: float,
        lr: float,
        temperature: float,
        duration_sec: float,
        n_batches: int,
        grad_norm_mean: Optional[float] = None,
        val_composite: float = float("nan"),
        validation_metrics: Optional[Dict[str, float]] = None,
    ):


        validation_metrics = validation_metrics or {}

        def _metric(key: str) -> str:
            value = validation_metrics.get(key)
            return f"{value:.6f}" if isinstance(value, (int, float)) and value == value else ""

        def _v(key: str) -> str:
            val = epoch_losses.get(key, "")
            if isinstance(val, (int, float)):
                return f"{val:.6f}"
            return str(val)

        self._epoch_writer.writerow([
            epoch, self._current_phase, n_batches, f"{duration_sec:.1f}",
            f"{lr:.6e}", f"{temperature:.4f}",
            f"{val_whdr:.4f}" if val_whdr == val_whdr else "",
            f"{val_composite:.6f}" if val_composite == val_composite else "",
            _metric("cgintrinsics_mse"),
            _metric("hypersim_refl_mse"),
            (f"{1.0 - validation_metrics['saw_error']:.6f}"
             if isinstance(validation_metrics.get("saw_error"), (int, float))
             and validation_metrics["saw_error"] == validation_metrics["saw_error"] else ""),
            _v("total"),
            _v("mrae"), _v("mrae_refl"),
            _v("entropy"), _v("normal"), _v("center"),
            _v("rank"), _v("shading_reg"), _v("shading_gt"), _v("cycle"),
            f"{grad_norm_mean:.4f}" if grad_norm_mean is not None else "",
        ])
        self._epoch_fh.flush()



    def log_diagnostics(
        self,
        epoch: int,
        global_step: int,
        outputs: Dict[str, torch.Tensor],
        P1_grad_norm: Optional[float] = None,
        P2_grad_norm: Optional[float] = None,
    ):

        C = outputs.get("C")
        Shading = outputs.get("Shading")
        R1 = outputs.get("R1")
        R2 = outputs.get("R2")
        n_map = outputs.get("n_map")
        N_vec_full = outputs.get("N_vec")
        H_x = outputs.get("H_x")
        focal = outputs.get("focal_length")

        C_ent = _quick_entropy(C, per_pixel=False)
        C_ent_std = _quick_entropy_std(C)
        C_max_m, C_max_s = _quick_max_stats(C)

        def _stats(t):
            if t is None:
                return -1, -1, -1, -1
            return _safe_mean(t), _safe_std(t), _safe_min(t), _safe_max(t)

        s_mean, s_std, s_min, s_max = _stats(Shading)
        r1_mean, r1_std, _, _ = _stats(R1)
        r2_mean, r2_std, _, _ = _stats(R2)
        nm_mean, nm_std, _, _ = _stats(n_map)
        nv_mean, nv_std, _, _ = _stats(N_vec_full)

        nx_mean = _safe_mean(N_vec_full[:, 0:1, :, :]) if N_vec_full is not None else -1
        nz_mean = _safe_mean(N_vec_full[:, 2:3, :, :]) if N_vec_full is not None else -1
        hx_mean = _safe_mean(H_x)
        focal_val = float(focal.detach().cpu().flatten()[0]) if focal is not None else -1

        P1 = outputs.get("P1")
        P2 = outputs.get("P2")

        self._diag_writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            epoch, self._current_phase, global_step,
            f"{C_ent:.4f}", f"{C_ent_std:.4f}",
            f"{C_max_m:.4f}", f"{C_max_s:.4f}",
            f"{s_mean:.4f}", f"{s_std:.4f}", f"{s_min:.4f}", f"{s_max:.4f}",
            f"{r1_mean:.4f}", f"{r1_std:.4f}", f"{r2_mean:.4f}", f"{r2_std:.4f}",
            f"{nm_mean:.2f}", f"{nm_std:.2f}",
            f"{_safe_norm(P1):.4f}", f"{_safe_norm(P2):.4f}",
            f"{P1_grad_norm:.4f}" if P1_grad_norm is not None else "",
            f"{P2_grad_norm:.4f}" if P2_grad_norm is not None else "",
            f"{nv_mean:.4f}", f"{nv_std:.4f}",
            f"{nx_mean:.4f}", f"{nz_mean:.4f}",
            f"{hx_mean:.4f}", f"{focal_val:.4f}",
        ])
        self._diag_fh.flush()






def _safe_mean(t: Optional[torch.Tensor]) -> float:
    if t is None:
        return -1.0
    return float(t.detach().mean().cpu())


def _safe_std(t: Optional[torch.Tensor]) -> float:
    if t is None:
        return -1.0
    return float(t.detach().std().cpu())


def _safe_min(t: Optional[torch.Tensor]) -> float:
    if t is None:
        return -1.0
    return float(t.detach().min().cpu())


def _safe_max(t: Optional[torch.Tensor]) -> float:
    if t is None:
        return -1.0
    return float(t.detach().max().cpu())


def _safe_norm(t: Optional[torch.Tensor]) -> float:
    if t is None:
        return -1.0
    return float(t.detach().norm(p=2).cpu())


def _quick_entropy(C: Optional[torch.Tensor], per_pixel: bool = True) -> float:

    if C is None:
        return -1.0
    eps = 1e-8
    ent = -(C.detach() * (C.detach() + eps).log()).sum(dim=1)
    if per_pixel:
        return float(ent.mean().cpu())
    return float(ent.detach().mean().cpu())


def _quick_entropy_std(C: Optional[torch.Tensor]) -> float:
    if C is None:
        return -1.0
    eps = 1e-8
    ent = -(C.detach() * (C.detach() + eps).log()).sum(dim=1)
    return float(ent.std().cpu())


def _quick_max_mean(C: Optional[torch.Tensor]) -> float:
    if C is None:
        return -1.0
    return float(C.detach().max(dim=1).values.mean().cpu())


def _quick_max_stats(C: Optional[torch.Tensor]) -> tuple:
    if C is None:
        return -1.0, -1.0
    mx = C.detach().max(dim=1).values
    return float(mx.mean().cpu()), float(mx.std().cpu())
