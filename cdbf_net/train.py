

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple
from pathlib import Path
import yaml
import time
import math
from contextlib import nullcontext
from .losses import compute_total_loss, get_temperature, AverageMeter
from .dataset import create_dataloader
from .log_utils import TrainingLogger
from .reflectance_source import select_reflectance_output
from datetime import datetime


try:
    from torch.cuda.amp import autocast, GradScaler
    _AMP_AVAILABLE = True
except ImportError:
    _AMP_AVAILABLE = False
    autocast = nullcontext


def _select_whdr_reflectance(outputs: dict, config: dict) -> torch.Tensor:
    source = config.get("evaluation", {}).get("reflectance_source", "total")
    return select_reflectance_output(outputs, source)


def _scheduled_temperature(
    epoch: int,
    total_epochs: int,
    end: float,
    resume_epoch: Optional[int] = None,
    resume_temperature: Optional[float] = None,
    start: float = 1.0,
) -> float:
    if resume_epoch is not None and resume_temperature is not None and epoch > resume_epoch:
        remaining_epochs = max(1, total_epochs - resume_epoch)
        resumed_step = min(remaining_epochs, epoch - resume_epoch)
        return get_temperature(
            resumed_step,
            remaining_epochs,
            start=resume_temperature,
            end=end,
        )
    return get_temperature(epoch - 1, total_epochs, start=start, end=end)


def _scheduler_state_matches_training_plan(state: dict, total_epochs: int) -> bool:
    return int(state.get("T_max", total_epochs)) == int(total_epochs)


def _compute_checkpoint_selection_score(metrics: dict, config: dict) -> float:

    weights = config.get("weights", {})
    references = config.get("references", {})
    weighted_sum = 0.0
    weight_sum = 0.0
    for name, weight in weights.items():
        value = metrics.get(name)
        reference = references.get(name)
        if (value is None or reference is None or reference <= 0 or
                not math.isfinite(float(value)) or weight < 0):
            return float("nan")
        weighted_sum += float(weight) * float(value) / float(reference)
        weight_sum += float(weight)
    if weight_sum <= 0:
        return float("nan")
    return weighted_sum / weight_sum


class Trainer:


    def __init__(
        self,
        model: nn.Module,
        config: dict,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader


        gpu_ids = config.get("training", {}).get("gpu_ids", [0])
        self._dp = False
        self.device = torch.device(f"cuda:{gpu_ids[0]}" if gpu_ids and torch.cuda.is_available()
                                   else next(model.parameters()).device)
        model.to(self.device)
        if gpu_ids and len(gpu_ids) > 1 and torch.cuda.device_count() >= len(gpu_ids):
            self.model = nn.DataParallel(model, device_ids=gpu_ids)
            self._dp = True
            print(f"  Using DataParallel on GPUs: {gpu_ids}")
        else:
            self._dp = False

        self.current_epoch = 0
        self.current_phase = 1
        self.global_step = 0
        self.best_selection_score = float("inf")
        self.best_whdr = float("inf")
        self.best_epoch = 0
        self._active_losses = ["mrae"]
        self._selection_eval_loaders = {}


        self.use_amp = config.get("training", {}).get("use_amp", False) and _AMP_AVAILABLE
        self.scaler = GradScaler() if self.use_amp else None
        if self.use_amp:
            print("  AMP mixed precision enabled (autocast + GradScaler)")


        encoder_name = config["model"].get("encoder", "resnet50")
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{encoder_name}"


        ckpt_base = Path(config["logging"].get("checkpoint_dir", "./checkpoints/"))
        self.ckpt_dir = ckpt_base / self.run_id
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)


        log_base = Path(config["logging"].get("log_dir", "./log/"))
        self.logger = TrainingLogger(str(log_base), run_id=self.run_id)
        self.log_dir = self.logger.log_dir
        self._diag_interval = config["logging"].get("diag_interval_steps", 50)


        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(str(self.log_dir / "tensorboard"))

        self._init_dataloaders_and_opt()

    @property
    def _raw_model(self) -> nn.Module:

        return self.model.module if self._dp else self.model

    @staticmethod
    def _print_batch_progress_enabled(config: dict) -> bool:
        return bool(config.get("logging", {}).get("print_batch_progress", False))

    def _rebuild_dataloaders(self, batch_size: int):

        from .dataset import create_dataloader

        self.config["training"]["batch_size"] = batch_size
        self.train_loader = create_dataloader(self.config, split="train")
        try:
            self.val_loader = create_dataloader(self.config, split="val")
        except Exception:
            self.val_loader = None
        self._selection_eval_loaders = {}

    def _init_dataloaders_and_opt(self):

        config = self.config
        with open(config.get("loss_weights_path", "config/loss_weights.yaml"),
                  encoding="utf-8") as f:
            self.loss_weights_cfg = yaml.safe_load(f)
        self.loss_weights = self.loss_weights_cfg["loss_weights"]

        self.history: Dict[str, list] = {
            "epoch": [], "phase": [], "train_loss": [], "val_whdr": [],
            "val_composite": [], "lr": [], "temperature": [],
        }

        self.optimizer, self.scheduler = self._configure_optimizers()

        if self.train_loader is None:
            self.train_loader = create_dataloader(config, split="train")
        if self.val_loader is None:
            try:
                self.val_loader = create_dataloader(config, split="val")
            except Exception:
                self.val_loader = None





    def _configure_optimizers(self) -> Tuple[optim.Optimizer, object]:
        base_lr = self.config["training"]["base_lr"]
        total_epochs = self.config["training"]["epochs"]


        base_params = self._raw_model.base_parameters() if hasattr(self._raw_model, "base_parameters") else []
        other_params = [p for p in self._raw_model.parameters()
                       if all(id(p) != id(bp) for bp in base_params)]

        param_groups = [
            {"params": other_params, "lr": base_lr},
            {"params": base_params, "lr": base_lr * 0.1},
        ]

        optimizer = optim.AdamW(param_groups, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epochs, eta_min=5e-5)

        return optimizer, scheduler





    def train(self) -> Dict[str, list]:

        phases = self.config["phases"]
        total_epochs = self.config["training"]["epochs"]
        selection_cfg = self.config.get("checkpoint_selection", {})
        selection_enabled = bool(selection_cfg.get("enabled", False))
        selection_interval = max(1, int(selection_cfg.get("eval_interval_epochs", 5)))
        best_selection_score = self.best_selection_score
        best_whdr = self.best_whdr
        best_loss = float("inf")
        best_epoch = self.best_epoch

        for phase_name, phase_cfg in phases.items():
            phase_id = int(phase_name.replace("phase", ""))
            start_e, end_e = phase_cfg["epochs"]
            active_losses = phase_cfg["active_losses"]


            if phase_id < self.current_phase:
                print(f"\n  Phase {phase_id}: skipped (resumed at phase {self.current_phase})")
                continue

            print(f"\n{'='*60}")
            print(f"  Phase {phase_id}: Epoch {start_e}-{end_e} | Losses: {active_losses}")
            print(f"{'='*60}")

            self.current_phase = phase_id
            self._active_losses = active_losses


            self._phase_image_size = phase_cfg.get("image_size",
                                                   self.config["data"].get("image_size", 256))
            print(f"  Image size: {self._phase_image_size}")


            phase_bs = phase_cfg.get("batch_size", None)
            current_bs = self.config["training"].get("batch_size", 64)
            if phase_bs is not None and phase_bs != current_bs:
                print(f"  Batch size: {current_bs} → {phase_bs}")
                self._rebuild_dataloaders(phase_bs)


            if "lr" in phase_cfg:
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = phase_cfg["lr"]
                print(f"  Phase {phase_id}: explicit lr={phase_cfg['lr']}")


            self.logger.set_phase(phase_id)


            if self.val_loader is not None and phase_id > 1 and self.current_epoch < start_e:
                whdr_before, whdr_diag = self._validate_whdr()
                diag_tail = f"  [{whdr_diag}]" if whdr_diag else ""
                print(f"  WHDR before Phase {phase_id}: {whdr_before:.4f}{diag_tail}")


            if phase_id > 1 and self.current_epoch < start_e:
                self._reset_adamw_state()
                print(f"  Phase {phase_id}: AdamW state reset (prevent momentum avalanche)")


            first_epoch = max(start_e, self.current_epoch + 1)
            if first_epoch > start_e:
                print(f"  Phase {phase_id}: skipping epochs {start_e}–{first_epoch - 1} (resumed at epoch {self.current_epoch})")

            for epoch in range(first_epoch, end_e + 1):
                self.current_epoch = epoch
                epoch_t0 = time.time()


                temp_end = self.config["training"].get("temperature_end", 0.3)
                tau = _scheduled_temperature(
                    epoch=epoch,
                    total_epochs=total_epochs,
                    start=self.config["training"].get("temperature_start", 1.0),
                    end=temp_end,
                    resume_epoch=getattr(self, "_resume_epoch", None),
                    resume_temperature=getattr(self, "_resume_temperature", None),
                )
                if hasattr(self._raw_model, "temperature"):
                    self._raw_model.temperature = tau


                train_losses, grad_norm_avg, n_batches = self._train_epoch(
                    self._get_ramp_weights(epoch, phase_cfg), epoch)
                epoch_dur = time.time() - epoch_t0


                lr = self.optimizer.param_groups[0]["lr"]
                self.history["epoch"].append(epoch)
                self.history["phase"].append(phase_id)
                self.history["train_loss"].append(train_losses.get("total", 0))
                self.history["lr"].append(lr)
                self.history["temperature"].append(tau)


                total_loss = train_losses.get("total", 0)
                rate_str = f"{n_batches / epoch_dur:.1f}it/s" if epoch_dur > 0 else "?it/s"
                bar_filled = "█" * 25
                bar_line = (f"Epoch {epoch:3d} [{bar_filled}] 100.0%  "
                           f"{n_batches}/{n_batches}  loss={total_loss:.4f}  {rate_str}  ETA 0s")


                _LOSS_ORDER = ["mrae", "mrae_refl",
                              "entropy", "normal", "center",
                              "shading_reg", "shading_gt", "saw_shading",
                              "rank", "perceptual",
                              "cycle"]
                _DS_ORDER = ["CGIntrinsics/HyperSim", "IIW", "SAW"]
                loss_parts = " | ".join(
                    f"{k}={train_losses[k]:.4f}" for k in _LOSS_ORDER
                    if k in train_losses
                )
                ds_summary = train_losses.get("_ds_summary", {})
                ds_parts = []
                for t in _DS_ORDER:
                    if t in ds_summary:
                        avg, _ = ds_summary[t]
                        ds_parts.append(f"{t.replace('CGIntrinsics/HyperSim','CGIn')}={avg:.2f}")
                ds_str = " | ".join(ds_parts)


                val_whdr = float("nan")
                whdr_str = ""
                selection_str = ""
                selection_score = float("nan")
                validation_metrics = {}
                if epoch % self.config["logging"]["eval_interval_epochs"] == 0:
                    if self.val_loader is not None:
                        val_whdr, whdr_diag = self._validate_whdr()
                        self.history["val_whdr"].append(val_whdr)
                        whdr_valid = math.isfinite(val_whdr)
                        if not whdr_valid:
                            whdr_str = "WHDR=NaN"
                        else:
                            whdr_improved = val_whdr < best_whdr
                            whdr_str = f"WHDR={val_whdr:.4f}"
                            self.writer.add_scalar("val/whdr", val_whdr, epoch)
                            best_whdr = min(best_whdr, val_whdr)
                            self.best_whdr = best_whdr

                        train_loss = total_loss
                        should_save_best = False
                        validation_metrics = {"whdr": val_whdr}

                        if (selection_enabled and whdr_valid and
                                epoch % selection_interval == 0):
                            validation_metrics, selection_score = (
                                self._validate_checkpoint_selection(val_whdr)
                            )
                            self.history["val_composite"].append(selection_score)
                            if math.isfinite(selection_score):
                                selection_str = f"Composite={selection_score:.4f}"
                                self.writer.add_scalar(
                                    "val/composite", selection_score, epoch
                                )
                                for name, value in validation_metrics.items():
                                    self.writer.add_scalar(f"val/{name}", value, epoch)
                                if selection_score < best_selection_score:
                                    best_selection_score = selection_score
                                    best_epoch = epoch
                                    self.best_selection_score = selection_score
                                    self.best_epoch = epoch
                                    should_save_best = True
                                    selection_str += " ▼"
                            else:
                                selection_str = "Composite=NaN"
                        elif not selection_enabled:
                            if whdr_valid and whdr_improved:
                                best_epoch = epoch
                                self.best_epoch = epoch
                                should_save_best = True
                                whdr_str += " ▼"
                            elif not whdr_valid and train_loss < best_loss:
                                best_loss = train_loss
                                best_epoch = epoch
                                self.best_epoch = epoch
                                should_save_best = True

                        if should_save_best:
                            best_path = self.ckpt_dir / "best.pth"
                            torch.save({
                                "state_dict": self.model.state_dict(),
                                "temperature": self._raw_model.temperature if hasattr(self._raw_model, "temperature") else 0.1,
                                "image_size": getattr(self, "_phase_image_size",
                                                      self.config["data"].get("image_size", 256)),
                                "epoch": epoch,
                                "val_whdr": val_whdr,
                                "selection_score": selection_score,
                                "validation_metrics": validation_metrics,
                                "best_selection_score": best_selection_score,
                                "best_whdr": best_whdr,
                                "best_epoch": best_epoch,
                            }, best_path)


                parts = [f"lr={lr:.2e}", f"τ={tau:.3f}"]
                if loss_parts:
                    parts.append(loss_parts)
                if ds_str:
                    parts.append(ds_str)
                if whdr_str:
                    parts.append(whdr_str)
                if selection_str:
                    parts.append(selection_str)
                parts.append(f"{epoch_dur:.0f}s")
                print(f"\r  {bar_line} | {' | '.join(parts)}")


                self.writer.add_scalar("train/loss_total", train_losses.get("total", 0), epoch)
                for k, v in train_losses.items():
                    if k != "total" and not k.startswith("_"):
                        self.writer.add_scalar(f"train/{k}", v, epoch)
                self.writer.add_scalar("train/lr", lr, epoch)
                self.writer.add_scalar("train/temperature", tau, epoch)


                if epoch % self.config["logging"]["save_interval_epochs"] == 0:
                    self._save_checkpoint(epoch)


                self.logger.log_epoch(
                    epoch=epoch,
                    epoch_losses=train_losses,
                    val_whdr=val_whdr,
                    lr=lr,
                    temperature=tau,
                    duration_sec=epoch_dur,
                    n_batches=n_batches,
                    grad_norm_mean=grad_norm_avg,
                    val_composite=selection_score,
                    validation_metrics=validation_metrics,
                )


                self.scheduler.step()



        print(f"\n{'='*60}")
        if selection_enabled and best_selection_score < float("inf"):
            print(
                f"  Training complete. Best composite: {best_selection_score:.4f} "
                f"(epoch {best_epoch}); best observed WHDR: {best_whdr:.4f}"
            )
        else:
            print(f"  Training complete. Best WHDR: {best_whdr:.4f} (epoch {best_epoch})"
                  if best_whdr < float("inf") else "  Training complete.")
        print(f"{'='*60}")

        self.logger.close()
        return self.history





    @staticmethod
    def _dataset_type(batch: dict) -> str:

        has_gt = (batch.get("reflectance") is not None or
                  batch.get("reflectance_gt") is not None)
        has_judge = batch.get("judgements") and any(
            js for js in batch["judgements"] if js)
        if has_gt:
            return "CGIntrinsics/HyperSim"
        elif has_judge:
            return "IIW"
        elif batch.get("saw_mask") is not None:
            return "SAW"
        else:
            return "unknown"

    def _train_epoch(self, ramp_weights: Dict[str, float],
                     epoch: int = 0) -> Tuple[Dict[str, float], float, int]:

        meter = AverageMeter()
        comp_meters = {}
        grad_norms = []

        ds_meters: Dict[str, Tuple[AverageMeter, int]] = {}
        total_batches = len(self.train_loader) if self.train_loader is not None else 0
        t0 = time.time()


        target_size = getattr(self, "_phase_image_size",
                              self.config["data"].get("image_size", 256))

        self.model.train()
        for i, batch in enumerate(self.train_loader):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}


            img_h, img_w = batch["image"].shape[2], batch["image"].shape[3]
            if (img_h, img_w) != (target_size, target_size):
                batch["image"] = F.interpolate(
                    batch["image"], size=(target_size, target_size),
                    mode="bilinear", align_corners=False)
                if batch.get("reflectance") is not None:
                    batch["reflectance"] = F.interpolate(
                        batch["reflectance"], size=(target_size, target_size),
                        mode="bilinear", align_corners=False)
                if batch.get("shading") is not None:
                    batch["shading"] = F.interpolate(
                        batch["shading"], size=(target_size, target_size),
                        mode="bilinear", align_corners=False)
                if batch.get("mask") is not None:
                    batch["mask"] = F.interpolate(
                        batch["mask"], size=(target_size, target_size),
                        mode="nearest")
                if batch.get("saw_mask") is not None:
                    batch["saw_mask"] = F.interpolate(
                        batch["saw_mask"], size=(target_size, target_size),
                        mode="nearest")


            with (autocast() if self.use_amp else nullcontext()):
                outputs = self.model(batch["image"])


                if self._dp and "P1" in outputs:
                    n_ch = outputs["R1"].shape[1]
                    outputs["P1"] = outputs["P1"][:n_ch]
                    outputs["P2"] = outputs["P2"][:n_ch]
                loss_dict = compute_total_loss(
                    outputs, batch, self.current_phase,
                    {**self.loss_weights, **ramp_weights},
                    self.loss_weights_cfg, model=self.model,
                    active_losses=self._active_losses,
                )
                loss = loss_dict["loss"]

            grad_pre = 0.0
            grad_post = 0.0
            if loss.requires_grad:
                self.optimizer.zero_grad()

                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)

                    if _grad_has_inf_nan(self.model):


                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        continue

                    grad_pre = _compute_grad_norm(self.model)
                    grad_clip = self.config["training"].get("grad_clip_norm", 1.0)
                    if grad_clip > 0:
                        grad_post_val = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), grad_clip)
                        grad_post = float(grad_post_val) if isinstance(grad_post_val, torch.Tensor) else grad_post_val
                    else:
                        grad_post = grad_pre
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()

                    if _grad_has_inf_nan(self.model):
                        self.optimizer.zero_grad()
                        continue

                    grad_pre = _compute_grad_norm(self.model)
                    grad_clip = self.config["training"].get("grad_clip_norm", 1.0)
                    if grad_clip > 0:
                        grad_post_val = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), grad_clip)
                        grad_post = float(grad_post_val) if isinstance(grad_post_val, torch.Tensor) else grad_post_val
                    else:
                        grad_post = grad_pre
                    self.optimizer.step()
                if math.isfinite(grad_pre) and grad_pre > 0:
                    grad_norms.append(grad_pre)

            meter.update(loss.item())

            ds_type = self._dataset_type(batch)
            if ds_type not in ds_meters:
                ds_meters[ds_type] = (AverageMeter(), 0)
            ds_meters[ds_type][0].update(loss.item())
            ds_meters[ds_type] = (ds_meters[ds_type][0], ds_meters[ds_type][1] + 1)
            for k, v in loss_dict["components"].items():
                if k not in comp_meters:
                    comp_meters[k] = AverageMeter()
                comp_meters[k].update(v.item() if isinstance(v, torch.Tensor) else v)
            self.global_step += 1


            lr = self.optimizer.param_groups[0]["lr"]
            tau = (self._raw_model.temperature if hasattr(self._raw_model, "temperature")
                   else 1.0)
            self.logger.log_step(
                epoch=epoch, batch_idx=i, global_step=self.global_step,
                loss_dict=loss_dict, lr=lr, temperature=tau,
                batch=batch, outputs=outputs,
                grad_norm_pre=grad_pre if grad_pre > 0 else None,
                grad_norm_post=grad_post if grad_post > 0 else None,
            )


            if self.global_step % self._diag_interval == 0:
                P1_grad = _compute_param_norm(self.model, "P1_raw")
                P2_grad = _compute_param_norm(self.model, "P2_raw")
                self.logger.log_diagnostics(
                    epoch=epoch, global_step=self.global_step,
                    outputs=outputs,
                    P1_grad_norm=P1_grad, P2_grad_norm=P2_grad,
                )


            if (self._print_batch_progress_enabled(self.config) and
                    (i == 0 or (i + 1) % 10 == 0 or i == total_batches - 1)):
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (total_batches - i - 1) / rate if rate > 0 else 0
                pct = (i + 1) / max(total_batches, 1) * 100
                bar_len = 25
                filled = int(bar_len * (i + 1) / max(total_batches, 1))
                bar = "█" * filled + "░" * (bar_len - filled)
                cur_loss = loss.item()
                print(f"\r  Epoch {epoch:3d} [{bar}] {pct:5.1f}%  {i+1}/{total_batches}"
                      f"  loss={cur_loss:.4f}  {rate:.1f}it/s  ETA {eta:.0f}s",
                      end="", flush=True)

        result = {k: m.avg for k, m in comp_meters.items()}
        result["total"] = meter.avg

        ds_summary = {}
        if len(ds_meters) > 1:
            for ds_type, (ds_meter, ds_count) in ds_meters.items():
                key = ds_type.replace('/', '_')
                result[f"total_{key}"] = ds_meter.avg
                ds_summary[ds_type] = (ds_meter.avg, ds_count)
        result["_ds_summary"] = ds_summary
        grad_avg = sum(grad_norms) / len(grad_norms) if grad_norms else 0.0
        self.writer.flush()
        return result, grad_avg, total_batches





    def _get_ramp_weights(self, epoch: int, phase_cfg: dict) -> Dict[str, float]:

        ramp_epochs = phase_cfg.get("loss_warmup_epochs", 0)
        if ramp_epochs <= 0:
            return {}

        start_e = phase_cfg["epochs"][0]
        progress = min(1.0, (epoch - start_e) / ramp_epochs)

        rampable = {"entropy", "normal", "shading_regularization", "center"}
        weights = {}
        for k in phase_cfg.get("active_losses", []):
            target = self.loss_weights.get(k, 1.0)
            if k in rampable:
                weights[k] = target * progress
            else:
                weights[k] = target
        return weights





    def _get_selection_eval_loader(self, dataset_name: str):
        if dataset_name == "iiw" and self.val_loader is not None:
            return self.val_loader
        if dataset_name not in self._selection_eval_loaders:
            self._selection_eval_loaders[dataset_name] = create_dataloader(
                self.config, split="val", dataset_name=dataset_name
            )
        return self._selection_eval_loaders[dataset_name]

    @torch.no_grad()
    def _validate_checkpoint_selection(
        self, val_whdr: float
    ) -> Tuple[Dict[str, float], float]:
        from .evaluate import (
            evaluate_cgintrinsics,
            evaluate_hypersim,
            evaluate_saw_ap,
        )

        selection_cfg = self.config.get("checkpoint_selection", {})
        metrics = {"whdr": float(val_whdr)}
        previous_size = getattr(self._raw_model, "_infer_image_size", None)
        self._raw_model._infer_image_size = getattr(
            self, "_phase_image_size", self.config["data"].get("image_size", 256)
        )

        try:
            cg_result = evaluate_cgintrinsics(
                self.model,
                self._get_selection_eval_loader("cgintrinsics"),
                device=str(self.device),
                config=self.config,
            )
            metrics["cgintrinsics_mse"] = float(cg_result["mse_mean"])

            hs_result = evaluate_hypersim(
                self.model,
                self._get_selection_eval_loader("hypersim"),
                device=str(self.device),
                config=self.config,
            )
            metrics["hypersim_refl_mse"] = float(hs_result["refl_mse_mean"])

            saw_result = evaluate_saw_ap(
                self.model,
                self._get_selection_eval_loader("saw"),
                device=str(self.device),
                config=self.config,
            )
            metrics["saw_error"] = 1.0 - float(saw_result["ap_mean"])
        except (FileNotFoundError, KeyError, RuntimeError, ValueError) as error:
            print(f"  [checkpoint selection] validation incomplete: {error}")
        finally:
            if previous_size is None:
                delattr(self._raw_model, "_infer_image_size")
            else:
                self._raw_model._infer_image_size = previous_size
            self.model.train()

        score = _compute_checkpoint_selection_score(metrics, selection_cfg)
        return metrics, score

    @torch.no_grad()
    def _validate_whdr(self) -> Tuple[float, Optional[str]]:

        if self.val_loader is None:
            return float("nan"), None

        self.model.eval()
        error_sum = 0.0
        weight_sum = 0.0


        _diag_samples = []
        _diag_max_samples = 8


        val_target_size = getattr(self, "_phase_image_size",
                                  self.config["data"].get("image_size", 256))

        for batch in self.val_loader:
            if not batch.get("judgements"):
                continue
            image = batch["image"].to(self.device)
            if image.shape[2] != val_target_size:
                image = F.interpolate(
                    image, size=(val_target_size, val_target_size),
                    mode="bilinear", align_corners=False)
            outputs = self.model(image)
            R = _select_whdr_reflectance(outputs, self.config)
            delta = self.config.get("evaluation", {}).get("whdr_delta", 0.10)
            ratio_threshold = 1.0 + delta

            for b in range(R.shape[0]):
                judgements = batch["judgements"]
                if isinstance(judgements, list) and len(judgements) > b:
                    js = judgements[b] if isinstance(judgements[b], list) else judgements
                else:
                    continue

                r_np = R[b].permute(1, 2, 0).cpu().numpy()
                H, W = r_np.shape[:2]

                for j in js:
                    w = j.get("weight", 0)
                    if w <= 0:
                        continue
                    p1, p2 = j["point1"], j["point2"]
                    r1 = max(1e-10, r_np[int(p1["y"]*H), int(p1["x"]*W)].mean())
                    r2 = max(1e-10, r_np[int(p2["y"]*H), int(p2["x"]*W)].mean())


                    if len(_diag_samples) < _diag_max_samples:
                        _diag_samples.append((r1, r2, j["darker"]))

                    if r2/r1 > ratio_threshold:
                        alg = "1"
                    elif r1/r2 > ratio_threshold:
                        alg = "2"
                    else:
                        alg = "E"

                    if j["darker"] != alg:
                        error_sum += w
                    weight_sum += w

        whdr = error_sum / weight_sum if weight_sum > 0 else float("nan")


        diag_str = None
        if (self.current_epoch <= 3 or self.current_epoch % 5 == 0) and _diag_samples:
            ratios = [f"r2/r1={r2/r1:.3f}" for r1, r2, _ in _diag_samples]
            r1_vals = [f"{r1:.4f}" for r1, _, _ in _diag_samples]
            diag_str = (f"error={error_sum:.0f}/{weight_sum:.0f} "
                       f"r1=[{', '.join(r1_vals[:4])}] "
                       f"r2/r1=[{', '.join(ratios[:4])}]")

        self.model.train()
        return whdr, diag_str





    def _reset_adamw_state(self):

        for state in self.optimizer.state.values():
            if "exp_avg" in state:
                state["exp_avg"].zero_()
            if "exp_avg_sq" in state:

                state["exp_avg_sq"].fill_(1.0)
            if "max_exp_avg_sq" in state:
                state["max_exp_avg_sq"].fill_(1.0)





    def _save_checkpoint(self, epoch: int):
        path = self.ckpt_dir / f"phase{self.current_phase}_epoch{epoch:04d}.pth"
        torch.save({
            "epoch": epoch,
            "phase": self.current_phase,
            "global_step": self.global_step,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "temperature": self._raw_model.temperature if hasattr(self._raw_model, "temperature") else 1.0,
            "image_size": getattr(self, "_phase_image_size",
                                  self.config["data"].get("image_size", 256)),
            "history": self.history,
            "best_selection_score": self.best_selection_score,
            "best_whdr": self.best_whdr,
            "best_epoch": self.best_epoch,
        }, path)

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        state_dict = ckpt["state_dict"]


        if any(k.startswith("module.") for k in state_dict.keys()):
            from collections import OrderedDict
            state_dict = OrderedDict(
                (k.replace("module.", "", 1), v) for k, v in state_dict.items()
            )




        missing, unexpected = self._raw_model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  [checkpoint] Missing keys ({len(missing)}): {missing[:5]}..."
                  if len(missing) > 5 else f"  [checkpoint] Missing keys ({len(missing)}): {missing}")
        if unexpected:
            print(f"  [checkpoint] Unexpected keys ({len(unexpected)}): {unexpected[:5]}..."
                  if len(unexpected) > 5 else f"  [checkpoint] Unexpected keys ({len(unexpected)}): {unexpected}")
        self.current_epoch = ckpt.get("epoch", 0)

        if "phase" in ckpt:
            self.current_phase = ckpt["phase"]
        else:
            self.current_phase = self._infer_phase(self.current_epoch)
        self.global_step = ckpt.get("global_step", 0)
        self.best_selection_score = ckpt.get("best_selection_score", float("inf"))
        self.best_whdr = ckpt.get("best_whdr", ckpt.get("val_whdr", float("inf")))
        self.best_epoch = ckpt.get("best_epoch", self.current_epoch)


        optimizer_loaded = "optimizer" in ckpt
        if optimizer_loaded:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        else:
            print("  [checkpoint] No optimizer state — using fresh optimizer "
                  "(weights are restored, but this is not an exact optimizer resume)")

        scheduler_state = ckpt.get("scheduler")
        total_epochs = self.config["training"]["epochs"]
        if optimizer_loaded and scheduler_state and self.scheduler:
            if _scheduler_state_matches_training_plan(scheduler_state, total_epochs):
                self.scheduler.load_state_dict(scheduler_state)
            else:
                remaining_epochs = max(1, total_epochs - self.current_epoch)
                eta_min = getattr(self.scheduler, "eta_min", 5e-5)
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=remaining_epochs, eta_min=eta_min
                )
                print(
                    f"  [checkpoint] Training horizon changed to {total_epochs}; "
                    f"rebuilt cosine scheduler for {remaining_epochs} remaining epochs"
                )


        if hasattr(self._raw_model, "temperature") and "temperature" in ckpt:
            self._raw_model.temperature = ckpt["temperature"]
        self._resume_epoch = self.current_epoch
        self._resume_temperature = (
            ckpt.get("temperature")
            if "temperature" in ckpt
            else getattr(self._raw_model, "temperature", None)
        )
        print(f"Resumed from epoch {self.current_epoch}, phase {self.current_phase}")

    def _infer_phase(self, epoch: int) -> int:

        phases = self.config.get("phases", {})
        for phase_name, phase_cfg in phases.items():
            start_e, end_e = phase_cfg["epochs"]
            if start_e <= epoch <= end_e:
                return int(phase_name.replace("phase", ""))
        return 1






def _grad_has_inf_nan(model: nn.Module) -> bool:

    for p in model.parameters():
        if p.grad is not None:
            if not torch.isfinite(p.grad).all():
                return True
    return False


def _compute_grad_norm(model: nn.Module) -> float:

    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            gn = p.grad.detach().norm(2).pow(2).item()
            if math.isfinite(gn):
                total += gn
    return total ** 0.5


def _compute_param_norm(model: nn.Module, param_name: str) -> Optional[float]:

    for name, p in model.named_parameters():
        if param_name in name and p.grad is not None:
            return float(p.grad.detach().norm(2).item())
    return None






def train_main(config_path: str = "config/default.yaml", resume: Optional[str] = None):

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    from .architecture import CDBFNet
    model = CDBFNet(config).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    trainer = Trainer(model, config)
    if resume:
        trainer.load_checkpoint(resume)

    history = trainer.train()
    return model, history
