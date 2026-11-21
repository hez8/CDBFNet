import copy
import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from cdbf_net import CDBFNet, load_model_weights
from cdbf_net.dataset import create_dataloader
from cdbf_net.evaluate import (
    evaluate_cgintrinsics,
    evaluate_hypersim,
    evaluate_saw_ap,
    evaluate_whdr_bootstrap,
)
from cdbf_net.train import Trainer


VARIANTS = {
    "full_model": {},
    "independent_coefficients": {"independent_coefficients": True},
    "direct_reflectance": {"direct_reflectance": True},
    "single_lambertian": {"single_lambertian": True},
    "direct_intrinsic": {"direct_intrinsic": True},
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def variant_config(base: dict, name: str, seed: int, output: Path) -> dict:
    if name not in VARIANTS:
        raise ValueError(f"Unknown variant {name!r}; choose from {sorted(VARIANTS)}")
    config = copy.deepcopy(base)
    switches = config["model"].setdefault("ablation", {})
    for key in switches:
        switches[key] = False
    switches.update(VARIANTS[name])
    config["training"]["seed"] = seed
    run_dir = output / name / f"seed_{seed}"
    config["logging"]["checkpoint_dir"] = str(run_dir / "checkpoints")
    config["logging"]["log_dir"] = str(run_dir / "log")
    return config


def evaluate_variant(model, config: dict, device: torch.device) -> dict:
    loaders = {
        name: create_dataloader(config, split="val", dataset_name=name)
        for name in ("iiw", "cgintrinsics", "hypersim", "saw")
    }
    bootstrap = int(config.get("ablation", {}).get("bootstrap_samples", 1000))
    return {
        "iiw": evaluate_whdr_bootstrap(
            model, loaders["iiw"], n_bootstrap=bootstrap,
            device=str(device), config=config,
        ),
        "cgintrinsics": evaluate_cgintrinsics(
            model, loaders["cgintrinsics"], device=str(device), config=config,
        ),
        "hypersim": evaluate_hypersim(
            model, loaders["hypersim"], device=str(device), config=config,
        ),
        "saw": evaluate_saw_ap(
            model, loaders["saw"], device=str(device), config=config,
        ),
    }


def run_ablation(
    base_config: dict,
    variants: Iterable[str],
    seeds: Iterable[int],
    output_dir: str,
) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for name in variants:
        for seed in seeds:
            set_seed(seed)
            config = variant_config(base_config, name, seed, output)
            run_dir = output / name / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            summary_path = run_dir / "summary.json"
            if config.get("ablation", {}).get("skip_completed", True) and summary_path.exists():
                summaries[f"{name}/seed_{seed}"] = json.loads(
                    summary_path.read_text(encoding="utf-8")
                )
                continue

            model = CDBFNet(config)
            trainer = Trainer(model, config)
            trainer.train()
            trainer.writer.close()
            best = trainer.ckpt_dir / "best.pth"
            if best.exists():
                load_model_weights(trainer._raw_model, str(best), map_location=str(trainer.device))

            summary = {
                "variant": name,
                "seed": seed,
                "parameters": int(sum(p.numel() for p in trainer._raw_model.parameters())),
                "metrics": evaluate_variant(trainer._raw_model, config, trainer.device),
            }
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            summaries[f"{name}/seed_{seed}"] = summary

    (output / "summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )
    return summaries
