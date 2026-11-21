import argparse
import json
from pathlib import Path

import torch
import yaml
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, resize

from cdbf_net import CDBFNet, load_model_weights
from cdbf_net.dataset import create_dataloader
from cdbf_net.evaluate import (
    evaluate_cgintrinsics,
    evaluate_hypersim,
    evaluate_saw_ap,
    evaluate_whdr_bootstrap,
)
from cdbf_net.inference import inference_fast, save_decomposition
from cdbf_net.train import train_main


DATASETS = ("iiw", "cgintrinsics", "hypersim", "saw")


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_model(config: dict, checkpoint: str, device: torch.device) -> CDBFNet:
    model = CDBFNet(config).to(device)
    saved = load_model_weights(model, checkpoint, map_location=str(device))
    if getattr(model, "_infer_image_size", None) is None:
        saved_epoch = int(saved.get("epoch", 0))
        for phase in config.get("phases", {}).values():
            start_epoch, end_epoch = phase["epochs"]
            if start_epoch <= saved_epoch <= end_epoch:
                model._infer_image_size = phase.get(
                    "image_size", config["data"]["image_size"]
                )
                print(f"Eval image_size = {model._infer_image_size} (from phase)")
                break
    model.eval()
    return model


def evaluate(model, config: dict, dataset: str, device: torch.device) -> dict:
    loader = create_dataloader(config, split="val", dataset_name=dataset)
    if dataset == "iiw":
        evaluation = config.get("evaluation", {})
        return evaluate_whdr_bootstrap(
            model,
            loader,
            n_bootstrap=int(evaluation.get("bootstrap_samples", 1000)),
            seed=int(evaluation.get("bootstrap_seed", 42)),
            device=str(device),
            config=config,
        )
    if dataset == "cgintrinsics":
        return evaluate_cgintrinsics(model, loader, device=str(device), config=config)
    if dataset == "hypersim":
        return evaluate_hypersim(model, loader, device=str(device), config=config)
    return evaluate_saw_ap(model, loader, device=str(device), config=config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal CDBF-Net implementation")
    parser.add_argument("--config", default="config/default.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--resume")

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--checkpoint")
    evaluate_parser.add_argument("--dataset", choices=(*DATASETS, "all"), default="all")

    infer_parser = subparsers.add_parser("infer")
    infer_parser.add_argument("--checkpoint")
    infer_parser.add_argument("--image", required=True)
    infer_parser.add_argument("--output", default="results/inference")

    ablation_parser = subparsers.add_parser("ablation")
    ablation_parser.add_argument("--variants", nargs="+", required=True)
    ablation_parser.add_argument("--seeds", nargs="+", type=int, default=[2021])
    ablation_parser.add_argument("--output", default="results/ablation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.command == "train":
        train_main(args.config, resume=args.resume)
        return

    if args.command == "ablation":
        from ablation import run_ablation

        run_ablation(config, args.variants, args.seeds, args.output)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = args.checkpoint or config.get("evaluation", {}).get("checkpoint")
    if not checkpoint:
        raise ValueError(
            "No checkpoint configured; set evaluation.checkpoint or pass --checkpoint"
        )
    model = load_model(config, checkpoint, device)
    if args.command == "evaluate":
        datasets = DATASETS if args.dataset == "all" else (args.dataset,)
        results = {name: evaluate(model, config, name, device) for name in datasets}
        print(json.dumps(results, indent=2))
        return

    image_path = Path(args.image)
    image = Image.open(image_path).convert("RGB")
    size = int(config["data"]["image_size"])
    tensor = resize(pil_to_tensor(image).float() / 255.0, [size, size]).unsqueeze(0).to(device)
    outputs = inference_fast(model, tensor)
    source = config.get("reflectance_sources", {}).get("default", "R2")
    save_decomposition(outputs, args.output, image_path.name, source)


if __name__ == "__main__":
    main()
