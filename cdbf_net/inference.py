from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from .reflectance_source import select_reflectance_output


@torch.no_grad()
def inference_fast(model: nn.Module, image: torch.Tensor) -> Dict[str, torch.Tensor]:

    was_training = model.training
    model.eval()
    if image.dim() == 3:
        image = image.unsqueeze(0)
    outputs = model(image)
    if was_training:
        model.train()

    result = {
        "reflectance": outputs.get("reflectance", outputs["R1"] + outputs["R2"]),
        "shading": outputs["Shading"],
        "R1": outputs["R1"],
        "R2": outputs["R2"],
        "N": outputs["N_vec"],
        "L_S_P": outputs["L_S_P"],
        "C": outputs["C"],
        "n_map": outputs["n_map"],
    }
    for key in ("reflectance_calibrated", "whdr_score"):
        if key in outputs:
            result[key] = outputs[key]
    return result


def save_decomposition(
    outputs: Dict[str, torch.Tensor],
    output_dir: str,
    filename: str,
    reflectance_source: str = "R2",
) -> None:

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    stem = Path(filename).stem
    reflectance = select_reflectance_output(outputs, reflectance_source)

    def save_tensor(tensor: torch.Tensor, suffix: str) -> None:
        array = tensor.detach().squeeze(0).permute(1, 2, 0).cpu().numpy()
        array = (np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
        Image.fromarray(array.squeeze() if array.shape[-1] == 1 else array).save(
            destination / f"{stem}_{suffix}.png"
        )

    save_tensor(reflectance, "reflectance")
    save_tensor(outputs["shading"], "shading")
    save_tensor(outputs["R1"], "specular")
    save_tensor(outputs["R2"], "diffuse")
