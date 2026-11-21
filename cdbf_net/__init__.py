

from collections import OrderedDict

import torch

from .architecture import CDBFNet


def load_model_weights(model, checkpoint_path: str, map_location="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    state_dict = checkpoint.get("state_dict", checkpoint)
    if any(key.startswith("module.") for key in state_dict):
        state_dict = OrderedDict(
            (key.removeprefix("module."), value)
            for key, value in state_dict.items()
        )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Missing checkpoint keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(
            f"Unexpected checkpoint keys: {unexpected[:5]}"
            f"{'...' if len(unexpected) > 5 else ''}"
        )

    if hasattr(model, "temperature"):
        model.temperature = checkpoint.get("temperature", 0.1)
        print(f"Model temperature = {model.temperature:.4f}")

    image_size = checkpoint.get("image_size")
    if image_size is not None:
        model._infer_image_size = image_size
        print(f"Eval image_size = {image_size}")

    print(f"Loaded weights from {checkpoint_path}")
    return checkpoint
