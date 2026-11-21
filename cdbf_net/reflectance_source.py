from typing import Dict, Optional

import torch


def select_reflectance_output(
    outputs: Dict[str, torch.Tensor],
    source: Optional[str] = None,
) -> torch.Tensor:

    source = (source or "total").lower()
    if source in {"calibrated", "whdr", "whdr_calibrated", "reflectance_calibrated"}:
        if "reflectance_calibrated" not in outputs:
            raise ValueError("reflectance_calibrated is unavailable; enable whdr_head first")
        return outputs["reflectance_calibrated"]
    if source in {"total", "sum", "r_total", "r1+r2", "reflectance"}:
        if "reflectance" in outputs:
            return outputs["reflectance"]
        return outputs["R1"] + outputs["R2"]
    if source in {"r1", "specular"}:
        return outputs["R1"]
    if source in {"r2", "diffuse", "albedo"}:
        return outputs["R2"]
    raise ValueError(
        "reflectance_source must be one of: calibrated, total, R1/specular, R2/diffuse"
    )
