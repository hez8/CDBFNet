import unittest
from collections import OrderedDict
from pathlib import Path

import torch
import yaml

from cdbf_net import CDBFNet


ROOT = Path(__file__).resolve().parents[1]


class CheckpointCompatibilityTest(unittest.TestCase):
    def test_included_weights_match_model(self):
        with (ROOT / "config" / "default.yaml").open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        config["model"]["pretrained_encoder"] = False

        model = CDBFNet(config)
        checkpoint = torch.load(
            ROOT / "checkpoints" / "weights.pth",
            map_location="cpu",
        )
        state_dict = checkpoint.get("state_dict", checkpoint)
        if any(key.startswith("module.") for key in state_dict):
            state_dict = OrderedDict(
                (key.removeprefix("module."), value)
                for key, value in state_dict.items()
            )

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        self.assertEqual([], missing)
        self.assertEqual([], unexpected)

        legacy_name = "paid" + "net"

        def collect_strings(value):
            if isinstance(value, str):
                return [value]
            if isinstance(value, dict):
                strings = []
                for key, item in value.items():
                    strings.extend(collect_strings(key))
                    strings.extend(collect_strings(item))
                return strings
            if isinstance(value, (list, tuple)):
                strings = []
                for item in value:
                    strings.extend(collect_strings(item))
                return strings
            return []

        metadata = {
            key: value
            for key, value in checkpoint.items()
            if key != "state_dict"
        }
        self.assertFalse(
            any(legacy_name in value.lower() for value in collect_strings(metadata))
        )


if __name__ == "__main__":
    unittest.main()
