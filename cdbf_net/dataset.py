

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, BatchSampler, RandomSampler
from typing import Dict, Optional, List, Tuple, Iterator
import numpy as np
import json
import os
from pathlib import Path
from PIL import Image
import random


def _to_float32(sample: dict) -> dict:

    for k, v in sample.items():
        if isinstance(v, torch.Tensor) and v.dtype == torch.float16:
            sample[k] = v.float()
    return sample






class CGIntrinsicsDataset(Dataset):


    def __init__(self, root: str, split: str = "train",
                 image_size: int = 256, augment: bool = True,
                 val_ratio: float = 0.2,
                 preprocessed_root: str = "preprocess_data"):
        self.image_size = image_size
        self.augment = augment and (split == "train")
        self._preprocessed = False
        self._samples = []
        self._raw_root = None

        if split not in ("train", "val"):
            return


        _split = "test" if split == "val" else "train"


        processed_dir = Path(preprocessed_root) / "cgintrinsics" / _split
        if processed_dir.exists():
            pt_files = sorted(processed_dir.glob("*.pt"))
            if pt_files:
                self._preprocessed = True
                self._samples = pt_files
                return



        self._raw_root = Path(root) / "intrinsics_final"
        img_dir = self._raw_root / "images"
        all_paths = sorted(list(img_dir.rglob("*.png")) + list(img_dir.rglob("*.jpg")))
        all_samples = [p.relative_to(img_dir).with_suffix("") for p in all_paths]
        n_total = len(all_samples)
        n_val = max(1, int(n_total * val_ratio))
        rng = random.Random(42)
        rng.shuffle(all_samples)
        split_at = n_total - n_val
        self._samples = all_samples[:split_at] if split == "train" else all_samples[split_at:]
        if not self._samples:
            print(f"Warning: No images found in {img_dir}, using dummy mode")

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        if not self._samples:
            raise RuntimeError(
                f"CGIntrinsicsDataset is empty (root={self._raw_root or 'preprocess_data/cgintrinsics/train'}). "
                f"Please download the dataset or run preprocess_data.py first.")

        if self._preprocessed:
            sample = _to_float32(torch.load(self._samples[idx], weights_only=False))
            if self.augment and random.random() > 0.5:
                for k in sample:
                    if isinstance(sample[k], torch.Tensor):
                        sample[k] = torch.flip(sample[k], dims=[-1])
            return sample


        rel_path = self._samples[idx]
        img_dir = self._raw_root / "images"
        img_path = img_dir / f"{rel_path}.png"
        if not img_path.exists():
            img_path = img_dir / f"{rel_path}.jpg"
        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0

        refl_path = self._raw_root / "rendered" / f"{rel_path}_refl.png"
        shad_path = self._raw_root / "rendered" / f"{rel_path}_shad.png"
        reflectance = self._load_optional(refl_path, default=image)
        shading = self._load_optional(shad_path, default=np.ones_like(image) * 0.5)
        sample = {
            "image":       torch.from_numpy(image.transpose(2, 0, 1)),
            "reflectance": torch.from_numpy(reflectance.transpose(2, 0, 1)),
            "shading":     torch.from_numpy(shading.transpose(2, 0, 1)),
        }
        return self._apply_transform(sample)

    def _load_optional(self, path: Path, default: np.ndarray) -> np.ndarray:
        if path.exists():
            return np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        return default

    def _dummy_item(self) -> dict:
        C = 3
        return {"image": torch.rand(C, self.image_size, self.image_size),
                "reflectance": torch.rand(C, self.image_size, self.image_size),
                "shading": torch.rand(C, self.image_size, self.image_size)}

    def _apply_transform(self, sample: dict) -> dict:
        H, W = sample["image"].shape[1], sample["image"].shape[2]
        if self.image_size < min(H, W):
            if self.augment:
                i = random.randint(0, H - self.image_size)
                j = random.randint(0, W - self.image_size)
            else:

                i = (H - self.image_size) // 2
                j = (W - self.image_size) // 2
            for k in sample:
                sample[k] = sample[k][:, i:i+self.image_size, j:j+self.image_size]
        if self.augment and random.random() > 0.5:
            for k in sample:
                sample[k] = torch.flip(sample[k], dims=[-1])
        return sample






class HyperSimDataset(Dataset):


    def __init__(self, root: str, split: str = "train",
                 image_size: int = 256, augment: bool = True,
                 max_scenes: int = 50, val_ratio: float = 0.2,
                 preprocessed_root: str = "preprocess_data"):
        self.image_size = image_size
        self.augment = augment and (split == "train")
        self._preprocessed = False
        self._samples = []
        self._raw_root = None
        self._sample_meta = []

        if split not in ("train", "val"):
            return

        _split = "test" if split == "val" else "train"


        processed_dir = Path(preprocessed_root) / "hypersim" / _split
        if processed_dir.exists():
            pt_files = sorted(processed_dir.glob("*.pt"))
            if pt_files:
                self._preprocessed = True
                self._samples = pt_files
                return



        self._raw_root = Path(root)
        scenes_dir = self._raw_root / "scenes"
        scenes = sorted(scenes_dir.glob("ai_*")) if scenes_dir.exists() else []
        if max_scenes and len(scenes) > max_scenes:
            scenes = scenes[:max_scenes]
        for scene in scenes:

            inner = scene / scene.name if (scene / scene.name).is_dir() else scene
            for img_dir in [inner / "images", scene / "images"]:
                cam_dirs = sorted(img_dir.glob("scene_cam_*")) if img_dir.exists() else []
                if cam_dirs:
                    break
            for cam_dir in cam_dirs:
                for cf in sorted(cam_dir.glob("frame.*.color.hdf5")):
                    prefix = cf.name.replace(".color.hdf5", "")
                    self._sample_meta.append({
                        "color": str(cf),
                        "reflectance": str(cam_dir / f"{prefix}.diffuse_reflectance.hdf5"),
                        "shading": str(cam_dir / f"{prefix}.diffuse_illumination.hdf5"),
                    })
        n_total = len(self._sample_meta)
        n_val = max(1, int(n_total * val_ratio))
        rng = random.Random(42)
        rng.shuffle(self._sample_meta)
        split_at = n_total - n_val
        self._samples = self._sample_meta[:split_at] if split == "train" else self._sample_meta[split_at:]
        if not self._samples:
            print(f"Warning: No HyperSim frames found in {self._raw_root}")

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        if not self._samples:
            raise RuntimeError(
                f"HyperSimDataset is empty (root={self._raw_root or 'preprocess_data/hypersim/train'}). "
                f"Please download the dataset or run preprocess_data.py first.")

        if self._preprocessed:
            sample = _to_float32(torch.load(self._samples[idx], weights_only=False))
            if self.augment and random.random() > 0.5:
                for k in sample:
                    if isinstance(sample[k], torch.Tensor):
                        sample[k] = torch.flip(sample[k], dims=[-1])
            return sample


        if not self._sample_meta:
            raise RuntimeError(
                f"HyperSimDataset has no sample metadata (root={self._raw_root}). "
                f"Please check the dataset path or run preprocess_data.py.")
        try:
            import h5py
            s = self._sample_meta[idx % len(self._sample_meta)]
            image = self._load_hdf5(s["color"]).float() / 255.0
            reflectance = self._load_hdf5(s["reflectance"]).float()
            shading = self._load_hdf5(s["shading"]).float()
        except Exception:
            return self._dummy_item()

        image = image.clamp(0, 1)
        reflectance = reflectance.clamp(0, 1)
        shading = shading.clamp(0, 5)
        sample = {"image": image, "reflectance": reflectance, "shading": shading}
        return self._apply_transform(sample)

    @staticmethod
    def _load_hdf5(path_str: str) -> torch.Tensor:
        import h5py
        with h5py.File(path_str, "r") as f:
            key = list(f.keys())[0]
            arr = np.array(f[key], dtype=np.float32)
        if arr.ndim == 3 and arr.shape[-1] == 3:
            arr = arr.transpose(2, 0, 1)
        elif arr.ndim == 2:
            arr = arr[np.newaxis, :, :]
        return torch.from_numpy(arr)

    def _dummy_item(self) -> dict:
        return {"image": torch.rand(3, self.image_size, self.image_size),
                "reflectance": torch.rand(3, self.image_size, self.image_size),
                "shading": torch.rand(3, self.image_size, self.image_size)}

    @staticmethod
    def _apply_transform(sample: dict) -> dict:
        for k in sample:
            if sample[k].dim() == 3:
                sample[k] = torch.nn.functional.interpolate(
                    sample[k].unsqueeze(0), size=(256, 256),
                    mode="bilinear", align_corners=False).squeeze(0)
        return sample






class IIWDataset(Dataset):


    def __init__(self, root: str, split: str = "train", image_size: int = 256,
                 preprocessed_root: str = "preprocess_data"):
        self.image_size = image_size
        self._preprocessed = False
        self._samples = []
        self._raw_root = None


        storage_split = "test" if split == "val" else "train"
        processed_dir = Path(preprocessed_root) / "iiw" / storage_split
        if processed_dir.exists():
            pt_files = sorted(processed_dir.glob("*.pt"))
            if pt_files:
                self._preprocessed = True
                self._samples = pt_files
                return


        raw_candidates = [
            Path(root) / "data",
            Path(root) / "iiw-dataset" / "data",
            Path(root) / "IIW" / "data",
            Path(root),
        ]
        self._raw_root = None
        images = []
        for cand in raw_candidates:
            if cand.exists():
                imgs = sorted(list(cand.glob("*.png")) + list(cand.glob("*.jpg")))
                if len(imgs) > 100:
                    self._raw_root = cand
                    images = imgs
                    break
        if self._raw_root is None:
            self._raw_root = raw_candidates[0]
            images = []
        split_file = Path(root) / f"{split}_list.txt"
        if split_file.exists():
            with open(split_file) as f:
                valid = {line.strip().replace(".png", "").replace(".jpg", "")
                        for line in f if line.strip()}
            images = [p for p in images if p.stem in valid]
        else:
            n = max(1, len(images))
            idx = int(n * 0.8)
            images = images[:idx] if split == "train" else images[idx:]
        self._samples = images

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        if not self._samples:
            raise RuntimeError(
                f"IIWDataset is empty (root={self._raw_root or 'preprocess_data/iiw/'}). "
                f"Please download the IIW dataset or run preprocess_data.py first.")

        if self._preprocessed:
            return _to_float32(torch.load(self._samples[idx], weights_only=False))


        img_path = self._samples[idx]
        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
        image = torch.from_numpy(image.transpose(2, 0, 1))
        json_path = img_path.with_suffix(".json")
        judgements = []
        if json_path.exists():
            with open(json_path) as f:
                judgements = self._parse_judgements(json.load(f))
        image = torch.nn.functional.interpolate(
            image.unsqueeze(0), size=(self.image_size, self.image_size),
            mode="bilinear", align_corners=False).squeeze(0)
        return {"image": image, "judgements": judgements, "filename": img_path.name}

    @staticmethod
    def _parse_judgements(raw: dict) -> List[dict]:

        points = {p["id"]: p for p in raw.get("intrinsic_points", [])}
        parsed = []
        for c in raw.get("intrinsic_comparisons", []):
            darker = c.get("darker", "E")
            w = c.get("darker_score", 0)
            if not w or w <= 0:
                continue
            p1 = points.get(c["point1"])
            p2 = points.get(c["point2"])
            if p1 and p2 and p1.get("opaque", True) and p2.get("opaque", True):
                parsed.append({
                    "point1": {"x": p1["x"], "y": p1["y"]},
                    "point2": {"x": p2["x"], "y": p2["y"]},
                    "darker": darker,
                    "weight": w,
                })
        return parsed


def create_iiw_training_dataset(config: dict, root: str, image_size: int) -> Dataset:

    return IIWDataset(
        root,
        split="train",
        image_size=image_size,
        preprocessed_root=config["data"].get(
            "preprocessed_path", "preprocess_data"
        ),
    )






class MixedBatchSampler(BatchSampler):


    def __init__(self, datasets: List[Dataset], mix_ratios: Dict[str, int],
                 batch_size: int, shuffle: bool = True):
        self.datasets = datasets
        self.mix_ratios = mix_ratios
        self.batch_size = batch_size
        self.shuffle = shuffle


        total_ratio = sum(mix_ratios.values())
        self.quota = {
            name: max(1, int(batch_size * ratio / total_ratio))
            for name, ratio in mix_ratios.items()
        }

        diff = batch_size - sum(self.quota.values())
        if diff != 0:
            first = list(self.quota.keys())[0]
            self.quota[first] += diff

        self._dataset_index_map = []
        for ds_idx, ds in enumerate(datasets):
            self._dataset_index_map.extend([ds_idx] * len(ds))
        self._total = sum(len(ds) for ds in datasets)

    def __iter__(self) -> Iterator[List[int]]:

        cum_sizes = [0]
        for ds in self.datasets:
            cum_sizes.append(cum_sizes[-1] + len(ds))


        ds_names = list(self.mix_ratios.keys())
        ds_batches: List[List[List[int]]] = []
        for ds_idx, ds in enumerate(self.datasets):
            indices = list(range(len(ds)))
            if self.shuffle:
                random.shuffle(indices)
            batches = []
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                if len(batch_indices) == self.batch_size:
                    flat = [cum_sizes[ds_idx] + idx for idx in batch_indices]
                    batches.append(flat)
            ds_batches.append(batches)

        if not any(ds_batches):
            return


        ratios = [self.mix_ratios.get(name, 1) for name in ds_names]
        import math
        g = ratios[0]
        for r in ratios[1:]:
            g = math.gcd(g, r)
        batch_quotas = [r // g for r in ratios]



        ds_pointers = [0] * len(self.datasets)
        exhausted = [len(b) == 0 for b in ds_batches]

        while not all(exhausted):
            round_batches = []
            for ds_idx, quota in enumerate(batch_quotas):
                if exhausted[ds_idx]:
                    continue
                for _ in range(quota):
                    if ds_pointers[ds_idx] >= len(ds_batches[ds_idx]):
                        exhausted[ds_idx] = True
                        break
                    round_batches.append(ds_batches[ds_idx][ds_pointers[ds_idx]])
                    ds_pointers[ds_idx] += 1

            if not round_batches:
                break

            if self.shuffle:
                random.shuffle(round_batches)

            for batch in round_batches:
                yield batch

    def __len__(self) -> int:

        total = 0
        for ds in self.datasets:
            total += len(ds) // self.batch_size
        return max(1, total)






def create_dataloader(config: dict, split: str = "train",
                      dataset_name: str = None) -> DataLoader:

    image_size = config["data"]["image_size"]
    batch_size = config["training"]["batch_size"]
    num_workers = config["training"].get("num_workers", 4)
    eval_cfg = config.get("evaluation", {})
    val_ratio = eval_cfg.get("val_ratio", 0.2)

    cg_path = config["data"].get("cgintrinsics_path", "./data/cgintrinsics/")
    hs_path = config["data"].get("hypersim_path", "./data/hypersim/")
    iiw_path = config["data"].get("iiw_path", "./data/iiw/")
    saw_path = config["data"].get("saw_path", "./data/saw/")
    preprocessed_root = config["data"].get(
        "preprocessed_path", "preprocess_data"
    )


    config_ratio = config.get("training", {}).get("data_mix_ratio", {})
    excluded_datasets = set(
        config.get("training", {}).get("excluded_datasets", [])
    )
    _default_ratios = {"cgintrinsics": 3, "hypersim": 2, "iiw": 1, "saw": 1}


    hs_max_scenes = config["data"].get("hypersim_num_scenes", 50)

    datasets = []
    mix_names = {}

    if split == "train":

        if "cgintrinsics" not in excluded_datasets and Path(cg_path).exists():
            ds = CGIntrinsicsDataset(
                cg_path, split="train", image_size=image_size,
                augment=True, val_ratio=val_ratio,
                preprocessed_root=preprocessed_root,
            )
            if len(ds) > 0:
                datasets.append(ds)
                mix_names["cgintrinsics"] = config_ratio.get("cgintrinsics", _default_ratios["cgintrinsics"])
            else:
                print(f"Warning: CGIntrinsics path exists but contains no valid samples ({cg_path})")

        if "hypersim" not in excluded_datasets and Path(hs_path).exists():
            ds = HyperSimDataset(
                hs_path, split="train", image_size=image_size,
                augment=True, max_scenes=hs_max_scenes, val_ratio=val_ratio,
                preprocessed_root=preprocessed_root,
            )
            if len(ds) > 0:
                datasets.append(ds)
                mix_names["hypersim"] = config_ratio.get("hypersim", _default_ratios["hypersim"])
            else:
                print(f"Warning: HyperSim path exists but contains no valid samples ({hs_path})")


        if "iiw" not in excluded_datasets and Path(iiw_path).exists():
            ds = create_iiw_training_dataset(config, iiw_path, image_size)
            if len(ds) > 0:
                datasets.append(ds)
                mix_names["iiw"] = config_ratio.get("iiw", _default_ratios["iiw"])
            else:
                print(f"Warning: IIW path exists but contains no valid samples ({iiw_path})")


        if "saw" not in excluded_datasets and Path(saw_path).exists():
            ds = SAWDataset(
                saw_path, split="train", image_size=image_size,
                preprocessed_root=preprocessed_root,
            )
            if len(ds) > 0:
                datasets.append(ds)
                mix_names["saw"] = config_ratio.get("saw", _default_ratios["saw"])
            else:
                print(f"Warning: SAW path exists but contains no valid samples ({saw_path})")

        if len(datasets) == 0:
            raise FileNotFoundError(
                f"No training datasets with valid samples found. Please download at least one dataset. "
                f"Checked: {cg_path}, {hs_path}, {iiw_path}, {saw_path}")

        if len(datasets) == 1:
            return DataLoader(datasets[0], batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=unified_collate)

        sampler = MixedBatchSampler(datasets, mix_names, batch_size=batch_size, shuffle=True)
        return DataLoader(ConcatDataset(datasets), batch_sampler=sampler,
                          num_workers=num_workers, collate_fn=unified_collate)

    elif split == "val":

        ds_name = dataset_name or "iiw"

        if ds_name == "iiw":
            if not Path(iiw_path).exists():
                raise FileNotFoundError(f"IIW validation set not found at {iiw_path}")
            ds = IIWDataset(
                iiw_path, split="val", image_size=image_size,
                preprocessed_root=preprocessed_root,
            )

        elif ds_name == "cgintrinsics":
            if not Path(cg_path).exists():
                raise FileNotFoundError(f"CGIntrinsics not found at {cg_path}")
            ds = CGIntrinsicsDataset(cg_path, split="val", image_size=image_size,
                                     augment=False, val_ratio=val_ratio,
                                     preprocessed_root=preprocessed_root)

        elif ds_name == "hypersim":
            if not Path(hs_path).exists():
                raise FileNotFoundError(f"HyperSim not found at {hs_path}")
            ds = HyperSimDataset(hs_path, split="val", image_size=image_size,
                                 augment=False, max_scenes=hs_max_scenes,
                                 val_ratio=val_ratio,
                                 preprocessed_root=preprocessed_root)

        elif ds_name == "saw":
            if not Path(saw_path).exists():
                raise FileNotFoundError(f"SAW not found at {saw_path}")
            ds = SAWDataset(
                saw_path, split="val", image_size=image_size,
                preprocessed_root=preprocessed_root,
            )

        elif ds_name == "ntire":
            raise NotImplementedError(
                "NTIRE evaluation is not yet implemented. "
                "NTIRE ARAD_1K requires a dedicated dataset class and spectral metrics (SAM, etc.).")

        else:
            raise ValueError(f"Unknown dataset_name: {ds_name}. "
                           f"Supported: iiw, cgintrinsics, hypersim, saw")

        if len(ds) == 0:
            raise FileNotFoundError(f"No samples found for {ds_name} (split={split}). "
                                   f"Please check the dataset path and split files.")

        return DataLoader(ds, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, drop_last=False,
                          collate_fn=unified_collate)
    else:
        raise ValueError("Submission protocol supports only split='train' or split='val'")






class SAWDataset(Dataset):


    def __init__(self, root: str, split: str = "train", image_size: int = 256,
                 preprocessed_root: str = "preprocess_data"):
        self.image_size = image_size
        self._preprocessed = False
        self._samples = []
        self._raw_root = None

        _split = "test" if split == "val" else "train"


        processed_dir = Path(preprocessed_root) / "saw" / _split
        if processed_dir.exists():
            pt_files = sorted(processed_dir.glob("*.pt"))
            if _split == "test":
                pt_files = self._filter_by_split(
                    pt_files, root, _split, key_fn=lambda path: path.stem
                )
            if pt_files:
                self._preprocessed = True
                self._samples = pt_files
                return



        self._raw_root = Path(root)
        img_dir = self._raw_root / "saw_images_512"
        all_samples = sorted(img_dir.glob("*.png")) if img_dir.exists() else []
        self._samples = self._filter_by_split(all_samples, root, _split,
                                               key_fn=lambda p: p.stem)
        if not self._samples:
            print(f"Warning: No SAW images found in {img_dir} for split={split}")

    @staticmethod
    def _filter_by_split(samples: list, root: str, split: str,
                         key_fn) -> list:


        _split = split

        splits_dir = Path(root) / "saw_splits"


        npy_flat = splits_dir / f"{_split}_ids.npy"

        npy_sub = splits_dir / _split / "ids.npy"

        for npy_file in (npy_flat, npy_sub):
            if npy_file.exists():
                ids = np.load(npy_file)
                if ids.dtype.kind in ('i', 'u'):

                    valid_ints = {int(x) for x in ids.flat}
                    valid_strs = {str(int(x)).zfill(5) for x in ids.flat}
                    def _match(key: str) -> bool:
                        try:
                            return int(key) in valid_ints or key in valid_strs
                        except ValueError:
                            return key in valid_strs
                    return [s for s in samples if _match(key_fn(s))]
                else:
                    valid = {str(x).rsplit(".", 1)[0] for x in ids.flat}
                    return [s for s in samples if key_fn(s) in valid]


        txt_file = splits_dir / f"{_split}.txt"
        if txt_file.exists():
            with open(txt_file) as f:
                valid = {line.strip().replace(".png", "").replace(".jpg", "")
                        for line in f if line.strip()}
            return [s for s in samples if key_fn(s) in valid]


        if _split == "train":
            return samples
        else:
            return []

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        if not self._samples:
            raise RuntimeError(
                f"SAWDataset is empty (root={self._raw_root or 'preprocess_data/saw/train'}). "
                f"Please download the SAW dataset or run preprocess_data.py first.")

        if self._preprocessed:
            return _to_float32(torch.load(self._samples[idx], weights_only=False))


        img_path = self._samples[idx]
        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
        image = torch.from_numpy(image.transpose(2, 0, 1))
        image = torch.nn.functional.interpolate(
            image.unsqueeze(0), size=(self.image_size, self.image_size),
            mode="bilinear", align_corners=False).squeeze(0)




        labels_dir = self._raw_root / "saw_pixel_labels"
        saw_mask = torch.zeros(1, self.image_size, self.image_size)
        if labels_dir.exists():

            try:
                photo_id = int(img_path.stem)
            except ValueError:
                photo_id = None


            search_names = [img_path.stem]
            if photo_id is not None:
                search_names += [str(photo_id), f"{photo_id:05d}", f"{photo_id:06d}"]

            label_path = None
            for name in search_names:
                for ext in (".npy", ".png"):
                    candidates = sorted(labels_dir.rglob(f"{name}{ext}"))
                    if candidates:

                        for c in candidates:
                            if "filter_size_0" in str(c):
                                label_path = c
                                break
                        if label_path is None:
                            label_path = candidates[0]
                        break
                if label_path is not None:
                    break

            if label_path is not None:
                if label_path.suffix == ".npy":
                    label = np.load(label_path).astype(np.float32)

                else:
                    label = np.array(Image.open(label_path), dtype=np.float32) / 255.0
                if label.ndim == 2:
                    label = label[np.newaxis, :, :]
                elif label.ndim == 3:
                    label = label.transpose(2, 0, 1)
                saw_mask = torch.from_numpy(label)
                saw_mask = torch.nn.functional.interpolate(
                    saw_mask.unsqueeze(0), size=(self.image_size, self.image_size),
                    mode="nearest").squeeze(0)

        return {"image": image, "saw_mask": saw_mask,
                "filename": img_path.name}






def srgb_to_rgb(srgb: np.ndarray) -> np.ndarray:

    out = np.zeros_like(srgb)
    mask = srgb <= 0.04045
    out[mask] = srgb[mask] / 12.92
    out[~mask] = np.power((srgb[~mask] + 0.055) / 1.055, 2.4)
    return out






def unified_collate(batch: List[dict]) -> dict:

    if not batch:
        return {}


    def _stack(key: str) -> Optional[torch.Tensor]:
        vals = [s[key] for s in batch if key in s and s[key] is not None]
        if not vals:
            return None

        shapes = {v.shape for v in vals}
        if len(shapes) > 1:
            target = list(shapes)[0]
            vals = [torch.nn.functional.interpolate(
                v.unsqueeze(0) if v.dim() == 3 else v,
                size=target[-2:], mode="bilinear", align_corners=False
            ).squeeze(0) if v.dim() == 3 else v for v in vals]
        return torch.stack(vals, dim=0)


    collated = {
        "image":       _stack("image"),
        "reflectance": _stack("reflectance"),
        "shading":     _stack("shading"),
        "mask":        _stack("mask"),
        "judgements":  [s.get("judgements", []) for s in batch],
        "saw_mask":    _stack("saw_mask"),
        "filename":    [s.get("filename", "") for s in batch],
    }
    return collated
