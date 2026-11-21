# CDBF-Net

Official code and pretrained model for CDBF-Net.

## Setup

```bash
git clone https://github.com/hez8/CDBFNet.git
cd CDBFNet
git lfs pull

conda create -n py39 python=3.9 -y
conda activate py39
pip install -r requirements.txt
```

## Data

Set the dataset paths in `config/default.yaml`. Preprocessed samples follow this
layout:

```text
preprocess_data/<dataset>/train/*.pt
preprocess_data/<dataset>/test/*.pt
```

## Evaluation

Evaluate all supported datasets with the included checkpoint:

```bash
python main.py --config config/default.yaml evaluate --dataset all
```

To evaluate one dataset, replace `all` with `iiw`, `cgintrinsics`, `hypersim`,
or `saw`. A different checkpoint can be selected with `--checkpoint`.

## Training

```bash
python main.py --config config/default.yaml train
```

Resume from a checkpoint with `--resume`.

## Inference

```bash
python main.py --config config/default.yaml infer \
  --image input.png \
  --output results/inference
```
