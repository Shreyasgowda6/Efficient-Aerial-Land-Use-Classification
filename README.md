# Efficient Aerial Land-Use Classification

A computer vision project for classifying aerial land-use images with EuroSAT. It evaluates a ResNet-18 baseline and compares structured channel-pruning configurations for more efficient inference.

## Features

- EuroSAT image classification across 10 land-use classes
- Baseline accuracy and efficiency measurement
- Structured channel pruning with fine-tuning
- Comparison of model accuracy, size, compute, and latency
- Results recorded in a single CSV file

## Repository contents

- `load_eurosat_dense_checkpoint.py` — loads the model and EuroSAT data splits
- `baseline.py` — measures baseline accuracy and efficiency
- `prune_finetune.py` — trains the 25% channel-pruned model
- `evaluate_pruned.py` — evaluates the 25% model
- `prune_finetune_50pct.py` — trains the 50% channel-pruned model
- `evaluate_pruned_50pct.py` — evaluates the 50% model
- `baseline_results.csv` — comparison results for all model variants

## Setup

Use Python 3.12 and create a virtual environment from the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision ultralytics-thop torch-pruning tqdm
```

Large model checkpoint files are excluded from Git. Place the dense checkpoint in the repository root with this filename:

```text
eurosat_resnet18_dense_student_checkpoint.pt
```

The EuroSAT dataset downloads automatically to `data/` the first time it is loaded.

## Run

Run commands from the repository root.

Load the model and data:

```bash
python load_eurosat_dense_checkpoint.py
```

Measure the baseline:

```bash
python baseline.py
```

Train and evaluate the 25% pruned model:

```bash
python prune_finetune.py
python evaluate_pruned.py
```

Train and evaluate the 50% pruned model:

```bash
python prune_finetune_50pct.py
python evaluate_pruned_50pct.py
```

The evaluation scripts update the corresponding model variant in `baseline_results.csv`. Model weights and dataset files remain local and are not tracked by Git.

## Results

| Model | Test accuracy | Parameters | Raw weights | MACs per image | MPS latency |
|---|---:|---:|---:|---:|---:|
| Baseline | 98.44% | 11.18M | 42.69 MiB | 1,823.5M | 3.55 ms |
| Pruned 25% | 97.31% | 6.29M | 24.04 MiB | 1,049.7M | 2.97 ms |
| Pruned 50% | 96.10% | 2.80M | 10.71 MiB | 487.9M | 2.23 ms |

Operations per image are calculated as twice the MAC count.
