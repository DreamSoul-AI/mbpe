"""Download official CIFAR-10 into the repo data directory. No tokenizer run."""

from __future__ import annotations

import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import torch

torch.set_num_threads(8)

from torchvision import datasets

ROOT = Path(__file__).resolve().parents[1] / "data"
train = datasets.CIFAR10(str(ROOT), train=True, download=True)
test = datasets.CIFAR10(str(ROOT), train=False, download=True)
image, label = train[0]
print(f"train={len(train)} test={len(test)} image={image.size} mode={image.mode} label={label}", flush=True)
