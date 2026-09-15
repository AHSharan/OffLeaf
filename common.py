"""Small shared helpers: config loading, seeding, run directories, CSV logging.

This module is a deliberate addition to the spec's tree. Seeding and run-directory
layout are correctness-critical and used by ``train/``, ``masks/``,
``counterfactual/`` and ``eval/`` alike; having one implementation avoids the
classic reproducibility bug where two entry points seed differently.
"""

from __future__ import annotations

import csv
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent
RUNS_ROOT = REPO_ROOT / "runs"


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a yaml config.

    Raises:
        FileNotFoundError: If the config does not exist.
        ValueError: If it does not parse to a mapping.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with open(p, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config {p} must be a mapping, got {type(cfg).__name__}")
    return cfg


def save_config(cfg: dict[str, Any], path: str | Path) -> None:
    """Write the exact config a run used, next to its outputs."""
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed every RNG this repo touches.

    Args:
        seed: The seed, always taken from ``config.seed``.
        deterministic: Also pin cuDNN into deterministic mode. This costs
            throughput but makes ``test_lambda_zero_equals_baseline`` meaningful.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn`` so augmentation is reproducible per worker."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Return the CUDA device when available, else CPU."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_dir(exp_id: str, seed: int, create: bool = True) -> Path:
    """``runs/<exp_id>/<seed>/`` - the one place a run may write."""
    d = RUNS_ROOT / exp_id / str(seed)
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def write_metrics(metrics: dict[str, Any], path: str | Path) -> None:
    """Write ``metrics.json``."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, sort_keys=True)


class CSVLogger:
    """Append-only CSV logger. The default logging backend; W&B is behind a flag.

    The header is written from the first row's keys; later rows must match.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._fields: list[str] | None = None

    def log(self, row: dict[str, Any]) -> None:
        """Append one row.

        Raises:
            ValueError: If the row's keys differ from the established header.
        """
        new = self._fields is None
        if new:
            self._fields = list(row.keys())
        elif list(row.keys()) != self._fields:
            raise ValueError(
                f"CSV row keys changed.\n  expected: {self._fields}\n  got:      {list(row.keys())}"
            )
        with open(self.path, "a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=self._fields)
            if new:
                w.writeheader()
            w.writerow(row)
