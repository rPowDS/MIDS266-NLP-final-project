from __future__ import annotations
from datasets import load_dataset
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class DataConfig:
    name: str = 'ccdv/cnn_dailymail'
    config: str = '3.0.0'
    split_train: str = 'train'
    split_val: str = 'validation'
    split_test: str = 'test'

def load_cnndm(cfg: DataConfig | None = None):
    cfg = cfg or DataConfig()
    return load_dataset(cfg.name, cfg.config)
