import logging
import os
import random
from datetime import datetime
from typing import Dict, Any

import numpy as np
import torch
import torch.backends.cudnn as cudnn


def init_seed(seed: int) -> None:
    """Set all common random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def get_log_dir(work_dir: str, model_name: str, dataset_name: str) -> str:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return os.path.join(work_dir, model_name, dataset_name, timestamp)


def get_logger(log_dir: str, name: str = 'train', debug: bool = False) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', '%Y-%m-%d %H:%M:%S')

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if debug else logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(os.path.join(log_dir, 'run.log'), mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, 'module') else model


def get_model_state_dict(model: torch.nn.Module) -> Dict[str, Any]:
    return unwrap_model(model).state_dict()


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: torch.device):
    """Load checkpoint while handling both single-GPU and DataParallel state dicts."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    target_model = unwrap_model(model)

    try:
        target_model.load_state_dict(checkpoint)
        return
    except RuntimeError:
        pass

    if isinstance(checkpoint, dict):
        if any(k.startswith('module.') for k in checkpoint.keys()):
            checkpoint = {k.replace('module.', '', 1): v for k, v in checkpoint.items()}
        else:
            checkpoint = {f'module.{k}': v for k, v in checkpoint.items()}

    try:
        target_model.load_state_dict({k.replace('module.', '', 1): v for k, v in checkpoint.items()})
    except RuntimeError:
        model.load_state_dict(checkpoint)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
