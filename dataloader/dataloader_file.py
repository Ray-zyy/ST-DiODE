from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class SequenceStats:
    mean: np.ndarray
    std: np.ndarray


class SequenceNPYDataset(Dataset):
    """Dataset for sequences stored in one or multiple .npy files.

    Expected data shape after concatenation:
        [N, T, C, H, W]
    """

    def __init__(self, data: np.ndarray, input_length: int, stats: SequenceStats):
        if data.ndim != 5:
            raise ValueError(f'Expected data shape [N, T, C, H, W], but got {data.shape}.')
        if input_length <= 0 or input_length >= data.shape[1]:
            raise ValueError(
                f'input_length must be in [1, T-1]. Got input_length={input_length}, total_length={data.shape[1]}.'
            )

        self.data = torch.from_numpy(data).float()
        self.input_length = input_length
        self.mean = torch.as_tensor(stats.mean, dtype=torch.float32)
        self.std = torch.as_tensor(stats.std, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.data.shape[0])

    def __getitem__(self, index: int):
        sample = self.data[index]
        input_frames = sample[:self.input_length]
        output_frames = sample[self.input_length:]

        input_frames = (input_frames - self.mean) / self.std
        output_frames = (output_frames - self.mean) / self.std
        return input_frames, output_frames


def _load_npy_files(file_paths: Sequence[str | Path]) -> np.ndarray:
    arrays = [np.load(str(path)) for path in file_paths]
    if not arrays:
        raise ValueError('No .npy files were provided.')
    return np.concatenate(arrays, axis=0)


def _compute_stats(train_data: np.ndarray) -> SequenceStats:
    # Per-channel statistics over N, T, H, W
    mean = train_data.mean(axis=(0, 1, 3, 4), keepdims=True).astype(np.float32)
    std = train_data.std(axis=(0, 1, 3, 4), keepdims=True).astype(np.float32)
    std = np.maximum(std, 1e-6)
    return SequenceStats(mean=mean, std=std)


def _split_indices(num_samples: int, train_ratio: float, val_ratio: float, seed: int):
    generator = np.random.default_rng(seed)
    indices = generator.permutation(num_samples)

    train_end = int(num_samples * train_ratio)
    val_end = train_end + int(num_samples * val_ratio)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    if len(val_idx) == 0:
        val_idx = test_idx
    if len(test_idx) == 0:
        test_idx = val_idx

    return train_idx, val_idx, test_idx


def load_file(
    batch_size: int,
    val_batch_size: int,
    data_root: str,
    num_workers: int,
    input_length: int = 50,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 1,
):
    file_paths = [data_root]
    raw_data = _load_npy_files(file_paths)

    train_idx, val_idx, test_idx = _split_indices(
        num_samples=raw_data.shape[0],
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )

    train_data = raw_data[train_idx]
    val_data = raw_data[val_idx]
    test_data = raw_data[test_idx]

    stats = _compute_stats(train_data)

    train_dataset = SequenceNPYDataset(train_data, input_length=input_length, stats=stats)
    val_dataset = SequenceNPYDataset(val_data, input_length=input_length, stats=stats)
    test_dataset = SequenceNPYDataset(test_data, input_length=input_length, stats=stats)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        num_workers=num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=num_workers,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=num_workers,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader, stats.mean, stats.std
