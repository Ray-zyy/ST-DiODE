from __future__ import annotations

import random
from functools import partial
from itertools import repeat
from typing import Callable

import numpy as np
import torch
import torch.utils.data

try:
    from timm.data.distributed_sampler import OrderedDistributedSampler, RepeatAugSampler
except ImportError:  # pragma: no cover - optional dependency
    OrderedDistributedSampler = None
    RepeatAugSampler = None


def worker_init(worker_id, worker_seeding='all'):
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        return

    if isinstance(worker_seeding, Callable):
        seed = worker_seeding(worker_info)
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed % (2 ** 32 - 1))
        return

    if worker_seeding not in {'all', 'part'}:
        raise ValueError("worker_seeding must be 'all', 'part', or a callable")

    if worker_seeding == 'all':
        np.random.seed(worker_info.seed % (2 ** 32 - 1))


def expand_to_chs(x, n):
    if not isinstance(x, (tuple, list)):
        x = tuple(repeat(x, n))
    elif len(x) == 1:
        x = tuple(x) * n
    else:
        assert len(x) == n, 'Normalization stats must match image channels.'
    return x


class PrefetchLoader:
    def __init__(self, loader, mean=None, std=None, channels=3, fp16=False):
        self.loader = loader
        self.fp16 = fp16
        self.device = torch.device('cuda')

        if mean is not None and std is not None:
            mean = expand_to_chs(mean, channels)
            std = expand_to_chs(std, channels)
            norm_shape = (1, channels, 1, 1)

            self.mean = torch.tensor([x * 255 for x in mean], device=self.device).view(norm_shape)
            self.std = torch.tensor([x * 255 for x in std], device=self.device).view(norm_shape)
            if fp16:
                self.mean = self.mean.half()
                self.std = self.std.half()
        else:
            self.mean = None
            self.std = None

    def __iter__(self):
        if not torch.cuda.is_available():
            yield from self.loader
            return

        stream = torch.cuda.Stream()
        first = True
        current_input = None
        current_target = None

        for next_input, next_target in self.loader:
            with torch.cuda.stream(stream):
                next_input = next_input.cuda(non_blocking=True)
                next_target = next_target.cuda(non_blocking=True)

                if self.fp16:
                    next_input = next_input.half()
                    next_target = next_target.half()
                else:
                    next_input = next_input.float()
                    next_target = next_target.float()

                if self.mean is not None:
                    next_input = next_input.sub_(self.mean).div_(self.std)
                    next_target = next_target.sub_(self.mean).div_(self.std)

            if not first:
                yield current_input, current_target
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            current_input, current_target = next_input, next_target

        if current_input is not None:
            yield current_input, current_target

    def __len__(self):
        return len(self.loader)

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def dataset(self):
        return self.loader.dataset


def create_loader(
    dataset,
    batch_size,
    shuffle=True,
    is_training=False,
    mean=None,
    std=None,
    num_workers=1,
    num_aug_repeats=0,
    input_channels=1,
    use_prefetcher=False,
    distributed=False,
    pin_memory=False,
    drop_last=False,
    fp16=False,
    collate_fn=None,
    persistent_workers=True,
    worker_seeding='all',
):
    sampler = None
    is_iterable = isinstance(dataset, torch.utils.data.IterableDataset)

    if distributed and not is_iterable:
        if is_training:
            if num_aug_repeats:
                if RepeatAugSampler is None:
                    raise ImportError('timm is required for RepeatAugSampler.')
                sampler = RepeatAugSampler(dataset, num_repeats=num_aug_repeats)
            else:
                sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            if OrderedDistributedSampler is None:
                raise ImportError('timm is required for OrderedDistributedSampler.')
            sampler = OrderedDistributedSampler(dataset)
    else:
        assert num_aug_repeats == 0, 'Repeat augmentation requires distributed training.'

    if collate_fn is None:
        collate_fn = torch.utils.data.dataloader.default_collate

    loader_args = dict(
        batch_size=batch_size,
        shuffle=shuffle and (not is_iterable) and sampler is None and is_training,
        num_workers=num_workers,
        sampler=sampler,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last,
        worker_init_fn=partial(worker_init, worker_seeding=worker_seeding),
        persistent_workers=persistent_workers and num_workers > 0,
    )

    try:
        loader = torch.utils.data.DataLoader(dataset, **loader_args)
    except TypeError:
        loader_args.pop('persistent_workers', None)
        loader = torch.utils.data.DataLoader(dataset, **loader_args)

    if use_prefetcher:
        loader = PrefetchLoader(loader, mean=mean, std=std, channels=input_channels, fp16=fp16)

    return loader
