from __future__ import annotations

from dataloader_file import load_file


def _lazy_import(name: str):
    module = __import__(name, fromlist=['dummy'])
    return module


def load_data(dataname, batch_size, val_batch_size, data_root, num_workers, in_shape=None,
              train_ratio=0.8, val_ratio=0.1, seed=1, **kwargs):
    """Unified dataset factory.

    Only the requested dataset branch is imported, so missing optional files do not
    break unrelated experiments.
    """
    input_length = in_shape[0] if in_shape is not None else 50

    if dataname == 'taxibj':
        module = _lazy_import('utils.dataloader_taxiBJ')
        return module.load_taxibj(batch_size, val_batch_size, data_root, num_workers)

    if dataname == 'mmnist':
        module = _lazy_import('utils.dataloader_moving_mnist')
        return module.load_moving_mnist(batch_size, val_batch_size, data_root, num_workers)

    if dataname == 'navier':
        module = _lazy_import('utils.dataloader_navier')
        return module.load_navier(batch_size, val_batch_size, data_root, num_workers)

    if dataname == 'kth':
        module = _lazy_import('utils.dataloader_kth')
        return module.load_kth(batch_size, val_batch_size, data_root, num_workers)

    if dataname == 'sevir':
        module = _lazy_import('utils.dataloader_sevir')
        return module.load_kth(batch_size, val_batch_size, data_root, num_workers)

    raise ValueError(f'Unsupported dataset: {dataname}')
