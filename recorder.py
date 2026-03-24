from __future__ import annotations

import torch

from utils import get_model_state_dict


class Recorder:
    """Track the best validation loss and save the corresponding checkpoint."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.best_val_loss = float('inf')

    def __call__(self, val_loss: float, model: torch.nn.Module, logger, checkpoint_path: str) -> bool:
        if val_loss < self.best_val_loss:
            previous = self.best_val_loss
            self.best_val_loss = val_loss
            torch.save(get_model_state_dict(model), checkpoint_path)
            if self.verbose:
                logger.info(
                    'Validation loss improved from %.6f to %.6f. Saved best checkpoint to %s',
                    previous,
                    val_loss,
                    checkpoint_path,
                )
            return True
        return False
