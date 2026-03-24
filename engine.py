from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import torch

from metrics import metric
from recorder import Recorder
from utils import load_checkpoint


class Engine:
    def __init__(
        self,
        args,
        train_loader,
        valid_loader,
        test_loader,
        scaler,
        model,
        optimizer,
        scheduler,
        criterion,
        logger,
        device,
    ):
        self.args = args
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.mean, self.std = scaler
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.logger = logger
        self.device = device

    @staticmethod
    def _reduce_loss(loss: torch.Tensor) -> torch.Tensor:
        return loss.mean() if isinstance(loss, torch.Tensor) and loss.ndim > 0 else loss

    def train(self):
        recorder = Recorder(verbose=True)

        for epoch in range(1, self.args.epochs + 1):
            self.model.train()
            train_losses = []

            for batch_x, batch_y in self.train_loader:
                batch_x = batch_x.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)

                self.optimizer.zero_grad(set_to_none=True)

                pred_y_det, agd_loss = self.model(batch_x, target_y=batch_y)
                recon_loss = self.criterion(pred_y_det, batch_y)
                total_loss = recon_loss + 0.1 * agd_loss
                total_loss = self._reduce_loss(total_loss)

                total_loss.backward()
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

                train_losses.append(float(total_loss.item()))

            train_loss = float(np.mean(train_losses)) if train_losses else 0.0

            if epoch % self.args.log_step == 0:
                val_loss, val_metrics = self.valid()
                mse_value, mae_value, ssim_value, psnr_value = val_metrics
                self.logger.info(
                    'Epoch [%d/%d] | Train Loss: %.6f | Val Loss: %.6f | '
                    'Val MSE: %.6f | Val MAE: %.6f | Val SSIM: %.6f | Val PSNR: %.6f',
                    epoch,
                    self.args.epochs,
                    train_loss,
                    val_loss,
                    mse_value,
                    mae_value,
                    ssim_value,
                    psnr_value,
                )
                recorder(val_loss, self.model, self.logger, self.args.checkpoint)

    @torch.no_grad()
    def valid(self) -> Tuple[float, Tuple[float, float, float, float]]:
        self.model.eval()

        preds_list, trues_list = [], []
        total_losses = []

        for batch_x, batch_y in self.valid_loader:
            batch_x = batch_x.to(self.device, non_blocking=True)
            batch_y = batch_y.to(self.device, non_blocking=True)

            pred_y = self.model(batch_x)
            loss = self.criterion(pred_y, batch_y)
            loss = self._reduce_loss(loss)
            total_losses.append(float(loss.item()))

            preds_list.append(pred_y.detach().cpu().numpy())
            trues_list.append(batch_y.detach().cpu().numpy())

        preds = np.concatenate(preds_list, axis=0)
        trues = np.concatenate(trues_list, axis=0)
        metrics_value = metric(preds, trues, self.mean, self.std, return_ssim_psnr=True)
        avg_loss = float(np.mean(total_losses)) if total_losses else 0.0

        self.model.train()
        return avg_loss, metrics_value

    @torch.no_grad()
    def test(self):
        if not self.args.checkpoint or not os.path.exists(self.args.checkpoint):
            raise FileNotFoundError(f'Checkpoint not found: {self.args.checkpoint}')

        load_checkpoint(self.model, self.args.checkpoint, self.device)
        self.model.eval()

        inputs_list, trues_list, preds_list = [], [], []

        for batch_x, batch_y in self.test_loader:
            batch_x = batch_x.to(self.device, non_blocking=True)
            pred_y = self.model(batch_x)

            inputs_list.append(batch_x.detach().cpu().numpy())
            trues_list.append(batch_y.detach().cpu().numpy())
            preds_list.append(pred_y.detach().cpu().numpy())

        inputs = np.concatenate(inputs_list, axis=0)
        trues = np.concatenate(trues_list, axis=0)
        preds = np.concatenate(preds_list, axis=0)

        mse_value, mae_value, ssim_value, psnr_value = metric(
            preds,
            trues,
            self.mean,
            self.std,
            return_ssim_psnr=True,
        )
        self.logger.info(
            'Test | MSE: %.6f | MAE: %.6f | SSIM: %.6f | PSNR: %.6f',
            mse_value,
            mae_value,
            ssim_value,
            psnr_value,
        )

        if self.args.is_save_data:
            np.save(os.path.join(self.args.log_dir, 'inputs.npy'), inputs)
            np.save(os.path.join(self.args.log_dir, 'trues.npy'), trues)
            np.save(os.path.join(self.args.log_dir, 'preds.npy'), preds)
            self.logger.info('Saved prediction arrays to %s', self.args.log_dir)
