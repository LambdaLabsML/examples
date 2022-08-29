from typing import Any, Dict

import lpips
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr
from PIL import Image
from torch.optim import Optimizer

from utils.metrics import ImageGrid


class SRLightningModule(pl.LightningModule):
    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__()
        self.params = params

        lpips_net = params.get('lpips_net', 'alex')
        l1_weight = params.get('l1_weight', 1)
        p_weight = params.get('p_weight', 1)
        l1_loss = nn.L1Loss()
        self.perceptual_loss = lpips.LPIPS(net=lpips_net)
        self.G_loss = lambda x, y: torch.sum(
            l1_loss(x, y)*l1_weight + self.perceptual_loss(x, y)*p_weight
        )
        self.val_ims = None

    def set_model(self, model: nn.Module) -> None:
        self.model = model

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        # print("BATCH: ", batch.keys())
        pred = self(batch['lq_img'])
        # print("PRED: ", pred.shape)
        g_loss = self.G_loss(pred, batch['hq_img'])
        # print("G_LOSS: ", g_loss)
        return g_loss

    def configure_optimizers(self) -> Optimizer:
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.params['lr'],
            betas=self.params['betas'],
            eps=self.params['eps'],
            weight_decay=self.params['weight_decay'],
        )
        lr_scheduler = lr.MultiStepLR(
            optimizer,
            milestones=self.params['milestones'],
            gamma=self.params['gamma']
        )
        return [optimizer], [lr_scheduler]

    def validation_step(self, batch, batch_index, dataloader_idx=0):
        pred = self.model(batch['lq_img'])
        g_loss = self.G_loss(pred, batch['hq_img'])

        self.val_ims = {name: ImageGrid() for name in (
            'hq_pred',
        )}
        self.val_ims['hq_pred'].update(pred)
        # Image.fromarray((pred[0]).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)).save(
        #     "/home/cll/Desktop/pred.png")

        self.log_dict({
            "val/g_loss": g_loss
        })
        return g_loss

    def on_validation_end(self):
        # reset all images metrics
        [metric.reset() for metric in self.val_ims.values()]
