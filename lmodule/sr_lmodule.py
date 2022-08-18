from typing import Any, Dict

import lpips
import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr
from torch.optim import Optimizer


class SRLightningModule(pl.LightningModule):
    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__()
        self.params = params

        lpips_net = params.get('lpips_net', 'alex')
        l1_weight = params.get('l1_weight', 1)
        p_weight = params.get('p_weight', 1)
        l1_loss = nn.L1Loss()
        perceptual_loss = lpips.LPIPS(net=lpips_net)
        self.G_loss = lambda x, y: l1_loss(x['hq_img'], y)*l1_weight + \
            perceptual_loss(x['hq_img'], y)*p_weight

    def set_model(self, model: nn.Module) -> None:
        self.model = model

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        print("BATCH: ", batch.keys())
        pred = self(batch['lq_img'])
        g_loss = self.G_loss(pred, batch['hq_img'])
        print("G_LOSS: ", g_loss)
        exit()
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

    def check_validation_set(self, batch, batch_index):
        print("HERE check_validation_set")
