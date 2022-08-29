from callbacks.default import get_logger, SanityCheckSkipperCallback
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger


class WandBCallback(SanityCheckSkipperCallback):
    """Helper for logging images at validation"""

    def on_validation_epoch_end(self, trainer: Trainer, pl_module):
        if not self.ready:
            return
        logger = get_logger(trainer, WandbLogger)
        experiment = logger.experiment

        to_log = {f"val/{k}": wandb.Image(v.compute())
                  for k, v in pl_module.val_ims.items()}
        to_log["trainer/global_step"] = trainer.global_step

        experiment.log(to_log)
