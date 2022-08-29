import utils.checkpoint as utils

from pathlib import Path
from typing import Dict, List, Tuple
import fnmatch
import logging

import hydra
from omegaconf import DictConfig, OmegaConf
from omegaconf.omegaconf import open_dict
import pytorch_lightning
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.loggers.base import LightningLoggerBase
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.utilities.distributed import rank_zero_only
from rich.pretty import pprint
import torch
import torchinfo
import cv2
cv2.setNumThreads(0)


log = logging.getLogger(__name__)


@rank_zero_only
def print_config(cfg, resolve=False):
    pprint(OmegaConf.to_container(cfg, resolve=resolve), expand_all=True)


@rank_zero_only
def log_zero(message, level="info"):
    log_fun = getattr(log, level)
    log_fun(message)


@rank_zero_only
def display_model_summary(
    cfg: DictConfig,
    model: torch.nn.Module,
) -> None:
    if 'info' in cfg and cfg.info.get('display_summary', False):
        torchinfo.summary(model,
                          tuple(cfg.info.summary_input_shape),
                          depth=cfg.info.summary_display_depth)


def load_from_checkpoint(cfg):
    """If find the checkpoint from `load_from` and return
    Also update the config to match that found in the checkpoint if needed
    (This is indicated by the cfg.model.resolution key being "missing")

    The full path to the found ckpt is stored in the ckpt_loaded key

    if the load_from key = `last` then look for the last symlink in the outputs/name
    directory. If found resolve the symlink (so we know what it was pointing to at the
    time and load this)
    """
    # This could be a .ckpt, a folder containing one
    if cfg.load_from == "last":
        # current dir will be outputs/name/version/timestamp
        name_dir = Path(".").absolute().parent.parent
        log_zero(f"Looking for last checkpoint in {name_dir}")
        expected_ckpt = name_dir/"last.ckpt"
        assert expected_ckpt.exists(), f"Could not find a last checkpoint in {expected_ckpt}"
        ckpt_file = expected_ckpt.resolve()
    else:
        ckpt_file = utils.find_checkpoint(Path(hydra.utils.to_absolute_path(cfg.load_from)),
                                          filename="last.ckpt")

    log_zero(f"Will load weights from: {ckpt_file}")
    with open_dict(cfg):
        cfg.ckpt_loaded = str(ckpt_file)

    # If the resolution is missing from model we should take it from the prev checkpoint
    if OmegaConf.is_missing(cfg.model, "resolution"):
        log_zero(f"Model resolution undefined, loading model def from {ckpt_file}")
        prev_config_file = ckpt_file.parent.parent/"logs"/"hparams.yaml"
        prev_config = OmegaConf.load(prev_config_file)
        cfg.model = OmegaConf.merge(prev_config.model, cfg.model)

    return ckpt_file


def load_ckpt_state(cfg, ckpt_file, lmodule) -> Tuple[int, int]:
    """Load state of models and optimisers from a checkpoint file
    If the cfg has an exclude_keys key a glob pattern will be used to
    remove certain keys from the loaded state dict before use.

    If the corresponding optimiser are found we will try and load the state
    of those too.

    state dicts are loaded with strict=False so that if there are non-matching keys
    these wont cause and error, but if keys have different shape tensors this will
    error.
    """
    log_zero(f"Loading weights from: {ckpt_file}")
    # Make sure to load on cpu (rather than device in ckpt)
    state = torch.load(ckpt_file, map_location="cpu")
    epoch = state.get('epoch', 0)
    global_step = state.get('global_step', 0)
    log_zero(f"Epoch: {epoch}")
    log_zero(f"Global step: {global_step}")
    log_zero(f"Loading weights from: {ckpt_file}")
    if cfg.exclude_keys:
        keys_to_exclude = fnmatch.filter(state["state_dict"].keys(), cfg.exclude_keys)
        log_zero(f"Excluding '{keys_to_exclude}' from state dict")
        for k in keys_to_exclude:
            state["state_dict"].pop(k)

    missing_keys, unexpected_keys = lmodule.load_state_dict(state["state_dict"], strict=False)
    if unexpected_keys:
        log_zero(f"Found the following unexpected keys:")
        for k in unexpected_keys:
            log_zero(f"\t: {k}")

    if cfg.exclude_keys and "optimizer_states" in cfg.exclude_keys:
        log_zero(f"excluding optimizer_state as specified in exclude_keys")
    else:
        if "optimizer_states" in state:
            log_zero(f"Found optimiser state")
            for optim, optim_state in zip(lmodule.configure_optimizers(),
                                          state["optimizer_states"]):
                log_zero(f"Loading optimser state")
                required_keys = utils.required_optim_state(type(optim))
                loaded_state_keys = optim_state['state'][0].keys()
                if required_keys == loaded_state_keys:
                    log_zero("states match, attempting to load")
                    try:
                        optim.load_state_dict(optim_state)
                    except Exception as e:
                        log_zero(f"Failed to load optimiser: {e}")
                else:
                    log_zero(f"Saved optimiser state ({loaded_state_keys}) does not match " +
                             "new class ({type(optim)}), not loading state")

    return epoch, global_step


def tidy_up(loggers: List[LightningLoggerBase]):
    """A space for any misc tidy up tasks.

    Seems like this is need to make wandb logger behave correctly during sweeps.
    """
    for lg in loggers:
        if isinstance(lg, pytorch_lightning.loggers.wandb.WandbLogger):
            lg.experiment.finish()


def merge_user_overrides(cfg: DictConfig):
    """To minimize syntax problems in user configs, provide them an
    "overrides:" section where they can use dot-notation for one-liners.

    This function will find and parse that section and merge into the config.
    """
    if "overrides" in cfg.keys():
        if cfg.overrides:
            # merge each dotlist override into the config
            with open_dict(cfg):
                for k, v in cfg.overrides.items():
                    OmegaConf.update(cfg, k, v)
                # remove the overrides group
                del cfg.overrides


def _build_model(
    model_cfg: DictConfig,
    use_channels_last: bool
) -> torch.nn.Module:
    model = hydra.utils.instantiate(model_cfg)
    if isinstance(model, DictConfig):
        print_config(model, resolve=True)
        model = hydra.utils.instantiate(model)
    if use_channels_last:
        model.to(memory_format=torch.channels_last)
    return model


def _build_losses(cfg: DictConfig) -> torch.nn.Module:
    return None


def _add_callbacks(callback_config: DictConfig, callbacks: List):
    for k, v in callback_config.items():
        if v and "_target_" in v:
            callbacks.append(hydra.utils.instantiate(v))


def _build_callbacks(cfg: DictConfig) -> List[Callback]:
    callbacks: List[Callback] = []
    if "callbacks" in cfg:
        _add_callbacks(cfg.callbacks, callbacks)
    return callbacks


def _build_loggers(cfg: DictConfig) -> List[LightningLoggerBase]:
    loggers: List[LightningLoggerBase] = []
    if "logger" in cfg:
        for key, lg_conf in cfg.logger.items():
            # TODO: fix. This if statement is janky
            if key == "callbacks":
                continue
            if lg_conf and "_target_" in lg_conf:
                loggers.append(hydra.utils.instantiate(lg_conf))
    return loggers


def _make_dict(x) -> Dict[str, int]:
    return {k: x for k in ('ready', 'completed', 'started', 'processed')}


def _set_epoch_step(trainer, epoch, global_step):
    """Manually load the progress states (for upgrade to pl 1.6)"""
    # This might give strange behaviour if you resume and change
    # datamodule.iterations_per_epoch but that's quite unusual

    epoch_state_dict = {
            'total': _make_dict(epoch),
            'current': _make_dict(epoch),
            }
    batch_state_dict = {
            'total': _make_dict(global_step),
            'current': _make_dict(0),
            'is_last_batch': False,
        }
    trainer.fit_loop.epoch_progress.load_state_dict(epoch_state_dict)
    trainer.fit_loop.epoch_loop.batch_progress.load_state_dict(batch_state_dict)

    optimizer_loop = trainer.fit_loop.epoch_loop.batch_loop.optimizer_loop
    existing_optim_state = optimizer_loop.optim_progress.state_dict()
    new_optim_state = {}
    for k, v in existing_optim_state.items():
        new_optim_state[k] = v
        if k == "optimizer":
            new_optim_state[k]["step"] = batch_state_dict
            new_optim_state[k]["zero_grad"] = batch_state_dict
    optimizer_loop.optim_progress.load_state_dict(new_optim_state)
    trainer.fit_loop.epoch_loop._batches_that_stepped = global_step


@hydra.main(config_path="configs", config_name="config.yaml")
def main(cfg: DictConfig):
    """Load the config and start a training run"""
    merge_user_overrides(cfg)
    print_config(cfg)

    ckpt_file = load_from_checkpoint(cfg) if cfg.checkpoint_path else None

    model = _build_model(cfg.arch, cfg.use_channels_last)
    data_module = hydra.utils.instantiate(cfg.datamodule)
    lmodule = hydra.utils.instantiate(
        cfg.lmodule,
        # optim={"params": model.parameters()},
    )
    lmodule.set_model(model)

    callbacks = _build_callbacks(cfg)
    loggers = _build_loggers(cfg)
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=loggers,
        _convert_="partial"
    )
    if trainer.logger:
        trainer.logger.log_hyperparams(cfg)

    if ckpt_file:
        epoch, global_step = load_ckpt_state(cfg, ckpt_file, lmodule)
        _set_epoch_step(trainer, epoch, global_step)

    display_model_summary(cfg, lmodule)
    log_zero(f"Starting training")
    trainer.fit(model=lmodule, datamodule=data_module)
    log_zero(f"Finished")

    tidy_up(loggers)

    return trainer


if __name__ == "__main__":
    main()
