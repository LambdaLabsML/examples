import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_only


class CustomModelCheckpoint(ModelCheckpoint):
    """Maintains a symlink to the current latest checkpoint

    Symlink is set to be 3 directories above checkpoint folder
    """
    def __init__(self, symlink_filename="last.ckpt", symlink_location="../../..", **kwargs):
        self.symlink_filename = symlink_filename
        self.symlink_location = symlink_location
        super().__init__(**kwargs)

    def _save_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
        """Create the symlink to last checkpoint"""
        super()._save_checkpoint(trainer, filepath)
        self._do_symlink(filepath)

    @rank_zero_only
    def _do_symlink(self, last_model_path):
        # If we are not using a logger (or one that doesn't write to checkpoints)
        # then we have to avoid doing anything that won't work
        symlink = Path(self.dirpath)/self.symlink_location/self.symlink_filename
        if not (Path(last_model_path).exists() and symlink.parent.exists()):
            return
        symlink.unlink(missing_ok=True)
        symlink.symlink_to(last_model_path)
        # log.info(f"Updated symlink to {last_model_path}")


class SanityCheckSkipperCallback(pl.Callback):
    """Only set ready after sanity check is finished"""

    ready: bool

    def __init__(self):
        super().__init__()
        self.ready = True

    def on_sanity_check_start(self, trainer, pl_module):
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True


class ChannelsLastCallback(pl.Callback):
    def on_train_batch_start(self, trainer: Trainer, pl_module, batch, batch_idx, unused=0):
        # TODO: this makes assumptions about the structure
        # of batch. We should probably standardize batch
        # structure, but that can wait until we are
        # consistently using multiple models besides AMP.
        for input_key, inputs in batch.items():
            batch[input_key] = {
                k: v.to(memory_format=torch.channels_last)
                for k, v in inputs.items()
            }


class PerformanceBenchmarkCallback(SanityCheckSkipperCallback):
    def __init__(self, benchmark_opts: Dict[str, Any]):
        super().__init__()
        self.benchmark_opts = benchmark_opts
        self.throughput_times = []
        self.steps_elapsed = 0

    def _benchmark(
        self,
        global_step: int,
        stage: str = "train"
    ) -> None:
        """
        Houses functions for benchmarking/measuring model training/inference
        performance.
        """
        curr_iter = global_step
        if self.benchmark_opts[stage]['throughput'] and curr_iter >= self.benchmark_opts['warmup']:
            freq = self.benchmark_opts[stage]['freq']

            if self.steps_elapsed % freq == 0 and self.steps_elapsed > 0:
                self._end_time = time.perf_counter()
                throughput_time_elapsed = abs(self._end_time - self._start_time)
                iterations_per_sec = (1 / throughput_time_elapsed * freq)
                self.throughput_times.append(iterations_per_sec)
                self.steps_elapsed = 0

            if self.steps_elapsed == 0:
                self._start_time = time.perf_counter()

            self.steps_elapsed += 1

    def _compute_benchmark_metrics(self, pl_module, stage: str) -> None:
        mean_throughput_key = stage + ":" + "mean(it/s)"
        std_throughput_key = stage + ":" + "std_dev(it/s)"

        if self.throughput_times:
            metrics_to_log = {
                f"metrics/{mean_throughput_key}": np.mean(self.throughput_times),
                f"metrics/{std_throughput_key}": np.std(self.throughput_times),
            }
            pl_module.log_dict(metrics_to_log, on_epoch=True, logger=True, rank_zero_only=True)

    def _reset_benchmark(self) -> None:
        """
        Performs any resetting necessary at the end of an epoch for benchmarking.
        """
        self.throughput_times = []
        self.steps_elapsed = 0

    @rank_zero_only
    def on_train_batch_start(self, trainer: Trainer, pl_module, batch, batch_idx, unused=0):
        self._benchmark(trainer.global_step, stage="train")

    @rank_zero_only
    def on_train_epoch_end(self, trainer: Trainer, pl_module) -> None:
        self._compute_benchmark_metrics(pl_module, stage="train")
        self._reset_benchmark()
