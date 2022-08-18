from pathlib import Path
from typing import Mapping

import pytorch_lightning.core.datamodule as pldata
import torch
from torch.utils import data
from torch.utils.data.distributed import DistributedSampler


# def get_equal_sampler(ds, key, num_samples) -> WeightedRandomSampler:
#     """
#     Returns a WeightedRandomSampler
#     """
#     metadata = ds.metadata
#     assert metadata is not None, "Dataset has no metadata"
#     assert key in metadata, f"Key '{key}' not found in metadata"
#
#     weight_key = f"{key}_weight"
#     if weight_key not in metadata:
#         add_weightings(metadata, key)
#
#     # FIXME ideally we should have already ensured that the metadata is
#     # ordered the same as the dataset
#     filenames = [x.name for x in ds.image_files]
#     weights = np.array(metadata.loc[filenames][weight_key])
#     sampler = WeightedRandomSampler(weights, num_samples)
#     return sampler


class SRDataModule(pldata.LightningDataModule):
    def __init__(
        self,
        train_dataset: data.Dataset,
        val_dataset: data.Dataset,
        batch_size: int = 8,
        num_workers: int = 2,
        num_val_workers: int = 2,
        iterations_per_epoch: int = 1000,
        use_random_sampler: bool = False,
    ):
        super().__init__()

        self.train_dataset = train_dataset
        if isinstance(val_dataset, Mapping):
            val_dataset = self._create_val_from_train(val_dataset)
        elif isinstance(val_dataset, (Path, str)):
            val_dataset = self._create_val_from_path(val_dataset)

        self.val_dataset = val_dataset

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_val_workers = num_val_workers
        self.use_random_sampler = use_random_sampler

        self.num_samples = iterations_per_epoch*batch_size

    def train_dataloader(self) -> data.DataLoader:
        sampler = None
        if self.use_random_sampler:
            sampler = data.RandomSampler(
                self.dataset,
                replacement=True,
                num_samples=self.num_samples,
            )
        data_loader = data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=sampler,
            drop_last=True,
            pin_memory=True,
        )
        return data_loader

    def val_dataloader(self):
        if torch.distributed.is_initialized():
            sampler = DistributedSampler(self.val_dataset, shuffle=False, drop_last=False)
        else:
            sampler = None

        val_loader = data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_val_workers,
            drop_last=False,
            pin_memory=True,
            sampler=sampler,
            shuffle=False,
        )
        return val_loader
