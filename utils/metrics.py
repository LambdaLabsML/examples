from typing import Dict, List

import torch
from torchmetrics import Metric
from torchvision import utils as vision_utils


class ImageGrid(Metric):
    """A Metric used to store a grid of images.
    Updating the metrics appends more images to be displayed in the grid.
    Use this to store images to simplify the collection across multiple GPUs

    Pass in any kwargs applicable to torchvision.utils.make_grid
    """
    images: List[torch.Tensor]
    grid_args: Dict

    def __init__(self, dist_sync_on_step=False, **kwargs):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("images", default=[], dist_reduce_fx=None)
        self.grid_args = kwargs
        # Need the batch size to order images correctly
        self.batch_size = None

    def update(self, ims: torch.Tensor):
        """Adds a new image to the collection"""
        self.images.append(ims)
        if self.batch_size is None:
            self.batch_size = ims.shape[0]

    def compute(self) -> torch.Tensor:
        """Create a montage of all the images in the grid
        """
        # Sort the images to be consistently ordered
        sorted_ims = self._sort_ims(self.images)
        grid = vision_utils.make_grid(sorted_ims, **self.grid_args)
        return grid

    def _sort_ims(self, ims: List[torch.Tensor]) -> torch.Tensor:
        """Sorts the images to be in a consistent order

        ims comes in as a List of batch_size tensors. This is the batch
        from each process in the following order:
        [rank0-batch0, rank1-batch0, rank0-batch1, rank1-batch1, etc]

        Remember that the validation samples have bee distributed to the processes
        with pytorch's distirbuted sampler which alternates samples:
        in: 0, 1, 2, 3, 4, 5 -> rank1:(0, 2, 4), rank2:(1, 3, 5)

        to make comparisson easier we need to a bit of work to make sure the
        images all come out in the same order no matter the num gpus.
        """
        all_ims = torch.cat(ims, dim=0)
        if not torch.distributed.is_initialized():
            # Don't arrange if not distributed
            return all_ims

        num_procs = torch.distributed.get_world_size()
        num_total_batches = len(all_ims)//self.batch_size//num_procs

        indexes = list(range(all_ims.shape[0]))
        sorted_indexes = []
        for i in range(num_total_batches):
            for j in range(self.batch_size):
                start = i*self.batch_size*num_procs + j
                stop = (i+1)*self.batch_size*num_procs
                step = self.batch_size
                sub_ims = indexes[start:stop:step]
                sorted_indexes.extend(sub_ims)

        return all_ims[sorted_indexes]
