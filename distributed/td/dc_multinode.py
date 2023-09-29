import os
import copy
import time
import numpy as np
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

import deepchem as dc
from deepchem.models.torch_models import GNNModular

class TorchDiskDataset(torch.utils.data.IterableDataset):

    def __init__(self,
                 disk_dataset: dc.data.DiskDataset,
                 epochs: int,
                 deterministic: bool = True,
                 batch_size: Optional[int] = None):
        self.disk_dataset = disk_dataset
        self.epochs = epochs
        self.deterministic = deterministic
        self.batch_size = batch_size

    def __len__(self):
        return len(self.disk_dataset)

    def __iter__(self):
        # Each time an iterator is created i.e when we call enumerate(dataloader),
        # num_worker number of worker processes get created.
        worker_info = torch.utils.data.get_worker_info()
        n_shards = self.disk_dataset.get_number_shards()
        if worker_info is None:
            process_id = 0
            num_processes = 1
        else:
            process_id = worker_info.id
            num_processes = worker_info.num_workers

        if dist.is_initialized():
            process_id += dist.get_rank() * num_processes
            num_processes *= dist.get_world_size()


        first_shard = process_id * n_shards // num_processes
        last_shard = (process_id + 1) * n_shards // num_processes

        if first_shard == last_shard:
            return

        # Last shard exclusive
        shard_indices = list(range(first_shard, last_shard))
        for X, y, w, ids in self.disk_dataset._iterbatches_from_shards(
                shard_indices,
                batch_size=self.batch_size,
                epochs=self.epochs,
                deterministic=self.deterministic):
            if self.batch_size is None:
                for i in range(X.shape[0]):
                    yield (X[i], y[i], w[i], ids[i])
            else:
                yield (X, y, w, ids)


def ddp_setup(backend='nccl'):
    if not torch.cuda.is_available():
        backend = 'gloo'
    dist.init_process_group(backend=backend)
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))


class Trainer:

    def __init__(self, dc_model, train_data: DataLoader,
                 optimizer: torch.optim.Optimizer, gpu_id: int,
                 save_every: int,
                 snapshot_path: str) -> None:
        self.local_rank = int(os.environ["LOCAL_RANK"])
        # global rank is set by torchrun
        self.global_rank = int(os.environ["RANK"])
        self.model = dc_model.model.to(self.local_rank)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.prepare_batch = dc_model._prepare_batch
        self.snapshot_path = snapshot_path

        # FIXME I feel something is not right here because we call self.forward in
        # _run_batch. Q: Do self.forward uses Trainer.model defined in next line or the
        # dc_model.model?
        self.forward = dc_model.loss_func
        self.model = DDP(dc_model.model, device_ids=[self.local_rank])

    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.local_rank}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def _run_batch(self, batch_idx, batch):
        inputs, labels, w = self.prepare_batch(batch)
        # Issue: Prepare batch method has got .to(device) calls which might
        # affect our training process
        # For each distributed process, we can create a new deepchem model object
        # whose prepare batch method we can use, thereby alleviating the problem of .to

        def _move_data_to_device(data):
            if isinstance(data, tuple):
                data = list(data)

            if isinstance(data, list):
                for i, input in enumerate(data):
                    if isinstance(input, torch.Tensor):
                        data[i] = input.to(self.gpu_id)
            elif isinstance(data, torch.Tensor):
                data = data.to(self.gpu_id)

            return data

        inputs = _move_data_to_device(inputs)
        labels = _move_data_to_device(labels)
        w = _move_data_to_device(w)

        self.optimizer.zero_grad()
        loss = self.forward(inputs, labels, w)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        # Why does it print batch size to be 1 always?
        b_sz = len(next(iter(self.train_data))[0])
        print(
            f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}"
        )
        # self.train_data.sampler.set_epoch(epoch)
        for i, batch in enumerate(self.train_data):
            self._run_batch(i, batch)

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if self.local_rank == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)


def load_train_objs():
    global_rank = os.environ['RANK']
    data_dir = os.path.join('data', 'zinc250k-snap-part' + global_rank)
    train = dc.data.DiskDataset(data_dir=os.path.join(data_dir, 'train_dir'))
    train_set = TorchDiskDataset(train, epochs=1, deterministic=True, batch_size=16)
    
    model = GNNModular(emb_dim = 8, task = "edge_pred")
    optimizer = torch.optim.Adam(model.model.parameters(), lr=1e-4)

    return train_set, model, optimizer


def collate_fn(batch):
    x, y, w, ids = batch[0][0], batch[0][1], batch[0][2], batch[0][3]
    return [[x], [y], [w]]


def prepare_dataloader(dataset, batch_size):
    return DataLoader(dataset,
                      batch_size=batch_size,
                      pin_memory=True,
                      shuffle=False,
                      collate_fn=collate_fn,
                      num_workers=4)


def main(save_every: int, total_epochs: int,
         batch_size: int, snapshot_path = 'snapshot.pt'):
    ddp_setup()
    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(model, train_data, optimizer, save_every, snapshot_path)
    trainer.train(total_epochs)
    destroy_process_group()


if __name__ == '__main__':
    main(10, 10, 100)
