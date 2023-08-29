import os
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

import deepchem as dc
from deepchem.models.torch_models import GroverModel
from deepchem.feat.vocabulary_builders import GroverAtomVocabularyBuilder, GroverBondVocabularyBuilder

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


def ddp_setup(rank, size, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    torch.cuda.set_device(rank)


class Trainer:

    def __init__(self, dc_model, train_data: DataLoader,
                 optimizer: torch.optim.Optimizer, gpu_id: int,
                 save_every: int) -> None:
        self.gpu_id = gpu_id
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.prepare_batch = dc_model._prepare_batch
        self.forward = dc_model.loss_func
        self.model = DDP(dc_model.model, device_ids=[gpu_id])

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

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = "checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)


def load_train_objs():
    data_dir = 'datadir/delaney-featurized/GroverFeaturizer/ScaffoldSplitter/NormalizationTransformer_transform_y_True'
    train = dc.data.DiskDataset(data_dir=os.path.join(data_dir, 'train_dir'))
    train_set = TorchDiskDataset(train, epochs=1, deterministic=True, batch_size=16)

    av = GroverAtomVocabularyBuilder.load('av.json')
    bv = GroverBondVocabularyBuilder.load('bv.json')

    model = GroverModel(node_fdim=151,
                        edge_fdim=165,
                        atom_vocab=av,
                        bond_vocab=bv,
                        hidden_size=128,
                        functional_group_size=85,
                        mode='regression',
                        features_dim=2048,
                        task='finetuning')
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


def main(rank: int, world_size: int, save_every: int, total_epochs: int,
         batch_size: int):
    ddp_setup(rank, world_size)
    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(model, train_data, optimizer, rank, save_every)
    trainer.train(total_epochs)
    dist.destroy_process_group()


if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    backend = 'nccl'
    mp.spawn(main, args=(world_size, 1, 5, 64), nprocs=world_size)
