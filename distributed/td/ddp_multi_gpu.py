import os
import time
import numpy as np

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

# To override prepare batch method, we can make a partial collate_fn which can collate and make a batch of data.


class TorchDataset(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset.X[idx], self.dataset.y[idx], self.dataset.w[idx]


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
        print ('gpu id is ', gpu_id)
        self.device = gpu_id
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.prepare_batch = dc_model._prepare_batch
        self.mode = dc_model.mode
        self.task = dc_model.task
        self.forward = dc_model.loss_func
        self.model = DDP(dc_model.model, device_ids=[gpu_id])

    def _run_batch(self, batch_idx, batch):
        inputs, labels, w = self.prepare_batch(batch)
        self.optimizer.zero_grad()
        loss = self.forward(inputs, labels, w)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(
            f"[GPU{self.device}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}"
        )
        self.train_data.sampler.set_epoch(epoch)
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
            if self.device == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)


def load_train_objs():
    data_dir = 'datadir/delaney-featurized/GroverFeaturizer/ScaffoldSplitter/NormalizationTransformer_transform_y_True'
    train = dc.data.DiskDataset(data_dir=os.path.join(data_dir, 'train_dir'))
    train_set = TorchDataset(train)

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
    x, y, w = map(list, list(zip(*batch)))
    return [[x], [np.vstack(y)], [np.vstack(w)]]


def prepare_dataloader(dataset, batch_size):
    return DataLoader(dataset,
                      batch_size=batch_size,
                      pin_memory=True,
                      shuffle=False,
                      collate_fn=collate_fn,
                      sampler=DistributedSampler(dataset))


def main(rank: int, world_size: int, save_every: int, total_epochs: int,
         batch_size: int):
    ddp_setup(rank, world_size)
    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(model, train_data, optimizer, rank, save_every)
    trainer.train(total_epochs)
    dist.destroy_process_group()


if __name__ == '__main__':

    if torch.cuda.is_available():
        world_size = torch.cuda.device_count()
        backend = 'nccl'
    else:
        world_size = 1
        backend = 'gloo'

    mp.spawn(main, args=(world_size, 1, 5, 64), nprocs=world_size)
