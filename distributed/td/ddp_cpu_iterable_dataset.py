import os
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset

import deepchem as dc
from deepchem.models.torch_models import GroverModel
from deepchem.feat.vocabulary_builders import GroverAtomVocabularyBuilder, GroverBondVocabularyBuilder


class Trainer:

    def __init__(self, dc_model, train_data: DataLoader,
                 optimizer: torch.optim.Optimizer):
        self.train_data = train_data
        self.optimizer = optimizer
        self.pt_model = dc_model.model
        self.loss_fn = dc_model._loss_fn
        self.dc_model = dc_model

    def _run_batch(self, batch, batch_idx):
        inputs, labels, w = self.dc_model._prepare_batch(batch)
        self.optimizer.zero_grad()
        loss = self.loss_fn(inputs, labels, w)
        loss.backward()
        self.optimizer.step()
        if batch_idx % 16 == 0:
            print ('Loss in %d iteration is %0.3f' % (batch_idx, loss.item()))

    def _run_epoch(self):
        for i, batch in enumerate(self.train_data):
            self._run_batch(batch, i)

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch()


def load_train_objs():
    data_dir = 'datadir/delaney-featurized/GroverFeaturizer/ScaffoldSplitter/NormalizationTransformer_transform_y_True'
    train = dc.data.DiskDataset(data_dir=os.path.join(data_dir, 'train_dir'))
    train_set = train.make_pytorch_dataset(batch_size=16)

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
    x, y, w = batch[0][0], batch[0][1], batch[0][2]
    return [[x], [y], [w]]

def prepare_dataloader(dataset, batch_size):
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=False,
                      collate_fn=collate_fn,
                      num_workers=4)


def main(batch_size: int):
    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(model, train_data, optimizer)
    trainer.train(10)


if __name__ == '__main__':
    main(32)
