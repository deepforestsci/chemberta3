import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.strategies import DDPStrategy

import deepchem as dc
from deepchem.models.torch_models import GroverModel
from deepchem.feat.vocabulary_builders import GroverAtomVocabularyBuilder, GroverBondVocabularyBuilder

av = GroverAtomVocabularyBuilder.load('av.json')
bv = GroverBondVocabularyBuilder.load('bv.json')

data_dir = 'datadir/delaney-featurized/GroverFeaturizer/ScaffoldSplitter/NormalizationTransformer_transform_y_True'
train = dc.data.DiskDataset(data_dir=os.path.join(data_dir, 'train_dir'))

class LitModel(pl.LightningModule):
    def __init__(self, dc_model):
        super().__init__()
        self.dc_model = dc_model 
        self.pt_model = dc_model.model
        self.loss_fn = dc_model._loss_fn

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.pt_model.parameters(), lr=1e-4)
        # TODO Configure learning rate scheduler 
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        # return [optimizer], [lr_scheduler]
        return [optimizer]

    def training_step(self, batch, batch_idx):
        inputs, labels, w = self.dc_model._prepare_batch(batch)
        loss = self.loss_fn(inputs, labels, w) 
        self.log("train_loss", loss)
        return loss

def collate_fn(batch):
    X, y, w, ids = batch[0]
    return [[X], [y], [w]]

model = GroverModel(node_fdim=151,
                    edge_fdim=165,
                    atom_vocab=av,
                    bond_vocab=bv,
                    hidden_size=128,
                    functional_group_size=85,
                    mode='regression',
                    features_dim=2048,
                    task='finetuning',
                    model_dir='gm-finetune')


dataset = train.make_pytorch_dataset(batch_size=32)
dataloader = torch.utils.data.DataLoader(dataset, collate_fn=collate_fn)

csv_logger = CSVLogger("log_dir", name="grover", flush_logs_every_n_steps=1)
lit_model = LitModel(model)
trainer = pl.Trainer(max_epochs=100, accelerator='gpu', logger=csv_logger, devices=2,
        strategy='ddp_find_unused_parameters_true')
trainer.fit(lit_model, train_dataloaders=dataloader)
