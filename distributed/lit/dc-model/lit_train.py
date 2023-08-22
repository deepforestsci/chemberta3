import deepchem as dc
from deepchem.models.lightning.dc_lightning_module import DCLightningModule
from deepchem.models.lightning.dc_lightning_dataset_module import DCLightningDatasetModule, collate_dataset_wrapper
from deepchem.feat.vocabulary_builders import GroverAtomVocabularyBuilder, GroverBondVocabularyBuilder

import pytorch_lightning as pl

atom_vocab = GroverAtomVocabularyBuilder.load('data/delaney_atom_vocab.json')
bond_vocab = GroverBondVocabularyBuilder.load('data/delaney_bond_vocab.json')

model = dc.models.torch_models.GroverModel(node_fdim=151,
                        edge_fdim=165,
                        atom_vocab=atom_vocab,
                        bond_vocab=bond_vocab,
                        features_dim=2048,
                        hidden_size=128,
                        functional_group_size=85,
                        mode='regression',
                        task='finetuning',
                        model_dir='grover-model')

litmodel = DCLightningModule(model)

train_dataset = dc.data.DiskDataset(data_dir='data/delaney-featurized/GroverFeaturizer/ScaffoldSplitter/train_dir')
litdataset = DCLightningDatasetModule(train_dataset, batch_size=64)

trainer = pl.Trainer(max_epochs=3, devices=2, accelerator='gpu')
trainer.fit(litmodel, litdataset)
