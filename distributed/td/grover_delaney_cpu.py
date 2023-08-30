import os
import time
import torch
import torch.nn as nn
import torch.optim as optim

import deepchem as dc
from deepchem.models.torch_models import GroverModel
from deepchem.feat.vocabulary_builders import GroverAtomVocabularyBuilder, GroverBondVocabularyBuilder

os.environ['DEEPCHEM_DATA_DIR'] = 'datadir'

av = GroverAtomVocabularyBuilder.load('av.json')
bv = GroverBondVocabularyBuilder.load('bv.json')

train = dc.data.DiskDataset('datadir/delaney-featurized/GroverFeaturizer/ScaffoldSplitter/NormalizationTransformer_transform_y_True/train_dir')

model = GroverModel(node_fdim=151,
                    edge_fdim=165,
                    atom_vocab=av,
                    bond_vocab=bv,
                    hidden_size=128,
                    functional_group_size=85,
                    mode='regression',
                    features_dim=2048,
                    task='finetuning',
                    model_dir='gm-finetune',
                    device=torch.device('cpu'))

model.fit(train, nb_epoch=100)
