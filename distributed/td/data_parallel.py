import os
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

import deepchem as dc
from deepchem.models.torch_models import GroverModel
from deepchem.feat.vocabulary_builders import GroverAtomVocabularyBuilder, GroverBondVocabularyBuilder

av = GroverAtomVocabularyBuilder.load('av.json')
bv = GroverBondVocabularyBuilder.load('bv.json')

data_dir='datadir/delaney-featurized/GroverFeaturizer/ScaffoldSplitter/NormalizationTransformer_transform_y_True/train_dir'
train = dc.data.DiskDataset(data_dir=data_dir)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
                    device=device)
model.model = nn.DataParallel(model.model)
time_per_epoch = []
for i in range(10):
    start = time.time()
    loss = model.fit(train, nb_epoch=1)
    end = time.time()
    time_per_epoch.append(end - start)
print('Average time taken is {:.3f}'.format(np.mean(time_per_epoch)))
