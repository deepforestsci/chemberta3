import os
import time
import torch
import torch.nn as nn
import torch.optim as optim

import deepchem as dc
from deepchem.models.torch_models import GroverModel
from deepchem.feat.vocabulary_builders import GroverAtomVocabularyBuilder, GroverBondVocabularyBuilder

os.environ['DEEPCHEM_DATA_DIR'] = 'datadir'

featurizer = dc.feat.DummyFeaturizer()
task, datasets, transformers = dc.molnet.load_delaney(featurizer=featurizer)
train, test, valid = datasets

av = GroverAtomVocabularyBuilder()
av.build(train)
av.save('av.json')
av = GroverAtomVocabularyBuilder.load('av.json')

bv = GroverBondVocabularyBuilder()
bv.build(train)
bv.save('bv.json')
bv = GroverBondVocabularyBuilder.load('bv.json')

featurizer = dc.feat.GroverFeaturizer(dc.feat.CircularFingerprint())
task, datasets, transformers = dc.molnet.load_delaney(featurizer=featurizer)
train, test, valid = datasets
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

time_per_epoch = []
for i in range(10):
    start = time.time()
    loss = model.fit(train, nb_epoch=1)
    end = time.time()
    time_per_epoch.append(end - start)
print(time_per_epoch)
