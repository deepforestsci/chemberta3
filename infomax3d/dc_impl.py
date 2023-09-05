import deepchem as dc
import torch
from deepchem.models.torch_models import infomax3d, infomax3d_layers
from deepchem.models.torch_models.gnn3d import InfoMax3DModular

import logging

logging.basicConfig(filename='losses2.log', level=logging.INFO)

from deepchem.data import CSVLoader
from deepchem.feat import RDKitConformerFeaturizer

filepath = 'data/zinc1k.csv'

featurizer = RDKitConformerFeaturizer(num_conformers=1)
loader = CSVLoader(tasks=['logp'],
                   feature_field='smiles',
                   featurizer=featurizer)
dataset = loader.create_dataset(filepath)

dataset.move('datadir/zinc1k')

dataset = dc.data.DiskDataset(data_dir='datadir/zinc1k')
all_losses = []

model = InfoMax3DModular(device=torch.device('cpu'),
                         log_frequency=10,
                         all_losses=all_losses,
                         batch_size=64,
                         hidden_dim=64,
                         target_dim=10,
                         aggregators=['mean'],
                         readout_aggregators=['mean'],
                         scalers=['identity'])

loss1 = model.fit(dataset, nb_epoch=1)
