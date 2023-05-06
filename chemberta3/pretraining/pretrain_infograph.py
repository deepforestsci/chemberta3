import os
import tempfile

from deepchem.feat import MolGraphConvFeaturizer
from deepchem.models.torch_models import InfoGraphModel

from chemberta3.pretraining.loaders import ZincDatasetLoader

featurizer = MolGraphConvFeaturizer(use_edges=True)
loader = ZincDatasetLoader(featurizer=featurizer)

ds = loader.load_shards(1, cleanup=True, parallel=True)

num_feat = max([ds.X[i].num_node_features for i in range(min(len(ds), 1e5))])
edge_dim = max([ds.X[i].num_edge_features for i in range(min(len(ds), 1e5))])


model = InfoGraphModel(
    num_feat, edge_dim, 15, use_unsup_loss=True, separate_encoder=False
)

loss = model.fit(ds, nb_epoch=1)
