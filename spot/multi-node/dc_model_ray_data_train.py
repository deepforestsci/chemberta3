from typing import Dict, Any

import numpy as np
import deepchem as dc
from deepchem.models.torch_models import InfoGraphModel

from ray_ds import RayDataset


def collate_fn(batch: Dict[str, np.ndarray]) -> Any:
    return


if __name__ == '__main__':
    # dataset_path = 's3://chemberta3/ray_test/featurized_data/MolGraphConv/zinc250k/'
    dc_model = InfoGraphModel(num_features=30,
                              embedding_dim=11,
                              num_gc_layers=3,
                              task='pretraining',
                              learning_rate=0.0001)
    dc_model._ensure_built()
    optimizer = dc_model._pytorch_optimizer

    dataset_path = 'data/zinc1k-molgraphconv-feat'
    train_dataset = RayDataset.read(dataset_path).dataset
    # I need to prepare batches here for InfoGraph model and train it.
    for batch in train_dataset.iter_batches(batch_size=16):
        inputs, labels, weights = dc_model._prepare_batch(
            ([batch['x']], None, None))
        loss = dc_model.loss_func(inputs, labels, weights)
        loss.backward()
        optimizer.zero_grad()
        optimizer.step()
        print(loss)
