from io import BytesIO
import argparse
import logging
import deepchem as dc
import numpy as np
import ray
from ray.data.block import Block, BlockAccessor
from ray.data.datasource.file_based_datasource import FileBasedDatasource
from ray.data.datasource.file_datasink import BlockBasedFileDatasink

from typing import Dict, Any, Optional, List, Iterator, Union
from functools import partial

logging.basicConfig(filename='ray.log', level=logging.INFO)

FEATURIZER_MAPPING = {
    "molgraphconv":
        dc.feat.MolGraphConvFeaturizer(use_edges=True),
    "dummy":
        dc.feat.DummyFeaturizer(),
    "grover":
        dc.feat.GroverFeaturizer(
            features_generator=dc.feat.CircularFingerprint()),
    "rdkit-conformer":
        dc.feat.RDKitConformerFeaturizer(),
    "snap":
        dc.feat.SNAPFeaturizer(),
}


class RayDataset(dc.data.Dataset):

    def __init__(self,
                 dataset: ray.data.Dataset,
                 x_column='x',
                 y_column: Optional[str] = None):
        self.dataset = dataset
        self.x_column, self.y_column = x_column, y_column

    def featurize(self, featurizer: Union[str, dc.feat.Featurizer], column):

        class RayFeaturizer:

            def __init__(self, featurizer, column):
                import deepchem as dc
                self.featurizer = featurizer
                self.column = column

            def __call__(self, batch):
                batch['x'] = self.featurizer(batch[self.column])
                return batch

        ray_featurizer = RayFeaturizer(featurizer, column)
        # Featurizing and dropping invalid SMILES strings
        self.dataset = self.dataset.map_batches(ray_featurizer).filter(lambda row: np.array(row['x']).size > 0)

    def write(self, path, columns):
        datasink = RayDcDatasink(path, columns)
        self.dataset.write_datasink(datasink)

    def iterbatches(self,
                    batch_size: int = 16,
                    epochs=1,
                    deterministic: bool = False,
                    pad_batches: bool = False):
        for batch in self.dataset.iter_batches(batch_size=batch_size,
                                               batch_format='numpy'):
            y = batch[self.y_column] if self.y_column else None
            x = batch[self.x_column]
            w, ids = np.ones(batch_size), np.ones(batch_size)
            yield (x, y, w, ids)

    @staticmethod
    def read(path) -> ray.data.Dataset:
        return RayDataset(ray.data.read_datasource(RayDcDatasource(path)))


def get_paths_from_args(args):
    result_path = 's3://chemberta3/featurized_data/' + args.dataset + '/' + args.featurizer
    if args.dataset == 'zinc100m':
        chunks = []
        for i in range(0, 10):
            chunks.append('chunk_00' + str(i) + '.csv')
        csv_paths = []
        for chunk_path in chunks:
            csv_paths.append('s3://chemberta3/datasets/zinc20/csv/' +
                             chunk_path)
        return csv_paths, result_path
    else:
        csv_path = 's3://chemberta3/datasets/' + args.dataset + '.csv'
        return csv_path, result_path


def test_is_empty_path(path):
    import boto3
    bucket_name = path.strip('s3://').split('/')[0]
    folder_name = path.strip('s3://' + bucket_name + '/')

    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket_name)
    count = bucket.objects.filter(Prefix=folder_name)
    assert len(list(count)) == 0, f'Result path is not empty {path}'

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset', type=str, default='zinc250k')
    argparser.add_argument('--featurizer', type=str, default='dummy')
    args = argparser.parse_args()

    csv_path, result_path = get_paths_from_args(args)
    test_is_empty_path(result_path)

    featurizer = FEATURIZER_MAPPING[args.featurizer]

    ds = ray.data.read_csv(csv_path, parallelism=100).repartition(10_000)
    ds = RayDataset(ds)

    ds.featurize(featurizer=featurizer, column='smiles')
    ds.write(result_path, columns=['x', 'smiles'])
    print('wrote data')
