from typing import List, Union
import deepchem as dc
from deepchem.data.datasets import NumpyDataset
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import s3fs
import tempfile
import os

from chemberta3.utils import copy_file_from_s3
from abc import ABC, abstractmethod


class PretrainingDatasetLoader(ABC):
    """
    Abstract class for pretraining dataset loaders.
    """

    def __init__(self) -> None:
        pass

    @abstractmethod
    def _load_smiles_from_csv(self, csv_path: str, parallel: bool = True) -> np.ndarray:
        """
        Load shard from CSV file.

        Parameters
        ----------
        csv_path: str
            Path to CSV file.
        parallel: bool
            Whether to use parallel processing.

        Returns
        -------
        smiles: np.ndarray
            Array of SMILES strings.
        """
        pass

    @abstractmethod
    def load_shards(
        self,
        shards_to_load: Union[int, List[int]],
        cleanup: bool = True,
        parallel: bool = True,
    ) -> dc.data.DiskDataset:
        """
        Load specified number of shards of the dataset.

        Parameters
        ----------
        shards: Union[int, List[int]]
            Number of shards to load, or list of shard numbers to load.
        cleanup: bool
            Whether to delete local copies of shards after loading.
        parallel: bool
            Whether to use parallel processing.

        Returns
        -------
        dataset: dc.data.DiskDataset
            Loaded dataset.
        """
        pass


class ZincLoader(PretrainingDatasetLoader):
    """
    Load ZINC dataset from the cloud.
    """

    def __init__(self, featurizer: dc.feat.Featurizer) -> None:
        super().__init__()
        self.raw_cloud_dir: str = "s3://chemberta3/datasets/zinc20/csv/"
        self.local_data_dir: str = os.path.join(tempfile.gettempdir(), "zinc20")
        self.featurizer = featurizer

        self.shards_available = [
            "s3://" + s for s in s3fs.S3FileSystem().ls(self.raw_cloud_dir)
        ]
        self.chunk_size = int(1e4)

    def _load_smiles_from_csv(self, csv_path: str, parallel: bool = True) -> np.ndarray:
        """
        Load shard from CSV file.

        Parameters
        ----------
        csv_path: str
            Path to CSV file.

        Returns
        -------
        smiles: np.ndarray
            Array of SMILES strings.
        """
        df = pd.read_csv(csv_path)
        smiles = df["smiles"].values
        return smiles

    def load_shards(
        self,
        shards_to_load: Union[int, List[int]],
        cleanup: bool = True,
        parallel: bool = True,
    ) -> dc.data.DiskDataset:
        """
        Load specified number of shards of the ZINC dataset.

        Parameters
        ----------
        shards: Union[int, List[int]]
            Number of shards to load, or list of shard indices to load.
        """

        if isinstance(shards_to_load, int):
            shards = self.shards_available[:shards_to_load]
        elif isinstance(shards_to_load, list):
            shards = [self.shards_available[i] for i in shards_to_load]
        else:
            raise TypeError(
                "`shards` must be an integer (number of shards to load) or a list of shard indices"
            )

        def _shard_generator(shards):
            for s in shards:
                local_shard_path = copy_file_from_s3(s)
                smiles_arr = self._load_smiles_from_csv(
                    local_shard_path, parallel=parallel
                )
                if cleanup:
                    os.remove(local_shard_path)
                print(f"Loaded shard {s}")
                for i in range(0, len(smiles_arr), self.chunk_size):
                    print(f"Featurizing chunk {i}...")
                    smiles_chunk = smiles_arr[i : i + self.chunk_size]
                    X = Parallel(n_jobs=os.cpu_count() if parallel else 1, verbose=10)(
                        delayed(self.featurizer.featurize)(s) for s in smiles_chunk
                    )
                    X_arr = np.array([x[0] for x in X if x.size])
                    yield (X_arr, None, None, None)

        shard_generator = _shard_generator(shards)
        ds_disk = dc.data.DiskDataset.create_dataset(
            shard_generator, data_dir=self.local_data_dir
        )
        return ds_disk
