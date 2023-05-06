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

    def load_shards(
        self,
        shards_to_load: Union[int, List[int]],
        max_num_samples: int = None,
        cleanup: bool = True,
        parallel: bool = True,
    ) -> dc.data.DiskDataset:
        """
        Load specified number of shards of the dataset.

        Parameters
        ----------
        shards_to_load: Union[int, List[int]]
            Number of shards to load, or list of shard indices to load.
        max_num_samples: int
            Maximum number of samples to load, in case individual shards are too large.
        cleanup: bool
            Whether to delete the local shard files after loading.
        parallel: bool
            Whether to use parallel processing for featurization.
        """

        if isinstance(shards_to_load, int):
            shards = self.shards_available[:shards_to_load]
        elif isinstance(shards_to_load, list):
            shards = [self.shards_available[i] for i in shards_to_load]
        else:
            raise TypeError(
                "`shards` must be an integer (number of shards to load) or a list of shard indices"
            )
        if max_num_samples is not None and max_num_samples < self.chunk_size:
            print(
                "Warning: `max_num_samples` is smaller than `chunk_size`; `chunk_size` samples will be returned."
            )

        def _shard_generator(shards):
            total_samples_loaded = 0
            for s in shards:
                local_shard_path = copy_file_from_s3(s)
                smiles_arr = self._load_smiles_from_csv(
                    local_shard_path, parallel=parallel
                )
                if cleanup:
                    os.remove(local_shard_path)
                print(f"Loaded shard {s}")
                for i in range(0, len(smiles_arr), self.chunk_size):
                    if (
                        max_num_samples is not None
                        and total_samples_loaded >= max_num_samples
                    ):
                        break
                    print(f"Featurizing chunk {i}...")
                    smiles_chunk = smiles_arr[i : i + self.chunk_size]
                    X = Parallel(n_jobs=os.cpu_count() if parallel else 1, verbose=10)(
                        delayed(self.featurizer.featurize)(s) for s in smiles_chunk
                    )
                    X_arr = np.array([x[0] for x in X if x.size])
                    total_samples_loaded += X_arr.shape[0]
                    yield (
                        X_arr,
                        np.zeros((X_arr.shape[0], 1), np.float32),
                        np.zeros((X_arr.shape[0], 1), np.float32),
                        np.arange(
                            total_samples_loaded - X_arr.shape[0], total_samples_loaded
                        ),
                    )

        shard_generator = _shard_generator(shards)
        ds_disk = dc.data.DiskDataset.create_dataset(
            shard_generator, data_dir=self.local_data_dir
        )
        return ds_disk


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

    def _load_smiles_from_csv(self, csv_path: str) -> np.ndarray:
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


class PubchemLoader(PretrainingDatasetLoader):
    """
    Load PubChem dataset from the cloud.
    """

    def __init__(self, featurizer: dc.feat.Featurizer) -> None:
        super().__init__()
        self.raw_cloud_dir: str = "s3://chemberta3/datasets/pubchem/csv/"
        self.local_data_dir: str = os.path.join(tempfile.gettempdir(), "pubchem")
        self.featurizer = featurizer

        self.shards_available = [
            "s3://" + s for s in s3fs.S3FileSystem().ls(self.raw_cloud_dir)
        ]
        self.chunk_size = int(1e4)

    def _load_smiles_from_csv(self, csv_path: str) -> np.ndarray:
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
        smiles = df["canonical_smiles"].values
        return smiles
