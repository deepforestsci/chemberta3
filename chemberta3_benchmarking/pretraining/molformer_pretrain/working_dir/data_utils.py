import ray
import deepchem as dc
import numpy as np
from io import BytesIO
from typing import List, Optional, Union, Dict, Any, Iterator

from ray.data.block import Block, BlockAccessor
from ray.data.datasource.file_based_datasource import FileBasedDatasource
from ray.data.datasource.file_datasink import BlockBasedFileDatasink

from functools import partial


class _RayDcDatasource(FileBasedDatasource):
    """Ray Datasource

    A datasource which reads data stored as npz files.

    Same as
    https://github.com/ray-project/ray/blob/8b73f185fa51d95c95f183778e3ecc716cfdffc0/python/ray/data/datasource/numpy_datasource.py#L15
    except that this can read multiple coulmns of value (x, y, ...) where as
    Ray's implementation of Numpy datasource can read only a single column.
    See: https://github.com/ray-project/ray/issues/43011
    """
    _FILE_EXTENSIONS = ["npz"]

    def __init__(
        self,
        paths: Union[str, List[str]],
        numpy_load_args: Optional[Dict[str, Any]] = None,
        **file_based_datasource_kwargs,
    ):
        """
        Parameters
        ----------
        paths: Union[str, List[str]]
            Path to dataset
        numpy_load_args: Optional[Dict]
            Arguments for numpy.load
        See: https://numpy.org/doc/stable/reference/generated/numpy.load.html
        """
        super().__init__(paths, **file_based_datasource_kwargs)

        if numpy_load_args is None:
            numpy_load_args = {}

        self.numpy_load_args = numpy_load_args

    def _read_stream(
            self,
            f: "pyarrow.NativeFile",  # noqa: F821
            path: str) -> Iterator[Block]:
        """Reads a stream of data"""
        buf = BytesIO()
        data = f.readall()
        buf.write(data)
        buf.seek(0)
        data = dict(np.load(buf, allow_pickle=True, **self.numpy_load_args))
        yield BlockAccessor.batch_to_block(data)


class _RayDcDatasink(BlockBasedFileDatasink):
    """Ray Datasink

    A datasink which is used to store featurized data as npz files.

    Same as
    https://github.com/ray-project/ray/blob/8b73f185fa51d95c95f183778e3ecc716cfdffc0/python/ray/data/datasource/numpy_datasource.py#L15
    except that this can write multiple coulmns of values (x, y, ...) where as
    Ray's implementation of Numpy datasink can read only a single column.

    _RayDcDatasink is used by RayDataset to read and write custom formats of
    data.
    """

    def __init__(
        self,
        path: str,
        columns: List[str],
        *,
        file_format: str = "npz",
        **file_datasink_kwargs,
    ):
        """
        Parameters
        ----------
        paths: Union[str, List[str]]
            Path to dataset
        columns: List[str]
            Columns to store in disk
        """
        super().__init__(path, file_format=file_format, **file_datasink_kwargs)

        self.columns = columns

    def write_block_to_file(self, block: BlockAccessor,
                            file: "pyarrow.NativeFile"):  # noqa: F821
        """Writes a block of data"""
        data = {}
        for column in self.columns:
            data[column] = block.to_numpy(column)
        np.savez(file, **data)


class RayDataset(dc.data.Dataset):
    """RayDataset

    RayDataset is a sub-class of DeepChem dataset.
    It accepts any ray.data.Dataset and adds utilities in it to:
    - featurize the dataset via a DeepChem featurizer
    - store the dataset as a npz file by using _RayDcDatasink
    - provides `iterbatches` which allows a RayDataset to be
    trained with DeepChem models.

    Example
    -------
    >>> csv_dataset = ray.data.read_csv('assets/zinc5k.csv')
    >>> ray_dataset = RayDataset(csv_dataset, x_column='smiles', y_column='logp')
    >>> ray_dataset.featurize(featurizer=dc.feat.DummyFeaturizer())
    >>> featurized_dataset_path = Path(tmpdir).joinpath('out')
    >>> columns = ['x', 'smiles', 'logp']
    >>> ray_dataset.write(featurized_dataset_path, columns)
    >>> read_dataset = RayDataset.read(featurized_dataset_path)
    """

    def __init__(self,
                 dataset: ray.data.Dataset,
                 x_column='x',
                 y_column: Optional[str] = None):
        """
        Parameters
        ----------
        dataset: ray.data.Dataset
            A ray dataset
        x_column: str
            The column name in the dataset containing the inputs for the model
        during training.
        y_column: Optional[str]
            The column name in the dataset containing the outputs for the model
        during model training.

        Note
        ----
        We can directly specify x_column, y_column as arguments in
        `iterbatches` but doing so will make it incompatible for use with
        DeepChem's TorchModel/KerasModel. Because DeepChem has not got a way
        to pass x_column, y_column in TorchModel as it iterates through the
        dataset.
        """
        self.dataset = dataset
        self.x_column, self.y_column = x_column, y_column

    def featurize(self,
                  featurizer: Union[str, dc.feat.Featurizer],
                  column: Optional[str] = None):
        """Featurizer

        Featurizes the RayDataset

        Parameters
        ----------
        featurizer: dc.feat.Featurizer
            A DeepChem featurizer
        column: str
            The column to featurize
        """
        if column is None:
            column = self.x_column

        def _featurize(batch: Dict[str, np.ndarray], x_column: str,
                       featurizer: dc.feat.Featurizer):
            batch['x'] = featurizer(batch[x_column])
            return batch

        ray_featurizer = partial(_featurize,
                                 x_column=column,
                                 featurizer=featurizer)
        # Featurizing and dropping invalid SMILES strings
        self.dataset = self.dataset.map_batches(ray_featurizer).filter(
            lambda row: np.array(row['x']).size > 0)

    def write(self, path: str, columns: List[str]):
        """Write dataset to the path

        Parameters
        ----------
        path: str
            Path to store the dataset
        columns: List[str]
            columns of the dataset to write
        """
        print('\n\n going to write data \n\n')
        datasink = _RayDcDatasink(path, columns)
        self.dataset.write_datasink(datasink)

    def iterbatches(self,
                    batch_size: int = 16,
                    epochs=1,
                    deterministic: bool = False,
                    pad_batches: bool = False):
        """Get an object that iterates over minibatches from the dataset.

        Each minibatch is returned as a tuple of four numpy arrays:
        `(X, y, w, ids)`.

        Parameters
        ----------
        batch_size: int, optional (default 16)
            Number of elements in each batch.
        epochs: int, default 1
            Number of epochs to walk over dataset.
        deterministic: bool, optional (default False)
            If True, follow deterministic order.
        pad_batches: bool, optional (default False)
            If True, pad each batch to `batch_size`.

        Returns
        -------
        Iterator[Batch]
            Generator which yields tuples of four numpy arrays `(X, y, w, ids)`.
        """

        for batch in self.dataset.iter_batches(batch_size=batch_size,
                                               batch_format='numpy'):
            y = batch[self.y_column] if self.y_column else None
            x = batch[self.x_column]
            w, ids = np.ones(batch_size), np.ones(batch_size)
            yield (x, y, w, ids)

    @staticmethod
    def read(path: str) -> ray.data.Dataset:
        """Read a dataset from a path

        Parameters
        ----------
        path: str
            Path to the dataset

        Returns
        -------
        ray.data.Dataset
            A Ray Dataset object.
        """
        return RayDataset(ray.data.read_datasource(_RayDcDatasource(path)))
