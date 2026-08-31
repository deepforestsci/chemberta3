from ray.data.datasource.file_based_datasource import FileBasedDatasource
from ray.data.datasource.file_datasink import BlockBasedFileDatasink
from typing import List, Optional, Union, Dict, Any, Iterator
import ray
import deepchem as dc
import numpy as np
from io import BytesIO
from ray.data.block import Block, BlockAccessor

class RayDcDatasink(BlockBasedFileDatasink):

    def __init__(
        self,
        path: str,
        columns: List[str],
        *,
        file_format: str = "npz",
        **file_datasink_kwargs,
    ):
        super().__init__(path, file_format=file_format, **file_datasink_kwargs)

        self.columns = columns

    def write_block_to_file(self, block: BlockAccessor,
                            file: "pyarrow.NativeFile"):
        data = {}
        for column in self.columns:
            data[column] = block.to_numpy(column)
        np.savez(file, **data)


class RayDcDatasource(FileBasedDatasource):

    _FILE_EXTENSIONS = ["npz"]

    def __init__(
        self,
        paths: Union[str, List[str]],
        numpy_load_args: Optional[Dict[str, Any]] = None,
        **file_based_datasource_kwargs,
    ):
        super().__init__(paths, **file_based_datasource_kwargs)

        if numpy_load_args is None:
            numpy_load_args = {}

        self.numpy_load_args = numpy_load_args

    def _read_stream(self, f: "pyarrow.NativeFile",
                     path: str) -> Iterator[Block]:
        # TODO(ekl) Ideally numpy can read directly from the file, but it
        # seems like it requires the file to be seekable.
        buf = BytesIO()
        data = f.readall()
        buf.write(data)
        buf.seek(0)
        data = dict(np.load(buf, allow_pickle=True, **self.numpy_load_args))
        yield BlockAccessor.batch_to_block(data)


class RayDataset(dc.data.Dataset):

    def __init__(self,
                 dataset: ray.data.Dataset,
                 x_column='x',
                 y_column: Optional[str] = None):
        self.dataset = dataset
        self.x_column, self.y_column = x_column, y_column

    def featurize(self, featurizer: dc.feat.Featurizer, column):

        class RayFeaturizer:

            def __init__(self, featurizer, column):
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
