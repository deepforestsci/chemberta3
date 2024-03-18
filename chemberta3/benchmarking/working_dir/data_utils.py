import ray
import deepchem as dc
import numpy as np
from io import BytesIO
from typing import List, Optional, Union, Dict, Any, Iterator

from ray.data.block import Block, BlockAccessor
from ray.data.datasource.file_based_datasource import FileBasedDatasource
from ray.data.datasource.file_datasink import BlockBasedFileDatasink


class RayDcDatasource(FileBasedDatasource):
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
        Paramters
        ---------
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

    def _read_stream(self, f: "pyarrow.NativeFile",
                     path: str) -> Iterator[Block]:
        """Reads a stream of data"""
        # TODO(ekl) Ideally numpy can read directly from the file, but it
        # seems like it requires the file to be seekable.
        buf = BytesIO()
        data = f.readall()
        buf.write(data)
        buf.seek(0)
        data = dict(np.load(buf, allow_pickle=True, **self.numpy_load_args))
        yield BlockAccessor.batch_to_block(data)


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
