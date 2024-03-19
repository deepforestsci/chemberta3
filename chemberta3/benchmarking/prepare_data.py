import os
import sys
import pickle
import argparse
import deepchem as dc
from deepchem.data import DiskDataset
from typing import List, Tuple, Optional
from functools import partial
import json
from ast import literal_eval as make_tuple
import shutil
from ray.job_submission import JobSubmissionClient

import pandas as pd
import logging

FEATURIZER_MAPPING = {
    "molgraphconv":
    dc.feat.MolGraphConvFeaturizer(use_edges=True),
    "ecfp":
    dc.feat.CircularFingerprint(),
    "convmol":
    dc.feat.ConvMolFeaturizer(),
    "weave":
    dc.feat.WeaveFeaturizer(max_pair_distance=2),
    "dummy":
    dc.feat.DummyFeaturizer(),
    "grover":
    dc.feat.GroverFeaturizer(features_generator=dc.feat.CircularFingerprint()),
    "rdkit-conformer":
    dc.feat.RDKitConformerFeaturizer(),
    "snap":
    dc.feat.SNAPFeaturizer(),
}


def load_nek(
    featurizer: dc.feat.Featurizer,
    tasks_wanted: List[str] = ["NEK2_ki_avg_value"],
    splitter=None,
) -> Tuple[List[str], Tuple[dc.data.Dataset, ...], List[dc.trans.Transformer],
           str]:
    """Load NEK dataset.

    The NEK dataset is a collection of datapoints related to the NEK kinases,
    including SMILES strings, ki, kd, inhibition, and ic50 values.

    NEK2_ki_avg_value contains the most non-nan data points. For other tasks,
    load the NEK .csv file directly from "s3://chemberta3/datasets/kinases/NEK/nek_mtss.csv".

    Mimics the loaders for molnet datasets.

    Parameters
    ----------
    featurizer: dc.feat.Featurizer
        Featurizer to use.
    tasks_wanted: List[str]
        Tasks to load. These should correspond to the columns in the dataframe.
    splitter: dc.splits.splitters.Splitter
        Splitter to use. This should be None, and is included only for compatibility.

    Returns
    -------
    tasks: List[str]
        List of tasks.
    datasets: Tuple[dc.data.Dataset, ...]
        Tuple of train, valid, test datasets.
    transformers: List[dc.trans.Transformer]
        List of transformers.

    """
    assert (
        splitter is None
    ), "Splitter arg only used for compatibility with other dataset loaders."
    nek_df = pd.read_csv("s3://chemberta3/datasets/kinases/NEK/nek_mtss.csv",
                         index_col=0)

    with dc.utils.UniversalNamedTemporaryFile(mode="w") as tmpfile:
        data_df = nek_df.dropna(subset=tasks_wanted)
        data_df.to_csv(tmpfile.name)
        loader = dc.data.CSVLoader(tasks_wanted,
                                   feature_field="raw_smiles",
                                   featurizer=featurizer)
        dc_dataset = loader.create_dataset(tmpfile.name)

    return [], [dc_dataset], []


DATASET_MAPPING = {
    'delaney': dc.molnet.load_delaney,
    'bace_c': dc.molnet.load_bace_classification,
    'bace_r': dc.molnet.load_bace_regression,
    'bbbp': dc.molnet.load_bbbp,
    'clintox': dc.molnet.load_clintox,
    'hiv': dc.molnet.load_hiv,
    'muv': dc.molnet.load_muv,
    'pcba': dc.molnet.load_pcba,
    'qm9': dc.molnet.load_qm9,
    'clearance': dc.molnet.load_clearance,
    'lipo': dc.molnet.load_lipo,
    'tox21': partial(dc.molnet.load_tox21, tasks=['SR-p53']),
    'zinc250k': partial(dc.molnet.load_zinc15, dataset_size='250K'),
    'zinc1m': partial(dc.molnet.load_zinc15, dataset_size='1M'),
    'zinc10m': partial(dc.molnet.load_zinc15, dataset_size='10M'),
    'chembl': partial(dc.molnet.load_chembl, set='sparse'),
    "nek": {
        "loader": load_nek,
        "output_type": "regression",
        "tasks_wanted": ["NEK2_ki_avg_value"],
        "n_tasks": 1,
    },
}


def _featurize_csv_part(i, base_dir, featurizer_name):
    """Performs featurization on a subset of csv file

    Parameters
    ----------
    i: int
        The i'th process which work on the i'th subset of data
    featurizer: dc.feat.Featurizer
        A deepchem featurizer which is to be applied on the dataset
    base_dir: str
        A location of a temporary directory to store artifcats during multicpu featurization
    """
    featurizer = FEATURIZER_MAPPING[featurizer_name]
    csvpath = os.path.join(base_dir, f'part{i}.csv')
    dest_dir = os.path.join(base_dir, f'part{i}')
    # FIXME Currently, it only works for zinc datasets
    # To make it work for other datasets, we need to add options argparse options for tasks and feature field
    loader = dc.data.CSVLoader(tasks=['logp'],
                               feature_field='smiles',
                               featurizer=featurizer,
                               id_field='smiles')
    dataset = loader.create_dataset(csvpath,
                                    data_dir=dest_dir,
                                    shard_size=4096)


def _merge_disk_dataset_by_move(data_dirs: List[str], merge_dir: str):
    """Merging of dataset by moving shards

    Parameters
    ----------
    data_dirs: str
        Directories of DiskDataset containing parts of featurized input data
    merge_dir: str
        Directory to store resultant merge dir

    Notes
    -----
    DiskDataset.merge merges data by copying shards. This process is inherently slow.
    Since we are featurizing data from same input in multi-cpu featurization, we can
    take advantage of the homogenity in the input and featurized datasets and 
    merge datasets just by moving and renumbering the shards.
    Such a merge is much faster than DiskDataset.merge.
    """
    shard_sizes = []
    num_shards = []
    shard_num = 0
    metadata_rows = []

    for i, data_dir in enumerate(data_dirs):
        if i == 0:
            tasks_filename = os.path.join(data_dir, 'tasks.json')
            with open(tasks_filename) as fin:
                tasks = json.load(fin)
        dataset = dc.data.DiskDataset(data_dir)
        shard_sizes.append(dataset.get_shard_size())
        num_shards.append(dataset.get_number_shards())
        del dataset

        n_shards = num_shards[i]

        metadata_file = os.path.join(data_dir, 'metadata.csv.gzip')
        metadata_df = pd.read_csv(metadata_file,
                                  compression='gzip',
                                  dtype=object)
        metadata_df = metadata_df.where((pd.notnull(metadata_df)), None)

        for local_shard_num in range(n_shards):

            from_shard_basename = 'shard-%d' % local_shard_num
            in_X = "%s-X.npy" % from_shard_basename
            in_y = "%s-y.npy" % from_shard_basename
            in_w = "%s-w.npy" % from_shard_basename
            in_ids = "%s-ids.npy" % from_shard_basename

            basename = 'shard-%d' % shard_num
            out_X = "%s-X.npy" % basename
            out_y = "%s-y.npy" % basename
            out_w = "%s-w.npy" % basename
            out_ids = "%s-ids.npy" % basename

            shutil.move(os.path.join(data_dir, in_X),
                        os.path.join(merge_dir, out_X))
            shutil.move(os.path.join(data_dir, in_y),
                        os.path.join(merge_dir, out_y))
            shutil.move(os.path.join(data_dir, in_w),
                        os.path.join(merge_dir, out_w))
            shutil.move(os.path.join(data_dir, in_ids),
                        os.path.join(merge_dir, out_ids))
            out_ids_shape = make_tuple(
                metadata_df.iloc[local_shard_num]['ids_shape'])
            out_X_shape = make_tuple(
                metadata_df.iloc[local_shard_num]['X_shape'])
            out_y_shape = make_tuple(
                metadata_df.iloc[local_shard_num]['y_shape'])
            out_w_shape = make_tuple(
                metadata_df.iloc[local_shard_num]['w_shape'])
            metadata_rows.append([
                out_ids, out_X, out_y, out_w, out_ids_shape, out_X_shape,
                out_y_shape, out_w_shape
            ])

            shard_num += 1
        del metadata_df
    metadata_df = DiskDataset._construct_metadata(metadata_rows)
    DiskDataset._save_metadata(metadata_df, merge_dir, tasks)
    return


def multicpu_featurization(csv_path: str, featurizer_name: str, nproc: int,
                           base_dir: str, merge_dir: str):
    """Performs featurization of dataset on multiple cpus

    Parameters
    ----------
    csv_path: str
        Path to raw csv file containing data to be featurized
    featurizer_name: str
        Name of featurizer to use for featurization
    nproc: int
        Number of processes
    base_dir: str
        A location of a temporary directory to store artifcats during multicpu featurization
    merge_dir: str
        Final location of featurized dataset
    """
    df = pd.read_csv(csv_path)
    partsize = df.shape[0] // nproc

    for i in range(0, nproc):
        start = i * partsize
        end = (i + 1) * partsize
        df_subset = df.iloc[start:end]

        dest_dir = os.path.join(base_dir, f'part{i}')
        os.makedirs(dest_dir, exist_ok=True)
        sub_csvpath = os.path.join(base_dir, f'part{i}.csv')
        df_subset.to_csv(sub_csvpath, index=False)

    processes = []
    for i in range(nproc):
        p = mp.Process(target=_featurize_csv_part,
                       args=(i, base_dir, featurizer_name))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    os.makedirs(merge_dir)
    data_dirs = [os.path.join(base_dir, f'part{i}') for i in range(0, nproc)]
    _merge_disk_dataset_by_move(data_dirs, merge_dir)
    shutil.rmtree(base_dir)


def prepare_data(dataset_name,
                 featurizer_name,
                 data_dir: Optional[str] = None,
                 split_dataset: Optional[bool] = True,
                 is_multicpu_feat: Optional[bool] = None,
                 csv_path: Optional[str] = None,
                 ncpu: Optional[int] = None):
    """Featurizes a raw dataset for training

    Parameters
    ----------
    dataset_name: str
        Name of dataset from dataset mapping
    featurizer_name: str
        Name of featurizer
    is_multicpu_feat: bool, optional (default None)
        If True, performs multicpu featurization
    csv_path: str, optional (default None)
        Path to csv file for performing multicpu featurization
    ncpu: int, optional (default None)
        Number of CPUs to use for multicpu featurization

    Example
    -------
    >>> # prepares delaney dataset by performing dummy featurization and stores dataset
    >>> # in the directory data
    >>> prepare_data('delaney', 'dummy', data_dir='data')
    """
    if data_dir is None:
        data_dir = os.path.join('/data')
    os.environ['DEEPCHEM_DATA_DIR'] = data_dir
    if is_multicpu_feat:
        base_dir = os.path.join(
            data_dir, 'multicpu-feat-' + str(ncpu) + '-' + featurizer_name)
        os.makedirs(base_dir)
        csv_key = csv_path.removesuffix('.csv').split('/')[-1]
        merge_dir = os.path.join(data_dir, csv_key, featurizer_name)
        multicpu_featurization(csv_path=csv_path,
                               featurizer_name=featurizer_name,
                               nproc=ncpu,
                               base_dir=base_dir,
                               merge_dir=merge_dir)
    else:
        featurizer = FEATURIZER_MAPPING[featurizer_name]
        if dataset_name == 'zinc5k':
            load_zinc5k(featurizer, data_dir)
        else:
            loader = DATASET_MAPPING[dataset_name]
            splitter = dc.splits.ScaffoldSplitter() if split_dataset else None
            tasks, datasets, transformers = loader(featurizer=featurizer,
                                                   data_dir=data_dir,
                                                   splitter=splitter)

        if featurizer_name == 'grover':
            # build grover vocabulary for grover featurizer
            prepare_vocab(dataset_name, data_dir, split_dataset)
            # FIXME currently not supporting vocabulary preparation in multi-cpu featurization


def load_zinc5k(featurizer, data_dir: Optional[str] = None) -> None:
    """Featurizes saves zinc5k dataset in `data_dir`

    Parameters
    ----------
    featurizer: dc.feat.Featurizer
        Featurizer
    data_dir: Optional[str]
        Directory to store data
    """
    filepath = 'data/zinc5k.csv'
    if data_dir is None:
        base_dir = os.path.join('data', 'zinc5k-featurized')
    else:
        base_dir = data_dir

    data_dir = os.path.join(base_dir, featurizer_name)
    if os.path.isdir(data_dir):
        if not os.listdir(data_dir):
            print(
                "Data directory already exists. Data may already exist in data directory. Aborting featurization."
            )
            return

    # Ideally, we don't need logp here - we should pass empty tasks ([]) but it casues error during model.fit call
    loader = dc.data.CSVLoader(['logp'],
                               feature_field='smiles',
                               featurizer=featurizer,
                               id_field='smiles')
    dataset = loader.create_dataset(filepath, data_dir=data_dir)

    if featurizer_name == 'grover':
        prepare_vocab('zinc5k', base_dir)
    return None


def prepare_vocab(dataset_name, data_dir, split_dataset):
    """Creates vocabulary from a dataset.

    The method currently supports only grover vocabulary builders.

    Parameters
    ----------
    dataset_name: str
        Name of dataset from dataset mapping
    data_dir: str
        Directory to store dataset 

    Example
    -------
    >>> prepare_vocab('delaney', 'data', False)
    """
    # Create grover vocabulary on the dataset
    # Ideally, we can reuse the featurized dataset from `prepare_data`
    # but GroverAtomVocabularyBuilder works with `smiles`
    # attribute as DiskDataset.X which is not the case when we have already applied a featurizer on the dataset
    from deepchem.feat.vocabulary_builders import GroverAtomVocabularyBuilder, GroverBondVocabularyBuilder
    featurizer = dc.feat.DummyFeaturizer()
    if dataset_name == 'zinc5k':
        filepath = 'data/zinc5k.csv'
        loader = dc.data.CSVLoader(['logp'],
                                   feature_field='smiles',
                                   featurizer=featurizer,
                                   id_field='smiles')
        dataset = loader.create_dataset(filepath)
    else:
        loader = DATASET_MAPPING[dataset_name]
        if split_dataset:
            tasks, datasets, transformers = loader(
                featurizer=featurizer,
                data_dir=data_dir,
                splitter=dc.splits.ScaffoldSplitter())
            dataset, _, _ = datasets
        else:
            tasks, datasets, transformers = loader(featurizer=featurizer,
                                                   data_dir=data_dir,
                                                   splitter=None)
            dataset = datasets[0]

    vocab_dir = os.path.join(data_dir, dataset_name, 'grover_vocab')
    os.makedirs(vocab_dir, exist_ok=True)
    atom_vocab_path = os.path.join(vocab_dir, 'atom_vocab.json')
    av = GroverAtomVocabularyBuilder()
    av.build(dataset)
    av.save(atom_vocab_path)

    bond_vocab_path = os.path.join(vocab_dir, 'bond_vocab.json')
    bv = GroverBondVocabularyBuilder()
    bv.build(dataset)
    bv.save(bond_vocab_path)
    return


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--use-ray',
                           default=False,
                           action='store_true',
                           help='enable multinode featurization using ray')
    argparser.add_argument("--dataset_name", type=str)
    argparser.add_argument("--featurizer_name", type=str)
    argparser.add_argument('--multicpu_feat',
                           default=False,
                           action='store_true',
                           help='enable multicpu featurization')
    argparser.add_argument('--csv_path',
                           help='path to csv file for multi-cpu featurization',
                           default=None,
                           type=str)
    argparser.add_argument('--ncpu',
                           help='Number of CPUs to use for featurization',
                           default=None,
                           type=int)
    argparser.add_argument('--ttv_split',
                           help='Train test validation split of dataset',
                           default=False,
                           action='store_true')
    argparser.add_argument('--data_dir',
                           help='data dir to store data',
                           default=None,
                           type=str)
    args = argparser.parse_args()

    split_dataset = args.ttv_split
    if args.use_ray:
        command = f"python3 feat.py --featurizer {args.featurizer_name} --dataset {args.dataset_name}"
        print ('job command is \n', command)
        client = JobSubmissionClient("http://127.0.0.1:8265")
        job_id = client.submit_job(
            entrypoint=command,
            runtime_env={"working_dir": "working_dir"})
        sys.exit(0)

    if args.multicpu_feat:
        if args.csv_path is None:
            raise ValueError(
                'Path to csv file for performing featurization is required')
        if args.ncpu is None:
            # -1 to leave out a cpu for the master process - the benchmark.py script
            args.ncpu = os.cpu_count() - 1
    prepare_data(dataset_name=args.dataset_name,
                 featurizer_name=args.featurizer_name,
                 data_dir=args.data_dir,
                 split_dataset=split_dataset,
                 is_multicpu_feat=args.multicpu_feat,
                 csv_path=args.csv_path,
                 ncpu=args.ncpu)
