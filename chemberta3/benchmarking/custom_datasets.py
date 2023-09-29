import os
import deepchem as dc
from typing import List, Tuple, Optional
from functools import partial
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
        dc.feat.GroverFeaturizer(
            features_generator=dc.feat.CircularFingerprint()),
    "rdkit-conformer":
        dc.feat.RDKitConformerFeaturizer(num_conformers=1),
    "snap":
        dc.feat.SNAPFeaturizer(),
}

DATASET_MAPPING = {
    'delaney': dc.molnet.load_delaney,
    'bace_classification': dc.molnet.load_bace_classification,
    'bace_regression': dc.molnet.load_bace_regression,
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
}


def prepare_data(dataset_name,
                 featurizer_name,
                 data_dir: Optional[str] = None,
                 split_dataset: Optional[bool] = True):
    """Featurizes a raw dataset for training

    Parameters
    ----------
    dataset_name: str
        Name of dataset from dataset mapping
    featurizer_name: str
        Name of featurizer

    Example
    -------
    >>> # prepares delaney dataset by performing dummy featurization and stores dataset
    >>> # in the directory data
    >>> prepare_data('delaney', 'dummy', data_dir='data')
    """
    if data_dir is None:
        data_dir = os.path.join('data')
    os.environ['DEEPCHEM_DATA_DIR'] = data_dir
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
    dataset = loader.create_dataset(filepath)
    dataset.move(data_dir)

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
