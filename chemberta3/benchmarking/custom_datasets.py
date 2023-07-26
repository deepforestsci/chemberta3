import os
import deepchem as dc
from typing import List, Tuple, Optional
from functools import partial
import pandas as pd

FEATURIZER_MAPPING = {
    "molgraphconv": dc.feat.MolGraphConvFeaturizer(use_edges=True),
    "ecfp": dc.feat.CircularFingerprint(),
    "convmol": dc.feat.ConvMolFeaturizer(),
    "weave": dc.feat.WeaveFeaturizer(max_pair_distance=2),
    "dummy": dc.feat.DummyFeaturizer(),
    "grover": dc.feat.GroverFeaturizer(features_generator=dc.feat.CircularFingerprint()),
}

def prepare_data(dataset_name, featurizer_name, data_dir: Optional[str] = None):
    if dataset_name == 'zinc5k':
        load_zinc5k(featurizer_name, data_dir)

def load_nek(
    featurizer: dc.feat.Featurizer,
    tasks_wanted: List[str] = ["NEK2_ki_avg_value"],
    splitter=None,
) -> Tuple[List[str], Tuple[dc.data.Dataset, ...], List[dc.trans.Transformer], str]:
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
    nek_df = pd.read_csv(
        "s3://chemberta3/datasets/kinases/NEK/nek_mtss.csv", index_col=0
    )

    with dc.utils.UniversalNamedTemporaryFile(mode="w") as tmpfile:
        data_df = nek_df.dropna(subset=tasks_wanted)
        data_df.to_csv(tmpfile.name)
        loader = dc.data.CSVLoader(
            tasks_wanted, feature_field="raw_smiles", featurizer=featurizer
        )
        dc_dataset = loader.create_dataset(tmpfile.name)

    return [], [dc_dataset], []

load_zinc250k = partial(dc.molnet.load_zinc15, dataset_size='250K')

def load_zinc5k(featurizer_name, data_dir: Optional[str] = None):
    filepath = 'data/zinc5k.csv'
    featurizer = FEATURIZER_MAPPING[featurizer_name]
    if data_dir is None:
        data_dir = os.path.join('data', 'zinc5k-featurized', featurizer_name)

    # Ideally, we don't need logp here - we should pass empty tasks ([]) but it casues error during model.fit call
    loader = dc.data.CSVLoader(['logp'], feature_field='smiles', featurizer=featurizer, id_field='smiles')
    dataset = loader.create_dataset(filepath)
    dataset.move(data_dir)
    return
