# Multi-task regression featurizer
import multiprocessing as mp
import deepchem as dc
import logging
import pandas as pd
import os

logging.basicConfig(filename='logs/mtr.log', level=logging.INFO)

def _featurize_part(i, nproc):
    """Private function which creates a disk dataset 
    from the i'th subset of the dataset.

    Parameter
    ---------
    i: int
        The i'th subset of the dataset
    nproc: int
        Total number of processes
    """
    filepath = '../benchmarking/data/zinc15_1M_2D.csv'

    dest_dir = os.path.join('data/zinc1m', f'part{i}')
    os.makedirs(dest_dir, exist_ok=True)
    df = pd.read_csv(filepath)
    nrows = df.shape[0]
    partsize = df.shape[0] // nproc

    start = i * partsize
    end = (i + 1) * partsize

    df_subset = df.iloc[start:end]

    csvpath = f'data/zinc1m/part{i}.csv'
    df_subset.to_csv(csvpath, index=False)
    del df, df_subset

    loader = dc.data.CSVLoader(tasks=['logp'],
                               feature_field='smiles',
                               featurizer=dc.feat.RDKitDescriptors(),
                               id_field='smiles')
    dataset = loader.create_dataset(csvpath)

    dataset.to_csv(f'result{i}.csv')
    tasks = ['X' + str(i) for i in range(1, 210)]
    loader = dc.data.CSVLoader(tasks=tasks,
                               feature_field='ids',
                               featurizer=dc.feat.DummyFeaturizer(),
                               id_field='ids')
    dataset = loader.create_dataset(f'result{i}.csv')
    dataset.move(dest_dir)


if __name__ == '__main__':
    """A standalone script to featurize a csv file in parallel by using 
    multiple processes where each process featurizes the i'th subset of the csv file.
    The featurized disk dataset from each subset of the csv file
    written to a unique location in disk where are later merged
    to create a final DiskDataset."""
    nproc = os.cpu_count()

    processes = []
    for i in range(nproc):
        p = mp.Process(target=featurize_part, args=(i, nproc))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    zinc1m_dir = os.path.join('data', 'zinc1m-featurized',
                                       'rdkit-descriptor')
    os.makedirs(zinc1m_dir, exist_ok=True)

    datasets = []
    for i in range(nproc):
        data_dir = os.path.join('data/zinc1m', f'part{i}')
        datasets.append(dc.data.DiskDataset(data_dir=data_dir))

    dataset = dc.data.DiskDataset.merge(datasets, merge_dir=zinc1m_dir)
