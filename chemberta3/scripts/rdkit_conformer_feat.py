import multiprocessing as mp
import deepchem as dc
import logging
import pandas as pd
import os

logging.basicConfig(filename='logs/feat.log', level=logging.INFO)


def featurize_part(i, basedir):
    csvpath = os.path.join(basedir, f'part{i}.csv')
    dest_dir = os.path.join(basedir, f'part{i}')
    featurizer = dc.feat.RDKitConformerFeaturizer()
    loader = dc.data.CSVLoader(tasks=['logp'],
                               feature_field='smiles',
                               featurizer=featurizer,
                               id_field='smiles')
    dataset = loader.create_dataset(csvpath)
    dataset.move(dest_dir)


if __name__ == '__main__':
    featurizer = dc.feat.RDKitConformerFeaturizer()
    nproc = os.cpu_count()

    # Update basedir, filepath to point raw dataset which is to be featurized
    basedir = 'data/zinc100'
    filepath = '../benchmarking/data/zinc100.csv'
    df = pd.read_csv(filepath)
    nrows = df.shape[0]
    partsize = df.shape[0] // nproc

    for i in range(0, nproc):
        start = i * partsize
        end = (i + 1) * partsize
        df_subset = df.iloc[start:end]

        dest_dir = os.path.join(basedir, f'part{i}')
        os.makedirs(dest_dir, exist_ok=True)
        csvpath = os.path.join(basedir, f'part{i}.csv')
        df_subset.to_csv(csvpath, index=False)

    processes = []
    for i in range(nproc):
        p = mp.Process(target=featurize_part, args=(i, basedir))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    # Update mergedir to point at destination of final data dir
    merge_dir = os.path.join('data', 'zinc100-featurized', str(featurizer))
    os.makedirs(merge_dir, exist_ok=True)

    datasets = []
    for i in range(0, nproc):
        dataset_dir = os.path.join(basedir, f'part{i}')
        datasets.append(dc.data.DiskDataset(data_dir=dataset_dir))

    dataset = dc.data.DiskDataset.merge(datasets, merge_dir=merge_dir)
