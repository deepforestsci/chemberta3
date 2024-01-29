# chemberta3
ChemBERTa-3 Repo

## Step 1: Setup Environment 

Clone the current repo as `git clone https://github.com/deepforestsci/chemberta3`.

The setup uses DeepChem and PyTorch, along with the additional python dependencies for running benchmarking tasks. Install the nightly version of DeepChem as `pip install --pre deepchem` and other dependencies via `requirements.txt` file as `pip install -r requirements.txt`

## Step 2: Prepare data

To run benchmarking tasks, data has to be prepared.
This involves featurization of data points from SMILES strings into the corresponding representation for the model.
In some cases, it also involves preparing vocabulary for models which follow self-supervised learning approaches (ex: GROVER model).

The `prepare_data.py` script can be used to prepare data for benchmarking models.

Example invocations:

```py
# featurize delaney dataset using circular fingerprint (ecfp) featurizer
python3 prepare_data.py --dataset_name delaney --featurizer_name ecfp
```

The following featurizers are supported:
- CircularFingerprint (ecfp) - Circular fingerprint featurizer
- DummyFeaturizer (dummy) - performs no featurization on the data 
- GroverFeaturizer (grover) - performs featurization for grover model
- MolGraphConvFeaturizer (molgraphconv) - performs Molecular Graph Convolution featurizer
- RDKitConformer (rdkit-conformer) - performs rdkit conformer featurization for infomax3d model
- SNAPFeaturizer (snap) - performs featurization 

All dataset from MoleculeNet suite of dataset are supported. The corresponding codes for the dataset are:
- delaney - Delaney dataset
- bace_c - bace classification dataset
- bace_r - bace regression dataset
- bbbp - Brain Blood Barrier Penetration dataset
- clintox - clintox dataset
- hiv - hiv dataset
- lipo - Lipo dataset
- tox21 - Tox21 dataset
- zinc250k - zinc250k dataset
- zinc1m - Zinc1m dataset
- zinc10m - Zinc10m dataset

## Step 3: Benchmark models

The `benchmark.py` script allows us to benchmark models against different downstream datasets. The script takes in a config file as input and runs the benchmarking task. The config file contains the following fields:
- `experiment_name` - name of the experiment
- `model_name` - name of the model to be benchmarked
- `model_parameters` - model parameters for the model to be benchmarked
- `dataset_name` - name of the dataset to be used for benchmarking
- `featurizer_name` - name of the featurizer to be used for featurizing the dataset
featurizing the dataset
- `nb_epoch` - number of training epochs
- `test_data_dir` - directory where the test dataset is stored
- `train_data_dir` - directory where the train dataset is stored
- `valid_data_dir` - directory where the validation dataset is stored


Example config file:
```yml
train: True

model_name: 'snap'
model_parameters:
  gnn_type: 'gin'
  num_layer: 3
  emb_dim: 64
  num_tasks: 1
  graph_pooling: 'mean'
  dropout: 0
  task: 'classification'
  mode: 'classification'
  n_tasks: 1
  num_classes: 2

nb_epoch: 100
checkpoint_interval: 4

train_data_dir: 'data/tox21-featurized/SNAPFeaturizer/ScaffoldSplitter/BalancingTransformer/train_dir'
test_data_dir: 'data/tox21-featurized/SNAPFeaturizer/ScaffoldSplitter/BalancingTransformer/test_dir'
valid_data_dir: 'data/tox21-featurized/SNAPFeaturizer/ScaffoldSplitter/BalancingTransformer/valid_dir'

early_stopper: True
patience: 100
```

After preparing the configuration file, run the benchmarking tasks as follows:
``python3 benchmark.py --config <path-to-config.yml-file>``

Other example config files can be found in the path `chemberta3/benchmarking/configs`.
