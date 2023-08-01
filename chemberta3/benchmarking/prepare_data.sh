# preparing zinc5k datasets
## dataset used by chemberta-mlm, chemberta-mtr pretraining models
python3 benchmark.py --prepare_data --dataset_name zinc5k --featurizer_name dummy 
## dataset used by grover model 
python3 benchmark.py --prepare_data --dataset_name zinc5k --featurizer_name grover
## dataset used by infograph model
python3 benchmark.py --prepare_data --dataset_name zinc5k --featurizer_name molgraph 
## dataset used by infomax3d model
python3 benchmark.py --prepare_data --dataset_name zinc5k --featurizer_name rdkit-conformer
## dataset used by snap model
python3 benchmark.py --prepare_data --dataset_name zinc5k --featurizer_name snap
