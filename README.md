# chemberta3
ChemBERTa-3 Repo


# Benchmarking
The `benchmark.py` script provides the ability to benchmark models against different downstream datasets.

Example command:
```python benchmark.py --dataset_name=delaney --model_name=infograph --featurizer_name=molgraphconv --checkpoint=checkpoint5.pt```

## Benchmarking using a config file
```
python3 benchmark.py --config configs/delaney.yml
```

### Benchmarking random forest

```
python benchmark.py --dataset_name=delaney --model_name=random_forest --featurizer_name=ecfp
python benchmark.py --dataset_name=bace_classification --model_name=random_forest --featurizer_name=ecfp
python benchmark.py --dataset_name=bace_regression --model_name=random_forest --featurizer_name=ecfp


### Benchmarking graphconv model
```
python3 benchmark.py --model_name=graphconv --featurizer_name=convmol --dataset_name=delaney
```

### Data preparation
```
python3 benchmark.py --prepare_data --dataset_name zinc5k --featurizer_name ecfp
```
