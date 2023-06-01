# chemberta3
ChemBERTa-3 Repo


# Benchmarking
The `benchmark.py` script provides the ability to benchmark models against different downstream datasets.

Example command:
```python benchmark.py --dataset_name=delaney --model_name=infograph --featurizer_name=molgraphconv --checkpoint=checkpoint5.pt```

### Benchmarking chemberta2 model on bace dataset

```
python3 scripts/main.py --model_name chemberta \
    --featurizer_name dummy \
    --checkpoint DeepChem/ChemBERTa-77M-MLM \
    --from-hf-checkpoint \
    --task mlm \
    --tokenizer-path DeepChem/ChemBERTa-77M-MLM \
    --job evaluate
```
