python3 main.py --model_name chemberta \
    --featurizer_name dummy \
    --checkpoint DeepChem/ChemBERTa-77M-MLM \
    --from-hf-checkpoint \
    --task mlm \
    --tokenizer-path DeepChem/ChemBERTa-77M-MLM \
    --job evaluate
