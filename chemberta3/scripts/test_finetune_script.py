import deepchem as dc
import numpy as np
from deepchem.models.torch_models import MoLFormer
from deepchem.molnet import load_delaney
from molformer_random_finetune import main
import boto3
import torch
import tempfile
import os

class Args:

    def __init__(self, dataset, splitter, featurizer, bucket_name, file_key,
                 learning_rate, batch_size, epochs, max_iter):
        # Direct assignment; no transformations
        self.dataset = dataset
        self.splitter = splitter
        self.featurizer = featurizer
        self.bucket_name = bucket_name
        self.file_key = file_key
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.max_iter = max_iter

def test_main():

    bucket_name = 'chemberta3'
    file_key = 'MoLFormer-pretrained-models/model_ckpt/checkpoint1.pt'
    learning_rate = np.random.uniform(1e-04, 3e-05, size=5).tolist()
    args = Args("delaney", "scaffold", "dummy", bucket_name, file_key,
                learning_rate, [32, 64], [1, 2], 2)
    best_score, best_params, best_model = main(args)
    assert best_score > 0
