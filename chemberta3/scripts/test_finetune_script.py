import deepchem as dc
import numpy as np
from deepchem.models.torch_models import Chemberta
from deepchem.molnet import load_delaney
from chemberta_finetune_script import load_dataset, get_tuning_utils, download_pretrained_model_from_S3, modify_model_keys, hyperparam_search, evaluate_model, main
import boto3
import torch
import tempfile
import os


def test_load_dataset():
    dataset = 'delaney'
    data = load_dataset(dataset, splitter='scaffold', featurizer='dummy')
    assert data[1][0].tasks == ['measured log solubility in mols per litre']

    dataset = 'bace_regression'
    data = load_dataset(dataset, splitter='scaffold', featurizer='dummy')
    assert data[1][0].tasks == ['pIC50']

    dataset = 'clearance'
    data = load_dataset(dataset, splitter='scaffold', featurizer='dummy')
    assert data[1][0].tasks == ['target']

    dataset = 'lipo'
    data = load_dataset(dataset, splitter='scaffold', featurizer='dummy')
    assert data[1][0].tasks == ['exp']

    dataset = 'bace_classification'
    data = load_dataset(dataset, splitter='scaffold', featurizer='dummy')
    assert data[1][0].tasks == ['Class']

    dataset = 'bbbp'
    data = load_dataset(dataset, splitter='scaffold', featurizer='dummy')
    assert data[1][0].tasks == ['p_np']

    dataset = 'hiv'
    data = load_dataset(dataset, splitter='scaffold', featurizer='dummy')
    assert data[1][0].tasks == ['HIV_active']

    dataset = 'sider'
    data = load_dataset(dataset, splitter='scaffold', featurizer='dummy')
    assert list(data[1][0].tasks) == [
        'Hepatobiliary disorders', 'Metabolism and nutrition disorders',
        'Product issues', 'Eye disorders', 'Investigations',
        'Musculoskeletal and connective tissue disorders',
        'Gastrointestinal disorders', 'Social circumstances',
        'Immune system disorders', 'Reproductive system and breast disorders',
        'Neoplasms benign, malignant and unspecified (incl cysts and polyps)',
        'General disorders and administration site conditions',
        'Endocrine disorders', 'Surgical and medical procedures',
        'Vascular disorders', 'Blood and lymphatic system disorders',
        'Skin and subcutaneous tissue disorders',
        'Congenital, familial and genetic disorders',
        'Infections and infestations',
        'Respiratory, thoracic and mediastinal disorders',
        'Psychiatric disorders', 'Renal and urinary disorders',
        'Pregnancy, puerperium and perinatal conditions',
        'Ear and labyrinth disorders', 'Cardiac disorders',
        'Nervous system disorders',
        'Injury, poisoning and procedural complications'
    ]

    dataset = 'tox21'
    data = load_dataset(dataset, splitter='scaffold', featurizer='dummy')
    assert list(data[1][0].tasks) == [
        'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
        'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
    ]

    dataset = 'clintox'
    data = load_dataset(dataset, splitter='scaffold', featurizer='dummy')
    assert list(data[1][0].tasks) == ['FDA_APPROVED', 'CT_TOX']


def test_get_tuning_utils():

    datasets = ['delaney', 'bace_regress', 'clearance', 'lipo']
    for dataset in datasets:
        output = get_tuning_utils(dataset)
        assert output[0] == 'regression'
        assert output[1] == False
        assert output[3] == 'rms_score'

    datasets = ['clintox', 'bace_class', 'bbbp', 'hiv', 'sider', 'tox21']
    for dataset in datasets:
        output = get_tuning_utils(dataset)
        assert output[0] == 'classification'
        assert output[1] == True
        assert output[3] == 'roc_auc_score'


def test_modify_model_keys_and_download_from_S3():

    temp_dir = tempfile.TemporaryDirectory()
    bucket_name = 'chemberta3'
    file_key = 'Chemberta-pretrained-models/Chemberta-100M-MLM/chemberta_100m_mlm_epoch_4/'

    model_file_path = download_pretrained_model_from_S3(
        bucket_name=bucket_name, file_key=file_key, temp_dir=temp_dir)
    assert os.path.exists(model_file_path)

    data = torch.load(model_file_path)
    assert any('module.' in key for key in data['model_state_dict'].keys())

    modified_checkpoint_path = modify_model_keys(model_path=model_file_path,
                                                 temp_dir=temp_dir)
    modified_checkpoint_data = torch.load(modified_checkpoint_path)
    assert not any(
        'module.' in key
        for key in modified_checkpoint_data['model_state_dict'].keys())


def test_hyperparam_search_and_evaluate_model():

    temp_dir = tempfile.TemporaryDirectory()
    model = Chemberta(n_tasks=1, task='mlm', model_dir=temp_dir.name)
    model.save_checkpoint()
    tasks, dataset, transformers = load_dataset('delaney', 'scaffold', 'dummy')
    train, val, test = dataset
    params = {
        'ckpt_path': [temp_dir.name],
        'learning_rate': [3e-05],
        'batch_size': [32, 64],
        'tasks': [tasks],
        'task': ['regression']
    }
    task, use_max, metric, metric_out = get_tuning_utils('delaney')
    epochs = [1]
    tuning_results = hyperparam_search(params, train, val, metric,
                                       transformers, epochs, use_max)
    assert len(tuning_results) == 1

    best_results_with_transform = evaluate_model(tuning_results, test, metric,
                                                 epochs, metric_out,
                                                 transformers)
    best_results_without_transform = evaluate_model(tuning_results, test,
                                                    metric, epochs, metric_out)

    assert best_results_with_transform[0] != best_results_without_transform[0]


class Args:

    def __init__(self, dataset, splitter, featurizer, bucket_name, file_key,
                 learning_rate, batch_size, epochs):
        # Direct assignment; no transformations
        self.dataset = dataset
        self.splitter = splitter
        self.featurizer = featurizer
        self.bucket_name = bucket_name
        self.file_key = file_key
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs


def test_main():

    bucket_name = 'chemberta3'
    file_key = 'Chemberta-pretrained-models/Chemberta-100M-MLM/chemberta_100m_mlm_epoch_4/checkpoint1.pt'
    args = Args("delaney", "scaffold", "dummy", bucket_name, file_key,
                [0.0001, 0.001], [32, 64], [1, 2])
    best_score, best_params, best_model = main(args)
    assert best_score > 0
