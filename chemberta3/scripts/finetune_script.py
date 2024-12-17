import argparse
import deepchem as dc
import numpy as np
from deepchem.models.torch_models import Chemberta
from deepchem.molnet import load_delaney
import boto3
import tempfile
import torch
import os

featurizers = {
  'dummy': dc.feat.DummyFeaturizer
}

regression_dataset = ['delaney', 'bace_regress', 'clearance', 'lipo']
classification_dataset = ['clintox', 'bace_class', 'bbbp', 'hiv', 'sider', 'tox21']

def load_dataset(dataset: str, splitter: str, featurizer: str):
    load_fn = getattr(dc.molnet, f"load_{dataset}")
    loader = load_fn(featurizer=featurizers[featurizer](), splitter=splitter)
    tasks, dataset, transformers = loader
    return tasks, dataset, transformers


def get_tuning_utils(dataset: str):
    if dataset in regression_dataset:
        task = 'regression'
        use_max = False
        metric = dc.metrics.Metric(dc.metrics.rms_score)
        metric_out = 'rms_score'

    elif dataset in classification_dataset:
        task = 'classification'
        use_max = True
        metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
        metric_out = 'roc_auc_score'
    return task, use_max, metric, metric_out


def download_pretrained_model_from_S3(bucket_name, file_key, temp_dir):

    s3 = boto3.client('s3')
    temp_file_path = os.path.join(temp_dir.name, 'tmp.pt')

    # Download the file from S3
    s3.download_file(bucket_name, file_key, temp_file_path)
    print(f"File downloaded to temporary directory: {temp_file_path}")
    return temp_file_path


def modify_model_keys(model_path: str, temp_dir):
    # model_path = tempdir.name
    data = torch.load(model_path)
    data['model_state_dict'] = {key.replace("module.", ""): value for key, value in data['model_state_dict'].items()}

    checkpoint_path = os.path.join(temp_dir.name, 'checkpoint1.pt')
    with open(checkpoint_path, 'wb') as f:
        torch.save(data, f)

    os.remove(model_path)
    return checkpoint_path


def load_and_restore(**params):
    ckpt_path = params['ckpt_path']
    tasks = params['tasks']
    task = params['task']
    del params['ckpt_path']
    del params['tasks']
    del params['task']
    model = Chemberta(n_tasks=len(tasks), task=task, **params)
    model.load_from_pretrained(ckpt_path)
    return model


def hyperparam_search(params, train, val, metric, transformers, epochs, use_max, logdir):
    
    for i, epoch in enumerate(epochs):
        # set use_max to false when using regression datasets 
        optimizer = dc.hyper.GridHyperparamOpt(load_and_restore)
        result = optimizer.hyperparam_search(params, train, val, metric, transformers, nb_epoch=epoch, use_max=use_max, logfile=f'results_{i}.txt', logdir=log_dir)
        tuning_results.append(result)
        del optimizer
        torch.cuda.empty_cache()
        return tuning_results


def evaluate_model(tuning_results, test, metric, transformers, epochs, metric_out):

    best_results = []
    for i in range(len(epochs)):
        eval = tuning_results[i][0].evaluate(test, metrics=[metric], transformers=transformers)
        print(eval)
        best_results.append(eval[metric_out])
        return best_results


if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset',
                           type=str,
                           help='name of dataset',
                           default=None)

    argparser.add_argument('--splitter',
                           type=str,
                           help='splitter type',
                           default='scaffold')
    
    argparser.add_argument('--featurizer',
                           type=str,
                           help='featurizer type',
                           default='dummy')

    argparser.add_argument('--bucket_name',
                           type=str,
                           help='name of S3 bucket',
                           default=None)

    argparser.add_argument('--file_key',
                           type=str,
                           help='file key of the pretrained checkpoint',
                           default=None)

    args = argparser.parse_args()
    # print(args.dataset)
    tasks, dataset, transformers = load_dataset(dataset=args.dataset, splitter=args.splitter, featurizer=args.featurizer)
    train, val, test = dataset

    task, use_max, metric, metric_out = get_tuning_utils(dataset=args.dataset)

    temp_dir = tempfile.TemporaryDirectory()
    tune_dir = os.path.join(os.getcwd(), 'tuning_runs')
    if not os.path.exists(tune_dir):
        os.mkdir(tune_dir)
    log_dir= os.path.join(temp_dir.name, 'tuning_results_metadata')
    os.mkdir(log_dir)

    pretrained_model_file_path = download_pretrained_model_from_S3(bucket_name=args.bucket_name, file_key=args.file_key, temp_dir=temp_dir)
    print("pretrained model download complete!")

    modified_checkpoint_path = modify_model_keys(pretrained_model_file_path, temp_dir)
    print("pretrained model keys modified!")

    # the parameters which are to be optimized
    params = {
    'ckpt_path': [temp_dir.name],
    'learning_rate': [3e-05, 6e-04],
    'batch_size': [32, 64],
    'tasks': [tasks],
    'task': [task]
    }

    # Creating optimizer and searching over hyperparameters
    tuning_results = []
    epochs = [1]

    tuning_results = hyperparam_search(params, train, val, metric, transformers, epochs, use_max, log_dir)
    print("hyperparameter search done!")

    best_results = evaluate_model(tuning_results, test, metric, transformers, epochs, metric_out)

    print(best_results)
