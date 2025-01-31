import ast
import argparse
import deepchem as dc
import numpy as np
from deepchem.models.torch_models import MoLFormer
from deepchem.molnet import load_delaney
import collections
import itertools
import boto3
import tempfile
import torch
import os
import logging
from functools import reduce
from operator import mul
from typing import Dict, List, Optional, Tuple
from deepchem.data import Dataset
from deepchem.trans import Transformer
from deepchem.models import Model
from deepchem.metrics import Metric
from deepchem.hyper.base_classes import HyperparamOpt
from deepchem.hyper.base_classes import _convert_hyperparam_dict_to_filename

logger = logging.getLogger(__name__)

featurizers = {
  'dummy': dc.feat.DummyFeaturizer
}

regression_dataset = ['delaney', 'bace_regression', 'clearance', 'lipo']
classification_dataset = ['clintox', 'bace_classification', 'bbbp', 'hiv', 'sider', 'tox21']


class RandomHyperparamOpt(HyperparamOpt):

    def __init__(self, model_builder, max_iter: int):
        super(RandomHyperparamOpt, self).__init__(model_builder=model_builder)
        self.max_iter = max_iter

    def hyperparam_search(
        self,
        params_dict: Dict,
        train_dataset: Dataset,
        valid_dataset: Dataset,
        metric: Metric,
        output_transformers: List[Transformer] = [],
        nb_epoch: int = 10,
        use_max: bool = True,
        logfile: str = 'results.txt',
        logdir: Optional[str] = None,
        **kwargs,
    ):

        # hyperparam_list should either be an Iterable sequence or a random sampler with rvs method
        for hyperparam in params_dict.values():
            assert isinstance(hyperparam,
                              collections.abc.Iterable) or callable(hyperparam)

        if use_max:
            best_validation_score = -np.inf
        else:
            best_validation_score = np.inf

        best_model = None
        all_scores = {}

        if logdir is not None:
            if not os.path.exists(logdir):
                os.makedirs(logdir, exist_ok=True)
            log_file = os.path.join(logdir, logfile)

        hyperparameter_combs = RandomHyperparamOpt.generate_random_hyperparam_values(
            params_dict, self.max_iter)

        for ind, model_params in enumerate(hyperparameter_combs):
            logger.info("Fitting model %d/%d" % (ind + 1, self.max_iter))
            logger.info("hyperparameters: %s" % str(model_params))

            hp_str = _convert_hyperparam_dict_to_filename(model_params)

            if logdir is not None:
                model_dir = os.path.join(logdir, hp_str+f"_e_{nb_epoch}")
                logger.info("model_dir is %s" % model_dir)
                try:
                    os.makedirs(model_dir)
                except OSError:
                    if not os.path.isdir(model_dir):
                        logger.info(
                            "Error creating model_dir, using tempfile directory"
                        )
                        model_dir = tempfile.mkdtemp()
            else:
                model_dir = tempfile.mkdtemp()

            model_params['model_dir'] = model_dir
            model = self.model_builder(**model_params)

            # mypy test throws error, so ignoring it in try
            try:
                model.fit(train_dataset, nb_epoch=nb_epoch)  # type: ignore
            # Not all models have nb_epoch
            except TypeError:
                model.fit(train_dataset)
            try:
                model.save()
            # Some models autosave
            except NotImplementedError:
                pass

            multitask_scores = model.evaluate(valid_dataset, [metric],
                                              output_transformers)
            valid_score = multitask_scores[metric.name]

            # Update best validation score so far
            if (use_max and valid_score >= best_validation_score) or (
                    not use_max and valid_score <= best_validation_score):
                best_validation_score = valid_score
                best_hyperparams = model_params
                best_model = model.model_dir
                all_scores[hp_str] = valid_score
            del model
            torch.cuda.empty_cache()


            # if `hyp_str` not in `all_scores`, store it in `all_scores`
            if hp_str not in all_scores:
                all_scores[hp_str] = valid_score

            logger.info("Model %d/%d, Metric %s, Validation set %s: %f" %
                        (ind + 1, nb_epoch, metric.name, ind, valid_score))
            logger.info("\tbest_validation_score so far: %f" %
                        best_validation_score)
        return best_model, best_hyperparams, all_scores

    @classmethod
    def generate_random_hyperparam_values(cls, params_dict: Dict,
                                          n: int):
        """Generates `n` random hyperparameter combinations of hyperparameter values

        Parameters
        ----------
        params_dict: Dict
            A dictionary of hyperparameters where parameter which takes discrete
            values are specified as iterables and continuous parameters are of
            type callables.
        n: int
            Number of hyperparameter combinations to generate

        Returns
        -------
        A list of generated hyperparameters

        Example
        -------
        >>> from scipy.stats import uniform
        >>> from deepchem.hyper import RandomHyperparamOpt
        >>> n = 1
        >>> params_dict = {'a': [1, 2, 3], 'b': [5, 7, 8], 'c': uniform(10, 5).rvs}
        >>> RandomHyperparamOpt.generate_random_hyperparam_values(params_dict, n)  # doctest: +SKIP
        [{'a': 3, 'b': 7, 'c': 10.619700740985433}]
        """
        print("helloooo", params_dict)
        hyperparam_keys, hyperparam_values = [], []
        for key, values in params_dict.items():
            if callable(values):
                # If callable, sample it for a maximum n times
                values = [values() for i in range(n)]
            hyperparam_keys.append(key)
            hyperparam_values.append(values)

        hyperparam_combs = []
        for iterable_hyperparam_comb in itertools.product(*hyperparam_values):
            hyperparam_comb = list(iterable_hyperparam_comb)
            hyperparam_combs.append(hyperparam_comb)

        indices = np.random.permutation(len(hyperparam_combs))[:n]
        params_subset = []
        for index in indices:
            param = {}
            for key, hyperparam_value in zip(hyperparam_keys,
                                             hyperparam_combs[index]):
                param[key] = hyperparam_value
            params_subset.append(param)
        return params_subset



def load_dataset(dataset: str, splitter: str, featurizer: str):
    load_fn = getattr(dc.molnet, f"load_{dataset}")
    loader = load_fn(featurizer=featurizers[featurizer](), splitter=splitter)
    tasks, dataset, transformers = loader
    print(f"loaded {dataset} successfully, splitter_type: {splitter}, featurizer: {featurizer}")
    print(f"tasks: {tasks}, transformers: {transformers}")
    logger.info(f"loaded {dataset} successfully, splitter_type: {splitter}, featurizer: {featurizer}")
    logger.info(f"tasks: {tasks}, transformers: {transformers}")
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

    print(f"tuning utils - task: {task}, use_max: {use_max}, metric_out: {metric_out}")
    logger.info(f"tuning utils - task: {task}, use_max: {use_max}, metric_out: {metric_out}")
    return task, use_max, metric, metric_out


def download_pretrained_model_from_S3(bucket_name, file_key, temp_dir):

    s3 = boto3.client('s3')
    temp_file_path = os.path.join(temp_dir.name, 'tmp.pt')

    # Download the file from S3
    s3.download_file(bucket_name, file_key, temp_file_path)
    print(f"File downloaded to temporary directory: {temp_file_path}")
    logger.info(f"File downloaded to temporary directory: {temp_file_path}")
    return temp_file_path


def modify_model_keys(model_path: str, temp_dir):
    # model_path = tempdir.name
    data = torch.load(model_path)
    data['model_state_dict'] = {key.replace("module.", ""): value for key, value in data['model_state_dict'].items()}

    checkpoint_path = os.path.join(temp_dir.name, 'checkpoint1.pt')
    with open(checkpoint_path, 'wb') as f:
        torch.save(data, f)

    print(f"model keys modified and stored successfully!")
    logger.info(f"model keys modified and stored successfully!")
    os.remove(model_path)
    return checkpoint_path


def load_and_restore(**params):
    ckpt_path = params['ckpt_path']
    n_tasks = len(params['tasks'])
    task = params['task']
    del params['ckpt_path']
    del params['tasks']
    del params['task']
    model = MoLFormer(n_tasks=n_tasks, task=task, **params)
    model.load_from_pretrained(ckpt_path)
    logger.info(f"model restored successfully from the checkpoint!")
    return model


def hyperparam_search(params, train, val, metric, transformers, epochs, use_max, max_iter, logdir=None):
    tuning_results = []
    for i, epoch in enumerate(epochs):
        # set use_max to false when using regression datasets 
        optimizer = RandomHyperparamOpt(load_and_restore, max_iter=max_iter)
        result = optimizer.hyperparam_search(params, train, val, metric, transformers, nb_epoch=epoch, use_max=use_max, logfile=f'results_{i}.txt', logdir=logdir)
        tuning_results.append(result)
        del optimizer
        torch.cuda.empty_cache()
    return tuning_results


def evaluate_model(tuning_results, test, metric, epochs, metric_out, transformers=[]):

    best_results = []
    print(tuning_results)
    for i in range(len(epochs)):
        params = tuning_results[i][1]
        if 'ckpt_path' in params:
            del params['ckpt_path']
        task = params['task']
        tasks = params['tasks']
        del params['task']
        del params['tasks']
        model = MoLFormer(n_tasks=len(tasks), task=task, **params)
        model.restore(os.path.join(tuning_results[i][0], 'checkpoint1.pt'))
        eval = model.evaluate(test, metrics=[metric], transformers=transformers)
        best_results.append(eval[metric_out])
    return best_results


def main(args):
    tasks, dataset, transformers = load_dataset(dataset=args.dataset, splitter=args.splitter, featurizer=args.featurizer)
    train, val, test = dataset

    task, use_max, metric, metric_out = get_tuning_utils(dataset=args.dataset)

    temp_dir = tempfile.TemporaryDirectory()
    tune_dir = os.path.join(temp_dir.name, 'tuning_runs')
    if not os.path.exists(tune_dir):
        os.mkdir(tune_dir)
    log_dir= os.path.join(temp_dir.name, 'tuning_results_metadata')
    if not os.path.exists(tune_dir):
        os.mkdir(tune_dir)
    print(f"downloading_model {args.file_key}")
    pretrained_model_file_path = download_pretrained_model_from_S3(bucket_name=args.bucket_name, file_key=args.file_key, temp_dir=temp_dir)
    modified_checkpoint_path = modify_model_keys(model_path=pretrained_model_file_path, temp_dir=temp_dir)
    # the parameters which are to be optimized
    params = {
    'ckpt_path': [os.path.dirname(modified_checkpoint_path)],
    'learning_rate': args.learning_rate,
    'batch_size': args.batch_size,
    'tasks': [tasks],
    'task': [task]
    }

    # Creating optimizer and searching over hyperparameters
    tuning_results = []
    tuning_results = hyperparam_search(params=params, train=train, val=val, metric=metric, transformers=transformers, epochs=args.epochs, use_max=use_max, max_iter=args.max_iter, logdir=log_dir)

    best_results = evaluate_model(tuning_results=tuning_results, test=test, metric=metric, epochs=args.epochs, metric_out=metric_out, transformers=transformers)

    results_arr = np.asarray(best_results)
    if use_max == True:
        best_score = max(results_arr)
        index = np.argmax(results_arr)
    else:
        best_score = min(results_arr)
        index = np.argmin(results_arr)
    best_params = tuning_results[index][1]
    best_params['epoch'] = args.epochs[index]
    best_model = tuning_results[index][0]
    print("best_score ->", best_score)
    print("best_params ->", best_params)
    return best_score, best_params, best_model

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

    argparser.add_argument('--learning_rate',
                           type=str,
                           help='learning rate',
                           default=None)

    argparser.add_argument('--batch_size',
                           type=str,
                           help='batch size',
                           default=None)

    argparser.add_argument('--epochs', type=str, help='epochs', default=None)
    argparser.add_argument('--max_iter', type=int, help='max_iter', default=None)
    args = argparser.parse_args()
    main(args)
