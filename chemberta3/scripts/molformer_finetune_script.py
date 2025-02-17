import argparse
import ast
import deepchem as dc
import numpy as np
from deepchem.models.torch_models import MoLFormer
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

featurizers = {'dummy': dc.feat.DummyFeaturizer}

regression_dataset = ['delaney', 'bace_regression', 'clearance', 'lipo']
classification_dataset = [
    'clintox', 'bace_classification', 'bbbp', 'hiv', 'sider', 'tox21'
]


class GridHyperparamOpt(HyperparamOpt):
    """
    Provides simple grid hyperparameter search capabilities.

    This class performs a grid hyperparameter search over the specified
    hyperparameter space. This implementation is simple and simply does
    a direct iteration over all possible hyperparameters and doesn't use
    parallelization to speed up the search.
    """

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
    ) -> Tuple[Model, Dict, Dict]:
        """Perform hyperparams search according to params_dict.

        Each key to hyperparams_dict is a model_param. The values should
        be a list of potential values for that hyperparam.

        Parameters
        ----------
        params_dict: Dict
            Maps hyperparameter names (strings) to lists of possible
            parameter values.
        train_dataset: Dataset
            dataset used for training
        valid_dataset: Dataset
            dataset used for validation(optimization on valid scores)
        metric: Metric
            metric used for evaluation
        output_transformers: list[Transformer]
            Transformers for evaluation. This argument is needed since
            `train_dataset` and `valid_dataset` may have been transformed
            for learning and need the transform to be inverted before
            the metric can be evaluated on a model.
        nb_epoch: int, (default 10)
            Specifies the number of training epochs during each iteration of optimization.
            Not used by all model types.
        use_max: bool, optional
            If True, return the model with the highest score. Else return
            model with the minimum score.
        logdir: str, optional
            The directory in which to store created models. If not set, will
            use a temporary directory.
        logfile: str, optional (default `results.txt`)
            Name of logfile to write results to. If specified, this is must
            be a valid file name. If not specified, results of hyperparameter
            search will be written to `logdir/results.txt`.

        Returns
        -------
        Tuple[`best_model`, `best_hyperparams`, `all_scores`]
        `(best_model, best_hyperparams, all_scores)` where `best_model` is
        the model dir of `dc.model.Model`, `best_hyperparams` is a
        dictionary of parameters, and `all_scores` is a dictionary mapping
        string representations of hyperparameter sets to validation
        scores."""

        hyperparams = params_dict.keys()
        hyperparam_vals = params_dict.values()
        for hyperparam_list in params_dict.values():
            assert isinstance(hyperparam_list, collections.abc.Iterable)

        number_combinations = reduce(mul,
                                     [len(vals) for vals in hyperparam_vals])

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

        for ind, hyperparameter_tuple in enumerate(
                itertools.product(*hyperparam_vals)):
            model_params = {}
            logger.info("Fitting model %d/%d" % (ind + 1, number_combinations))
            # Construction dictionary mapping hyperparameter names to values
            hyper_params = dict(zip(hyperparams, hyperparameter_tuple))
            for hyperparam, hyperparam_val in zip(hyperparams,
                                                  hyperparameter_tuple):
                model_params[hyperparam] = hyperparam_val
            logger.info("hyperparameters: %s" % str(model_params))

            hp_str = _convert_hyperparam_dict_to_filename(hyper_params)
            if logdir is not None:
                model_dir = os.path.join(logdir, hp_str + f"_e_{nb_epoch}")
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
            all_scores[hp_str] = valid_score

            # Stores the model directory in best_model and deletes the model instance to optimize caching efficiency.
            if (use_max and valid_score >= best_validation_score) or (
                    not use_max and valid_score <= best_validation_score):
                best_validation_score = valid_score
                best_hyperparams = hyper_params
                best_model = model.model_dir
            del model
            torch.cuda.empty_cache()

            logger.info(
                "Model %d/%d, Metric %s, Validation set %s: %f" %
                (ind + 1, number_combinations, metric.name, ind, valid_score))
            logger.info("\tbest_validation_score so far: %f" %
                        best_validation_score)
        return best_model, best_hyperparams, all_scores


def load_dataset(dataset: str, splitter: str, featurizer: str):
    """Loads the finetuning dataset, featurizes and splits it based on the specified featurizer and splitter.

    Parameters
    ----------
    dataset: str
        the name of the dataset to load. The user can choose from ['delaney', 'bace_regress', 'clearance', 'lipo', 'clintox', 'bace_class', 'bbbp', 'hiv', 'sider', 'tox21']
    splitter: str
        the name of the splitter to be used.
    featurizer: str
        the name of the splitter to be used.

    Returns
    -------
    Tuple[tasks, dataset, transformers]
    """
    load_fn = getattr(dc.molnet, f"load_{dataset}")
    loader = load_fn(featurizer=featurizers[featurizer](), splitter=splitter)
    tasks, dataset, transformers = loader
    logger.info(
        f"loaded {dataset} successfully, splitter_type: {splitter}, featurizer: {featurizer}"
    )
    logger.info(f"tasks: {tasks}, transformers: {transformers}")
    return tasks, dataset, transformers


def get_tuning_utils(dataset: str):
    """Fetches the training utils for finetuning task based on the dataset used.

    Parameters
    ----------
    dataset: str
        name of the dataset to be used for finetuning task.

    Returns
    -------
    Tuple[task, use_max, metric, metric_out]
    where, if task = `regression` use_max should be set to False and metric used should be RMS Score.
    when task = `classification` use_max should be set to False and metric used should be ROC AUC Score.
    """
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

    logger.info(
        f"tuning utils - task: {task}, use_max: {use_max}, metric_out: {metric_out}"
    )
    return task, use_max, metric, metric_out


def download_pretrained_model_from_S3(bucket_name: str, file_key: str,
                                      temp_dir: tempfile.TemporaryDirectory):
    """Downloads pretrained model from S3

    Parameters
    ----------
    bucket_name: str
        S3 bucket_name where the pretrained model is stored
    file_key: str
        S3 file path of the pretrained model
    temp_dir: tempfile.TemporaryDirectory
        temporary directory to store the downloaded pretrained model from S3

    Returns
    -------
    temp_file_path: str
        temporary file path where the pretrained model is stored
    """
    s3 = boto3.client('s3')
    temp_file_path = os.path.join(temp_dir.name, 'tmp.pt')

    # Download the file from S3
    s3.download_file(bucket_name, file_key, temp_file_path)
    logger.info(f"File downloaded to temporary directory: {temp_file_path}")
    return temp_file_path


def modify_model_keys(model_path: str, temp_dir: tempfile.TemporaryDirectory):
    """Modifies the keys of the downloaded model by removing the "module." prefix from the model keys if present.  
    This is necessary when the model was trained using Distributed Data Parallel (DDP), as DDP automatically 
    prepends "module." to the parameter keys to distinguish them in a multi-GPU setup. Removing the prefix 
    ensures compatibility with non-DDP environments for inference or further training.
    
    Parameters
    ----------
    model_path: str
        file path of the downloaded pretrained model
    temp_dir: tempfile.TemporaryDirectory
        temporary directory to store the modified pretrained model

    Returns
    -------
    checkpoint_path: str
        the temporary path of the modified pretrained model checkpoint
    """
    data = torch.load(model_path)
    data['model_state_dict'] = {
        key.replace("module.", ""): value
        for key, value in data['model_state_dict'].items()
    }

    checkpoint_path = os.path.join(temp_dir.name, 'checkpoint1.pt')
    with open(checkpoint_path, 'wb') as f:
        torch.save(data, f)
    logger.info(f"model keys modified and stored successfully!")
    os.remove(model_path)
    return checkpoint_path


def load_and_restore(**params):
    """
    Loads and restores the MoLFormer model from a checkpoint using `load_from_pretrained` method.

    Parameters
    ----------
    params: Dict
        A dictionary containing the following keys:
        - 'ckpt_path': Path to the directory of the modified checkpoint.
        - 'learning_rate': Learning rate for the model.
        - 'batch_size': Batch size for training or inference.
        - 'tasks': List of tasks associated with the model.
        - 'task': Specific task to be performed.
    """
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


def hyperparam_search(params: Dict,
                      train: Dataset,
                      val: Dataset,
                      metric: Metric,
                      transformers: List,
                      epochs: List,
                      use_max: bool,
                      logdir: Optional[str] = None):
    """Provides simple grid hyperparameter search capabilities.

    Parameters
    ----------
    params_dict: Dict
        Maps hyperparameter names (strings) to lists of possible
        parameter values.
    train_dataset: Dataset
        dataset used for training
    valid_dataset: Dataset
        dataset used for validation(optimization on valid scores)
    metric: Metric
        metric used for evaluation
    transformers: list[Transformer]
        Transformers for evaluation. This argument is needed since
        `train_dataset` and `valid_dataset` may have been transformed
        for learning and need the transform to be inverted before
        the metric can be evaluated on a model.
    epochs: list[int]
        Specifies the list of training epochs during each iteration of hyperparameter search.
    use_max: bool, optional
        If True, return the model with the highest score. Else return
        model with the minimum score.
    logdir: str, optional
        The directory in which to store created models. If not set, will
        use a temporary directory.
    """
    tuning_results = []
    for i, epoch in enumerate(epochs):
        # set use_max to false when using regression datasets
        optimizer = GridHyperparamOpt(load_and_restore)
        result = optimizer.hyperparam_search(params,
                                             train,
                                             val,
                                             metric,
                                             transformers,
                                             nb_epoch=epoch,
                                             use_max=use_max,
                                             logfile=f'results_{i}.txt',
                                             logdir=logdir)
        tuning_results.append(result)
        del optimizer
        torch.cuda.empty_cache()
    logger.info(f"hyperparameter search done!")
    return tuning_results


def evaluate_model(tuning_results: List,
                   test: Dataset,
                   metric: Metric,
                   epochs: List,
                   metric_out: str,
                   transformers=[]):
    """Evaluates the test dataset on the best model parameters obtained from the hyperparameter search.

    Parameters
    ----------
    tuning_results: List
        List of the Tuple[`best_model`, `best_hyperparams`, `all_scores`] obtained from the hyperparameter search
    test: Dataset
        dataset used for testing
    metric: Metric
        metric used for evaluation
    epochs: List
        The list of epochs
    metric_out: str
        name of the metric used for evaluation
    transformers: list[Transformer]
        Transformers for evaluation. This argument is needed since
        `train_dataset` and `valid_dataset` may have been transformed
        for learning and need the transform to be inverted before
        the metric can be evaluated on a model.
    """
    best_results = []
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
        eval = model.evaluate(test,
                              metrics=[metric],
                              transformers=transformers)
        best_results.append(eval[metric_out])
    return best_results


def filter_datasets(dataset):

    train, val, test = dataset
    selected_train_indices = [i for i, x in enumerate(train.X) if len(x) < 200]
    selected_val_indices = [i for i, x in enumerate(val.X) if len(x) < 200]
    selected_test_indices = [i for i, x in enumerate(test.X) if len(x) < 200]
    # Apply the selection
    filtered_train_dataset = train.select(selected_train_indices)
    filtered_val_dataset = val.select(selected_val_indices)
    filtered_test_dataset = test.select(selected_test_indices)
    return filtered_train_dataset, filtered_val_dataset, filtered_test_dataset


def main(args):

    tasks, dataset, transformers = load_dataset(dataset=args.dataset,
                                                splitter=args.splitter,
                                                featurizer=args.featurizer)
    learning_rate = ast.literal_eval(args.learning_rate)
    epochs = ast.literal_eval(args.epochs)
    batch_size = ast.literal_eval(args.batch_size)
    # train, val, test = dataset
    filtered_train_dataset, filtered_val_dataset, filtered_test_dataset = filter_datasets(
        dataset)

    task, use_max, metric, metric_out = get_tuning_utils(dataset=args.dataset)

    temp_dir = tempfile.TemporaryDirectory()
    tune_dir = os.path.join(temp_dir.name, 'tuning_runs')
    if not os.path.exists(tune_dir):
        os.mkdir(tune_dir)
    log_dir = os.path.join(temp_dir.name, 'tuning_results_metadata')
    if not os.path.exists(tune_dir):
        os.mkdir(tune_dir)

    pretrained_model_file_path = download_pretrained_model_from_S3(
        bucket_name=args.bucket_name,
        file_key=args.file_key,
        temp_dir=temp_dir)
    modified_checkpoint_path = modify_model_keys(
        model_path=pretrained_model_file_path, temp_dir=temp_dir)

    # the parameters which are to be optimized
    params = {
        'ckpt_path': [os.path.dirname(modified_checkpoint_path)],
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'tasks': [tasks],
        'task': [task]
    }

    # Creating optimizer and searching over hyperparameters
    tuning_results = []
    tuning_results = hyperparam_search(params=params,
                                       train=filtered_train_dataset,
                                       val=filtered_val_dataset,
                                       metric=metric,
                                       transformers=transformers,
                                       epochs=epochs,
                                       use_max=use_max,
                                       logdir=log_dir)
    best_results = evaluate_model(tuning_results=tuning_results,
                                  test=filtered_test_dataset,
                                  metric=metric,
                                  epochs=epochs,
                                  metric_out=metric_out,
                                  transformers=transformers)

    results_arr = np.asarray(best_results)
    if use_max == True:
        best_score = max(results_arr)
        index = np.argmax(results_arr)
    else:
        best_score = min(results_arr)
        index = np.argmin(results_arr)
    best_params = tuning_results[index][1]
    best_params['epoch'] = epochs[index]
    best_model = tuning_results[index][0]
    return best_score, best_params, best_model


if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset',
                           type=str,
                           help='name of finetuning dataset',
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

    args = argparser.parse_args()
    main(args)
