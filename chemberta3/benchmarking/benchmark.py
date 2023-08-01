import os
import yaml
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union, Optional

import numpy as np
import pandas as pd
import torch

import deepchem as dc
from deepchem.models import GraphConvModel, WeaveModel
from deepchem.models.torch_models import GroverModel

from custom_datasets import load_nek, load_zinc250k, prepare_data, FEATURIZER_MAPPING
from model_loaders import load_infograph, load_chemberta, load_random_forest

DATASET_MAPPING = {
    "bace_classification": {
        "loader": dc.molnet.load_bace_classification,
        "output_type": "classification",
        "n_tasks": 1,
    },
    "bace_regression": {
        "loader": dc.molnet.load_bace_regression,
        "output_type": "regression",
        "n_tasks": 1,
    },
    "bbbp": {
        "loader": dc.molnet.load_bbbp,
        "output_type": "classification",
        "n_tasks": 1,
    },
    "clintox": {
        "loader": dc.molnet.load_clintox,
        "output_type": "classification",
        "tasks_wanted": ["CT_TOX"],
        "n_tasks": 2,
    },
    "delaney": {
        "loader": dc.molnet.load_delaney,
        "output_type": "regression",
        "n_tasks": 1,
    },
    "hiv": {
        "loader": dc.molnet.load_hiv,
        "output_type": "classification",
        "n_tasks": 1,
    },
    "muv": {
        "loader": dc.molnet.load_muv,
        "output_type": "classification",
        "n_tasks": 17
    },
    "pcba": {
        "loader": dc.molnet.load_pcba,
        "output_type": "classification",
        "n_tasks": 128
    },
    "qm9": {
        "output_type": "regression",
        "loader": dc.molnet.load_qm9,
        "n_tasks": 12,
    },
    "tox21": {
        "loader": dc.molnet.load_tox21,
        "output_type": "classification",
        # TODO How to use `tasks_wanted` argument?
        "tasks_wanted": ["SR-p53"],
        # `tasks_wanted` will only be used if we are creating dataset from csv but here we are using
        # molnet loader and therefore n_tasks will be the number of tasks returned from molnet loader
        "n_tasks": 12,
    },
    "nek": {
        "loader": load_nek,
        "output_type": "regression",
        "tasks_wanted": ["NEK2_ki_avg_value"],
        "n_tasks": 1,
    },
    "zinc250k": {
        "loader": load_zinc250k,
        "output_type": "regression",
        "tasks_wanted": ["logp"],
    }
}

MODEL_MAPPING = {
    "infograph": load_infograph,
    "random_forest": load_random_forest,
    "graphconv": GraphConvModel,
    "weave": WeaveModel,
    "chemberta": load_chemberta,
    "GroverPretrain": GroverModel,
}


class BenchmarkingDatasetLoader:
    """A utility class for helping to load datasets for benchmarking.

    This class is used to load datasets for benchmarking. It is used to load relevant MoleculeNet datasets
    and other custom datasets (e.g. NEK datasets).
    """

    def __init__(self) -> None:
        self.dataset_mapping = DATASET_MAPPING

    @property
    def dataset_names(self) -> List[str]:
        return list(self.dataset_mapping.keys())

    def load_dataset(
        self,
        dataset_name: str,
        featurizer: dc.feat.Featurizer,
        data_dir: Optional[str] = None,
        **kwargs
    ) -> Tuple[List[str], Tuple[dc.data.Dataset, ...],
               List[dc.trans.Transformer], str]:
        """Load a dataset.

        Parameters
        ----------
        dataset_name: str
            Name of the dataset to load. Should be a key in `self.dataset_mapping`.
        featurizer: dc.feat.Featurizer
            Featurizer to use.
        data_dir: str
            Directory of dataset

        Returns
        -------
        tasks: List[str]
            List of tasks.
        datasets: Tuple[Dataset, ...]
            Tuple of train, valid, test datasets.
        transformers: List[dc.trans.Transformer]
            List of transformers.
        output_type: str
            Type of output (e.g. "classification" or "regression").
        """
        if dataset_name not in self.dataset_mapping:
            raise ValueError(
                f"Dataset {dataset_name} not found in dataset mapping.")

        dataset_loader = self.dataset_mapping[dataset_name]["loader"]
        output_type = self.dataset_mapping[dataset_name]["output_type"]
        n_tasks = self.dataset_mapping[dataset_name]["n_tasks"]
        tasks, datasets, transformers = dataset_loader(featurizer=featurizer,
                                                       splitter=None,
                                                       data_dir=data_dir)
        return tasks, datasets, transformers, output_type, n_tasks


class BenchmarkingModelLoader:
    """A utility class for helping to load models for benchmarking.

    This class is used to load models for benchmarking. It is used to load relevant pre-trained models
    """

    def __init__(self) -> None:
        """Initialize a BenchmarkingModelLoader.
        """
        self.model_mapping = MODEL_MAPPING

    def load_model(
        self,
        model_name: str,
        checkpoint_path: Optional[str] = None,
        from_hf_checkpoint=False,
        model_parameters: Dict = {},
        task: str = 'regression',
        tokenizer_path: Optional[str] = None,
    ) -> Union[dc.models.torch_models.modular.ModularTorchModel,
               dc.models.torch_models.TorchModel]:
        """Load a model.

        Parameters
        ----------
        model_name: str
            Name of the model to load. Should be a key in `self.model_mapping`.
        checkpoint_path: str, optional (default None)
            Path to checkpoint to load. If None, will not load a checkpoint and will return a new model.
        model_parameters: Dict, optional (default {})
            Parameters for the model 
        task: str, (default regression)
            The specific training task configuration for the model.
        from_hf_checkpoint: bool, (default False)
            Specify whether the checkpoint is a huggingface checkpoint
        tokenizer_path: str (None)
            Path to huggingface tokenizer. This option is used only for models from HuggingFace ecosystem, like chemberta and not other models.

        Returns
        -------
        model: dc.models.torch_models.modular.ModularTorchModel
            Loaded model.
        """
        if model_name not in self.model_mapping:
            raise ValueError(f"Model {model_name} not found in model mapping.")
        model_loader = self.model_mapping[model_name]
        if model_name == 'chemberta':
            model = model_loader(task=task, tokenizer_path=tokenizer_path)
        else:
            model = model_loader(**model_parameters)
        if checkpoint_path is not None:
            if model_name == 'chemberta':
                # a special case for chemberta model - chemberta model can also be loaded from
                # huggingface checkpoint while other models (deepchem models) can only be loaded
                # from deepchem checkpoint and hence, don't have the `from_hf_checkpoint` argument
                model.load_from_pretrained(
                    model_dir=checkpoint_path,
                    from_hf_checkpoint=from_hf_checkpoint)
            else:
                model.load_pretrained_components(checkpoint=checkpoint_path)
        return model


def get_infograph_loading_kwargs(dataset):
    num_feat = max(
        [dataset.X[i].num_node_features for i in range(len(dataset))])
    edge_dim = max(
        [dataset.X[i].num_edge_features for i in range(len(dataset))])
    return {"num_feat": num_feat, "edge_dim": edge_dim}


class BenchmarkingFeaturizerLoader:
    """A utility class for helping to load featurizers for benchmarking."""

    def __init__(self) -> None:
        self.featurizer_mapping = FEATURIZER_MAPPING

    def load_featurizer(self, featurizer_name: str) -> dc.feat.Featurizer:
        """Load a featurizer.

        Parameters
        ----------
        featurizer_name: str
            Name of the featurizer to load. Should be a key in `self.featurizer_mapping`.

        Returns
        -------
        featurizer: dc.feat.Featurizer
            Loaded featurizer.
        """
        if featurizer_name not in self.featurizer_mapping:
            raise ValueError(
                f"Featurizer {featurizer_name} not found in featurizer mapping."
            )

        featurizer = self.featurizer_mapping[featurizer_name]
        return featurizer


@dataclass
class EarlyStopper:
    """Early stopper for benchmarking."""

    patience: int = 5
    min_delta: float = 0.0
    min_loss: float = np.inf
    best_epoch: int = 0

    def __call__(self, loss: float, epoch: int) -> bool:
        if loss < self.min_loss:
            self.min_loss = loss
            self.best_epoch = epoch
            return False
        elif loss - self.min_loss > self.min_delta:
            if epoch - self.best_epoch > self.patience:
                return True
        return False


def train(args, data_dir: str):
    """Training loop

    Trains the specified model on the specified dataset using the specified featurizer,
    based on the command line arguments provided.

    Writes metrics to the specified output directory.

    Parameters
    ----------
    data_dir: str
        Data directory for loading dataset
    """
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dataset = dc.data.DiskDataset(data_dir=data_dir)

    # Load model
    model_loader = BenchmarkingModelLoader()
    model_loading_kwargs = {}
    if args.model_name == "infograph":
        model_loading_kwargs = get_infograph_loading_kwargs(train_dataset)
    elif args.model_name == "graphconv" or args.model_name == "weave":
        model_loading_kwargs = {'n_tasks': n_tasks, 'mode': output_type}
    model = model_loader.load_model(model_name=args.model_name,
                                    checkpoint_path=args.checkpoint,
                                    model_parameters=model_loading_kwargs,
                                    task=args.task)

    early_stopper = EarlyStopper(patience=args.patience)

    metrics = ([dc.metrics.Metric(dc.metrics.pearson_r2_score)]
               if output_type == "regression" else
               [dc.metrics.Metric(dc.metrics.roc_auc_score)])

    if isinstance(model, dc.models.SklearnModel):
        model.fit(train_dataset)
    else:
        for epoch in range(args.num_epochs):
            training_loss_value = model.fit(train_dataset, nb_epoch=1)
            eval_preds = model.predict(valid_dataset)
            eval_loss_fn = loss._create_pytorch_loss()
            eval_loss = torch.sum(
                eval_loss_fn(torch.Tensor(eval_preds),
                             torch.Tensor(valid_dataset.y))).item()

            eval_metrics = model.evaluate(
                valid_dataset,
                metrics=metrics,
            )
            print(
                f"Epoch {epoch} training loss: {training_loss_value}; validation loss: {eval_loss}; validation metrics: {eval_metrics}"
            )
            if early_stopper(eval_loss, epoch):
                break

    # compute test metrics
    test_metrics = model.evaluate(test_dataset, metrics=metrics)
    test_metrics_df = pd.DataFrame.from_dict(
        {k: np.array(v) for k, v in test_metrics.items()}, orient="index")
    print(f"Test metrics: {test_metrics_df}")
    test_metrics_df.to_csv(
        f"{args.output_dir}/{args.model_name}_{args.dataset_name}_test_metrics.csv",
    )


def evaluate(seed: int,
             featurizer_name: str,
             dataset_name: str,
             model_name: str,
             checkpoint_path: str,
             task: Optional[str] = None,
             tokenizer_path: Optional[str] = None,
             from_hf_checkpoint: Optional[bool] = None):
    """Evaluate method

    Evaluates the specified model on the specified dataset using the specified featurizer,
    based on the command line arguments provided.

    Parameters
    ----------
    seed: int
        Manual seed for generating random numbers
    featurizer_name: str
        Featurizer name to featurize dataset
    dataset_name: str
        Dataset to evaluate the model
    model_name: str
        Name of the model to evaluate
    checkpoint_path: str
        Path to model checkpoint
    task: str, (optional, default None)
        The task defines the type of learning task in the huggingface model. The supported tasks are
         - `mlm` - masked language modeling commonly used in pretraining
         - `mtr` - multitask regression - a task used for both pretraining base models and finetuning
         - `regression` - use it for regression tasks, like property prediction
         - `classification` - use it for classification tasks
        Note: The argument is valid only for HuggingFace models.
    from_hf_checkpoint: bool (default None)
        Load model from huggingface checkpoint (valid only for huggingface models like chemberta3)
    tokenizer_path: str (default None)
        Path to pretrained tokenizer (the option is valid only for huggingface models like chemberta3)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset_loader = BenchmarkingDatasetLoader()
    featurizer_loader = BenchmarkingFeaturizerLoader()

    splitter = dc.splits.ScaffoldSplitter()
    featurizer = featurizer_loader.load_featurizer(featurizer_name)

    tasks, datasets, transformers, output_type = dataset_loader.load_dataset(
        dataset_name, featurizer)
    unsplit_dataset = datasets[0]
    train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
        unsplit_dataset)

    if task == 'mlm':
        metrics = [dc.metrics.Metric(dc.metrics.accuracy_score)]

    model_loader = BenchmarkingModelLoader(metrics=metrics)
    if args.model_name == "infograph":
        model_loading_kwargs = get_infograph_loading_kwargs(train_dataset)

    model = model_loader.load_model(model_name=model_name,
                                    checkpoint_path=checkpoint_path,
                                    from_hf_checkpoint=from_hf_checkpoint,
                                    task=task,
                                    tokenizer_path=tokenizer_path)

    test_metrics = model.evaluate(test_dataset, metrics=metrics)
    print(test_metrics)
    return


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config',
                           type=argparse.FileType('r'),
                           help='config file path',
                           default=None)
    argparser.add_argument('--train',
                           help='train a model',
                           default=False,
                           action='store_true')
    argparser.add_argument('--evaluate',
                           help='evaluate a model',
                           default=False,
                           action='store_true')
    argparser.add_argument('--prepare_data',
                           help='parse data',
                           default=False,
                           action='store_true')
    argparser.add_argument("--model_name", type=str, default="infograph")
    argparser.add_argument("--task", type=str, default="regression")
    argparser.add_argument("--featurizer_name",
                           type=str,
                           default="molgraphconv")
    argparser.add_argument("--dataset_name", type=str, default="nek")
    argparser.add_argument("--checkpoint", type=str, default=None)
    argparser.add_argument("--num_epochs", type=int, default=50)
    argparser.add_argument("--patience", type=int, default=5)
    argparser.add_argument("--seed", type=int, default=123)
    argparser.add_argument("--output_dir", type=str, default=".")
    argparser.add_argument("--data-dir", type=str, required=False, default=None)
    # NOTE There might be a better argument than job
    argparser.add_argument("--job", type=str, default="train")
    argparser.add_argument("--from-hf-checkpoint",
                           action=argparse.BooleanOptionalAction)
    args = argparser.parse_args()

    if args.config:
        config_dict = yaml.load(args.config, Loader=yaml.FullLoader)
        arg_dict = args.__dict__
        for key, value in config_dict.items():
            arg_dict[key] = value

    if args.prepare_data:
        prepare_data(dataset_name=args.dataset_name,
                     featurizer_name=args.featurizer_name,
                     data_dir=args.data_dir)

    # FIXME All config's need not have model name, model parameters and others
    base_exp_dir = 'runs'
    model_parameters = config_dict['model_parameters']
    leaf_dir = '-'.join([config_dict['model_name'], model_parameters['task'], str(model_parameters['hidden_size'])])
    exp_dir = os.path.join(config_dict['experiment_name'], config_dict['dataset_name'], leaf_dir)
    os.makedirs(exp_dir, exist_ok=True)

    if args.train:
        train(args, data_dir=args.data_dir)
    if args.evaluate:
        evaluate(seed=args.seed,
                 featurizer_name=args.featurizer_name,
                 dataset_name=args.dataset_name,
                 model_name=args.model_name,
                 checkpoint_path=args.checkpoint_path,
                 task=args.task,
                 tokenizer_path=args.tokenizer_path,
                 from_hf_checkpoint=args.from_hf_checkpoint)
