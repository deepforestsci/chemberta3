import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch

import deepchem as dc
from deepchem.feat import MolGraphConvFeaturizer, CircularFingerprint

from custom_datasets import load_nek
from model_loaders import load_infograph, load_random_forest

DATASET_MAPPING = {
    "bace_classification": {
        "loader": dc.molnet.load_bace_classification,
        "output_type": "classification",
    },
    "bace_regression": {
        "loader": dc.molnet.load_bace_regression,
        "output_type": "regression",
    },
    "bbbp": {
        "loader": dc.molnet.load_bbbp,
        "output_type": "classification",
    },
    "clintox": {
        "loader": dc.molnet.load_clintox,
        "output_type": "classification",
        "tasks_wanted": ["CT_TOX"],
    },
    "delaney": {
        "loader": dc.molnet.load_delaney,
        "output_type": "regression",
    },
    "hiv": {
        "loader": dc.molnet.load_hiv,
        "output_type": "classification",
    },
    "muv": {"loader": dc.molnet.load_muv, "output_type": "classification"},
    "pcba": {"loader": dc.molnet.load_pcba, "output_type": "classification"},
    "qm9": {
        "dataset_type": "regression",
        "load_fn": dc.molnet.load_qm9,
    },
    "tox21": {
        "loader": dc.molnet.load_tox21,
        "output_type": "classification",
        "tasks_wanted": ["SR-p53"],
    },
    "nek": {
        "loader": load_nek,
        "output_type": "regression",
        "tasks_wanted": ["NEK2_ki_avg_value"],
    },
}

MODEL_MAPPING = {
    "infograph": load_infograph,
    "random_forest": load_random_forest,
}

FEATURIZER_MAPPING = {
    "molgraphconv": MolGraphConvFeaturizer(use_edges=True),
    "ecfp": CircularFingerprint(),
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
        self, dataset_name: str, featurizer: dc.feat.Featurizer, data_dir: Optional[str] = None
    ) -> Tuple[List[str], Tuple[dc.data.Dataset, ...], List[dc.trans.Transformer], str]:
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
            raise ValueError(f"Dataset {dataset_name} not found in dataset mapping.")

        dataset_loader = self.dataset_mapping[dataset_name]["loader"]
        output_type = self.dataset_mapping[dataset_name]["output_type"]
        tasks, datasets, transformers = dataset_loader(
            featurizer=featurizer, splitter=None, data_dir=data_dir
        )
        return tasks, datasets, transformers, output_type


class BenchmarkingModelLoader:
    """A utility class for helping to load models for benchmarking.

    This class is used to load models for benchmarking. It is used to load relevant pre-trained models
    """

    def __init__(
        self, loss: dc.models.losses.Loss, metrics: List[dc.metrics.Metric]
    ) -> None:
        """Initialize a BenchmarkingModelLoader.

        Parameters
        ----------
        loss: dc.models.losses.Loss
            Loss function to use.
        metrics: List[dc.metrics.Metric]
            List of metrics to use.
        """
        self.loss = loss
        self.metrics = metrics
        self.model_mapping = MODEL_MAPPING

    def load_model(
        self,
        model_name: str,
        output_type: str,
        checkpoint_path: str = None,
        model_loading_kwargs: Dict = {},
    ) -> dc.models.torch_models.modular.ModularTorchModel:
        """Load a model.

        Parameters
        ----------
        model_name: str
            Name of the model to load. Should be a key in `self.model_mapping`.
        checkpoint_path: str, optional (default None)
            Path to checkpoint to load. If None, will not load a checkpoint and will return a new model.
        model_loading_kwargs: Dict, optional (default {})
            Keyword arguments to pass to the model loader.

        Returns
        -------
        model: dc.models.torch_models.modular.ModularTorchModel
            Loaded model.
        """
        if model_name not in self.model_mapping:
            raise ValueError(f"Model {model_name} not found in model mapping.")

        model_loader = self.model_mapping[model_name]
        model = model_loader(
            metrics=self.metrics,
            checkpoint_path=checkpoint_path,
            output_type=output_type,
            **model_loading_kwargs,
        )
        return model


def get_infograph_loading_kwargs(dataset):
    num_feat = max([dataset.X[i].num_node_features for i in range(len(dataset))])
    edge_dim = max([dataset.X[i].num_edge_features for i in range(len(dataset))])
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


def train(args):
    """Training loop

    Trains the specified model on the specified dataset using the specified featurizer,
    based on the command line arguments provided.

    Writes metrics to the specified output directory.
    """
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset_loader = BenchmarkingDatasetLoader()
    featurizer_loader = BenchmarkingFeaturizerLoader()

    splitter = dc.splits.ScaffoldSplitter()
    featurizer = featurizer_loader.load_featurizer(args.featurizer_name)

    tasks, datasets, transformers, output_type = dataset_loader.load_dataset(
        args.dataset_name, featurizer, args.data_dir
    )
    unsplit_dataset = datasets[0]
    train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
        unsplit_dataset
    )

    metrics = (
        [dc.metrics.Metric(dc.metrics.pearson_r2_score)]
        if output_type == "regression"
        else [dc.metrics.Metric(dc.metrics.auc, mode="classification")]
    )
    loss = (
        dc.models.losses.L2Loss()
        if output_type == "regression"
        else dc.models.losses.BinaryCrossEntropy()
    )

    model_loader = BenchmarkingModelLoader(loss=loss, metrics=metrics)
    model_loading_kwargs = {}
    if args.model_name == "infograph":
        model_loading_kwargs = get_infograph_loading_kwargs(train_dataset)

    model = model_loader.load_model(
        model_name=args.model_name,
        output_type=output_type,
        checkpoint_path=args.checkpoint,
        model_loading_kwargs = model_loading_kwargs
    )

    early_stopper = EarlyStopper(patience=args.patience)

    if isinstance(model, dc.models.SklearnModel):
        model.fit(train_dataset)
    else:
        for epoch in range(args.num_epochs):
            training_loss_value = model.fit(train_dataset, nb_epoch=1)
            eval_preds = model.predict(valid_dataset)
            eval_loss_fn = loss._create_pytorch_loss()
            eval_loss = torch.sum(
                eval_loss_fn(torch.Tensor(eval_preds), torch.Tensor(valid_dataset.y))
            ).item()

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
        {k: np.array(v) for k, v in test_metrics.items()}, orient="index"
    )
    print(f"Test metrics: {test_metrics_df}")
    test_metrics_df.to_csv(
        f"{args.output_dir}/{args.model_name}_{args.dataset_name}_test_metrics.csv",
    )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model_name", type=str, default="infograph")
    argparser.add_argument("--featurizer_name", type=str, default="molgraphconv")
    argparser.add_argument("--dataset_name", type=str, default="nek")
    argparser.add_argument("--checkpoint", type=str, default=None)
    argparser.add_argument("--num_epochs", type=int, default=50)
    argparser.add_argument("--patience", type=int, default=5)
    argparser.add_argument("--seed", type=int, default=123)
    argparser.add_argument("--output_dir", type=str, default=".")
    argparser.add_argument("--data-dir", type=str, required=False, default=None)
    args = argparser.parse_args()
    train(args)
