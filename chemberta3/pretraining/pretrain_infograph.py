import argparse
import os
import torch
from ray import tune
from ray.tune import CLIReporter

from chemberta3.pretraining.loaders import ZincLoader
from deepchem.feat import MolGraphConvFeaturizer
from deepchem.models.torch_models import InfoGraphModel
from deepchem.splits import RandomSplitter


def train(args):
    featurizer = MolGraphConvFeaturizer(use_edges=True)
    loader = ZincLoader(featurizer=featurizer)

    ds = loader.load_shards(
        shards_to_load=args.num_shards,
        max_num_samples=args.max_num_samples,
        cleanup=True,
        parallel=True,
    )
    splitter = RandomSplitter()
    train_ds, valid_ds, test_ds = splitter.train_valid_test_split(
        ds, frac_train=1.0 - args.frac_valid, frac_valid=args.frac_valid, frac_test=0.0
    )
    num_feat = max([ds.X[i].num_node_features for i in range(min(len(ds), 1e5))])
    edge_dim = max([ds.X[i].num_edge_features for i in range(min(len(ds), 1e5))])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"

    infograph_build_kwargs = {
        "num_features": num_feat,
        "edge_features": edge_dim,
        "use_unsup_loss": True,
        "separate_encoder": False,
        "device": device,
    }

    def _train_infograph(config):
        with tune.checkpoint_dir(0) as checkpoint_dir:
            model = InfoGraphModel(
                **infograph_build_kwargs,
                dim=config["dim"],
                learning_rate=config["lr"],
                model_dir=checkpoint_dir,
            )
        # TODO: compute validation loss; what does this mean with infograph?
        loss = model.fit(train_ds, nb_epoch=args.num_epochs)
        tune.report(loss=loss)
        return

    config = {
        "dim": tune.lograndint(4, 9, base=2),
        "lr": tune.loguniform(1e-4, 1e-1),
    }
    reporter = CLIReporter(metric_columns=["loss"])
    result = tune.run(
        _train_infograph,
        resources_per_trial={
            "cpu": os.cpu_count(),
            "gpu": 1 if torch.cuda.is_available() else 0,
        },
        config=config,
        num_samples=args.num_samples,
        progress_reporter=reporter,
    )
    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    best_trained_model = InfoGraphModel(
        **infograph_build_kwargs,
        dim=best_trial.config["dim"],
        learning_rate=best_trial.config["lr"],
        model_dir=args.output_dir,
    )
    best_trained_model.load_pretrained_components(
        model_dir=best_trial.checkpoint.dir_or_data
    )
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    best_trained_model.save_checkpoint(
        max_checkpoints_to_keep=1, model_dir=args.output_dir
    )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--max_num_samples", type=int, default=2.5e5)
    argparser.add_argument("--num_shards", type=int, default=1)
    argparser.add_argument(
        "--num_epochs", type=int, default=1, help="Number of epochs per trial."
    )
    argparser.add_argument("--patience", type=int, default=5)
    argparser.add_argument("--seed", type=int, default=123)
    argparser.add_argument("--frac_valid", type=float, default=0.01)
    argparser.add_argument("--output_dir", type=str, default="./best_infograph_model")
    argparser.add_argument("--num_samples", type=int, default=10)
    args = argparser.parse_args()
    train(args)
