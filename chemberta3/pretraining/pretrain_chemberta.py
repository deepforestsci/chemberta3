import argparse
import os
import torch
import math
from ray import tune
from ray.tune import CLIReporter

from chemberta3.pretraining.loaders import ZincLoader

from deepchem.models.torch_models.chemberta import Chemberta
from deepchem.splits import RandomSplitter


class IdentityFeaturizer:
    def __init__(self):
        pass

    def featurize(self, smiles):
        return smiles


def train(args):
    featurizer = IdentityFeaturizer()
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

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"

    chemberta_build_args = {
        "device": device,
    }

    def _train_chemberta(config):
        tokenizer_path = "seyonec/PubChem10M_SMILES_BPE_60k"
        # update config so hidden size is a multiple of num_attention_heads
        config["hidden_size"] = int(
            math.ceil(config["hidden_size"] / config["num_attention_heads"])
            * config["num_attention_heads"]
        )
        with tune.checkpoint_dir(0) as checkpoint_dir:
            model = Chemberta(
                task="mlm",
                model_config_kwargs=config,
                model_dir=checkpoint_dir,
                tokenizer_path=tokenizer_path,
            )
        # TODO: compute validation loss; how to do this with infograph?
        train_loss = model.fit(train_ds, nb_epoch=args.num_epochs)
        tune.report(loss=train_loss)
        return

    config = {
        "hidden_size": tune.lograndint(64, 512, base=2),
        "num_hidden_layers": tune.randint(3, 8),
        "num_attention_heads": tune.randint(3, 10),
        "intermediate_size": tune.lograndint(64, 1600, base=2),
    }
    reporter = CLIReporter(metric_columns=["loss"])
    result = tune.run(
        _train_chemberta,
        resources_per_trial={
            "cpu": os.cpu_count(),
            "gpu": 1 if torch.cuda.is_available() else 0,
        },
        config=config,
        num_samples=args.num_trials,
        progress_reporter=reporter,
    )
    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    best_trained_model = Chemberta(
        task="mlm",
        model_config_kwargs=best_trial.config,
        model_dir=args.output_dir,
    )
    best_trained_model.load_from_pretrained(model_dir=best_trial.checkpoint.dir_or_data)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    best_trained_model.save_checkpoint(
        max_checkpoints_to_keep=1, model_dir=args.output_dir
    )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--max_num_samples", type=int, default=1e4)
    argparser.add_argument("--num_shards", type=int, default=1)
    argparser.add_argument(
        "--num_epochs", type=int, default=1, help="Number of epochs per trial."
    )
    argparser.add_argument("--patience", type=int, default=5)
    argparser.add_argument("--seed", type=int, default=123)
    argparser.add_argument("--frac_valid", type=float, default=0.01)
    argparser.add_argument("--output_dir", type=str, default="./best_infograph_model")
    argparser.add_argument("--num_trials", type=int, default=10)
    args = argparser.parse_args()
    train(args)
