import argparse
from chemberta3.benchmarking.benchmark import train, evaluate

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
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
    # NOTE There might be a better argument than job
    argparser.add_argument("--job", type=str, default="train")
    argparser.add_argument("--from-hf-checkpoint", action=argparse.BooleanOptionalAction)
    argparser.add_argument("--tokenizer-path", type=str, default=None)
    args = argparser.parse_args()
    print ('model name is ', args.model_name, args.checkpoint, args.from_hf_checkpoint)
    if args.job == 'train':
        train(args)
    elif args.job == 'evaluate':
        evaluate(args)
