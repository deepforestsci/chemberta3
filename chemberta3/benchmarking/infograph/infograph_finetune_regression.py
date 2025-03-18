import os
import torch
import argparse
import pandas as pd
import deepchem as dc
from deepchem.models.torch_models import InfoGraphModel


def infograph_finetune(dataset: str, epochs: int, pretrained_checkpoint_path: str, output_filename: str, n_tasks: int):

    finetune_train_data = dc.data.DiskDataset('')
    finetune_valid_data = dc.data.DiskDataset('')
    finetune_test_data = dc.data.DiskDataset('')

    num_feat, edge_dim = 30, 11  # num feat and edge dim by molgraph conv featurizer
    pretrain_model = InfoGraphModel(num_feat, edge_dim, num_gc_layers=1, task='pretraining')
    pretrain_model.restore(checkpoint=pretrained_checkpoint_path)

    finetune_model = InfoGraphModel(num_feat, edge_dim, num_gc_layers=1, task='regression', n_tasks=n_tasks)
    finetune_model.load_from_pretrained(pretrain_model, components=['encoder'])

    # Model training
    print("Training model")
    nb_epoch = epochs
    metric = dc.metrics.Metric(dc.metrics.rms_score)
    results = []
    for epoch in range(0, nb_epoch + 1):
        loss = finetune_model.fit(finetune_train_data, nb_epoch=1, restore=epoch>1)
        train_scores = finetune_model.evaluate(finetune_train_data, metrics=[metric])
        valid_scores = finetune_model.evaluate(finetune_valid_data, metrics=[metric])
        test_scores = finetune_model.evaluate(finetune_test_data, metrics=[metric])
        scores = [loss, train_scores, valid_scores, test_scores]
        # Store the results in a list
        results.append([epoch, loss, train_scores['rms_score'], valid_scores['rms_score'], test_scores['rms_score']])

    # Convert results to a DataFrame
    df = pd.DataFrame(results, columns=['Epoch', 'Loss', 'Train RMS Score', 'Valid RMS Score', 'Test RMS Score'])

    # Save to CSV
    output_file_path = os.getcwd() + output_filename
    df.to_csv(output_file_path, index=False)
    print(f"Results saved to {output_file_path}")

    test_rms_score = df.loc[df['Valid Score'] == min(df['Valid Score']), 'Test Score'].values[0]
    print("min_test_rms_score: ", test_rms_score)
    print("min_valid_rms_score: ", min(df['Valid Score']))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset',
                           type=str,
                           help='dataset name',
                           default=None)
    argparser.add_argument('--n_tasks',
                           type=int,
                           help='number of tasks',
                           default=None)
    argparser.add_argument('--epochs',
                           type=int,
                           help='epochs',
                           default=None)
    argparser.add_argument('--pretrained_checkpoint_path',
                           type=str,
                           help='pretrained checkpoint path',
                           default=None)
    argparser.add_argument('--output_filename',
                           type=str,
                           help='output result filename',
                           default=None)
    args = argparser.parse_args()
    infograph_finetune(dataset=args.dataset, 
                        epochs=args.epochs, 
                        pretrained_checkpoint_path=args.pretrained_checkpoint_path, 
                        output_filename=args.output_filename,
                        n_tasks=args.n_tasks)
