"""
Simple demo of training, interrupting and resuming from checkpoint
"""
import time
import torch
import os

import deepchem as dc
from deepchem.models import GCNModel
import tempfile

from ray.train import RunConfig, CheckpointConfig, ScalingConfig, Checkpoint, SyncConfig
from ray.train.torch import TorchTrainer
from ray import train


def train_func(config):
    n_epochs = 10
    # data_dir = '/Users/arun/proj/dfs/chemberta3/spot/script/data'
    data_dir = '/home/ubuntu/data'
    dataset = dc.data.DiskDataset(data_dir=data_dir)
    model = GCNModel(mode='regression',
                     n_tasks=1,
                     batch_size=16,
                     learning_rate=0.001,
                     device='cpu')

    start_epoch = 0
    checkpoint = train.get_checkpoint()
    if checkpoint:
        print('checkpoint is present')
        model._ensure_built()
        with checkpoint.as_directory() as ckpt_dir:
            ckpt_dict = torch.load(os.path.join(ckpt_dir, 'ckpt.pt'))
            model.model.load_state_dict(ckpt_dict['model'])
            model._pytorch_optimizer.load_state_dict(ckpt_dict['optimizer'])
            start_epoch = int(ckpt_dict['epoch']) + 1
            loss = ckpt_dict['loss']
    else:
        print('checkpoint is not present')

    print('start epochs is ', start_epoch)

    for epoch in range(start_epoch, n_epochs):
        print('epoch is ', epoch)
        loss = model.fit(dataset, nb_epoch=1, max_checkpoints_to_keep=1)
        time.sleep(3)
        with tempfile.TemporaryDirectory() as ckpt_dir:
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.model.state_dict(),
                    'optimizer': model._pytorch_optimizer.state_dict(),
                    'loss': loss,
                }, os.path.join(ckpt_dir, "ckpt.pt"))
            metrics_report = {'loss': loss}
            train.report(metrics=metrics_report,
                         checkpoint=Checkpoint.from_directory(ckpt_dir))

        if epoch == 1:
            raise RuntimeError("Interrupting to demonstrate checkpointing")


if __name__ == '__main__':
    ckpt_config = CheckpointConfig(checkpoint_score_attribute='loss',
                                   # num_to_keep=2,
                                   checkpoint_score_order='min')
    sync_config = SyncConfig(sync_period=150)
    train_loop_config = {}
    scaling_config = ScalingConfig(num_workers=1, use_gpu=False)
    experiment_path = 's3://chemberta3/spot/test'
    if TorchTrainer.can_restore(experiment_path):
        print('restoring from experiment')
        trainer = TorchTrainer.restore(experiment_path)
        result = trainer.fit()
    else:
        print('ckpt not avl, starting new ')
        trainer = TorchTrainer(
            train_loop_per_worker=train_func,
            train_loop_config=train_loop_config,
            scaling_config=scaling_config,
            run_config=train.RunConfig(
                storage_path='s3://chemberta3/spot/',
                name='test',
                checkpoint_config=ckpt_config,
                sync_config=sync_config,
            ),
        )
        result = trainer.fit()
