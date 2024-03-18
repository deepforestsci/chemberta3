import torch
import yaml
import tempfile
import os
import deepchem as dc
import ray
import numpy as np
from ray import train
from ray_ds import RayDataset
import ray.train.torch
from ray.train.torch import TorchTrainer
from ray.train import Checkpoint, CheckpointConfig, RunConfig, ScalingConfig


def train_loop_per_worker(config):
    world_rank = train.get_context().get_world_rank()
    local_rank = train.get_context().get_local_rank()
    device = ray.train.torch.get_device()
    print('world rank ', world_rank, ' local rank ', local_rank, ' device ',
          device)

    dc_model = dc.models.torch_models.Chemberta(task='mlm',
                                                learning_rate=0.0001)
    dc_model.device = device
    batch_size = 16

    dc_model.model = ray.train.torch.prepare_model(dc_model.model)
    dc_model._ensure_built()
    optimizer = dc_model._pytorch_optimizer

    train_data_shard = train.get_dataset_shard("train")
    train_dataloader = train_data_shard.iter_batches(batch_size=batch_size)

    checkpoint = train.get_checkpoint()
    if checkpoint:
        print('checkpoint is present')
        with checkpoint.as_directory() as ckpt_dir:
            ckpt_dict = torch.load(os.path.join(ckpt_dir, 'ckpt.pt'))
            # We are checkpointing only the PyTorch model and not using DeepChem checkpointing.
            dc_model.model.load_state_dict(ckpt_dict['model'])
            optimizer.load_state_dict(ckpt_dict['optimizer'])
            start_epoch = int(ckpt_dict['epoch']) + 1
            loss = ckpt_dict['loss']
    else:
        print('checkpoint is not present')
        start_epoch = 0

    for epoch in range(start_epoch, config['num_epochs']):
        for i, batch in enumerate(train_dataloader):
            inputs, labels, weights = dc_model._prepare_batch(
                ([batch['x']], None, None))
            # FIXME This works only for ModularTorchModel
            # In ModularTorchModel, loss_func performs the forward pass
            # loss = dc_model.loss_func(inputs, labels, weights)
            # FIXME The below works only for hugging-face models
            # because only they return outputs as dict
            outputs = dc_model.model(**inputs)
            loss = outputs.get("loss")
            loss.backward()
            optimizer.zero_grad()
            optimizer.step()
            loss = loss.detach().cpu().item()

            metrics = {"loss": loss, "epoch": epoch, "iteration": i}
            with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                data = {
                    'model': dc_model.model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'loss': loss
                }
                torch.save(data, "ckpt.pt")
                ray.train.report(metrics,
                                 checkpoint=Checkpoint.from_directory(
                                     temp_checkpoint_dir))

if __name__ == '__main__':
    with open('config.yaml', 'r'):
        config = yaml.load(f, Loader=yaml.SafeLoader)

    use_gpu = True

    dataset_path = train_data_dir
    train_dataset = RayDataset.read(dataset_path).dataset
    # train_dataset.materialize()
    train_loop_config = {"num_epochs": 20}
    ckpt_config = CheckpointConfig(num_to_keep=5,
                                   checkpoint_score_attribute='loss',
                                   checkpoint_score_order='min')
    run_config = RunConfig(checkpoint_config=ckpt_config,
                           name=exp_name,
                           storage_path=storage_path)

    exp_path = os.path.join(storage_path, exp_name)
    scaling_config = ScalingConfig(num_workers=num_workers, use_gpu=use_gpu)
    datasets = {"train": train_dataset}
    if not args.force_restore and TorchTrainer.can_restore(exp_path):
        # By default, experiments are restored if a checkpoint exists.
        # Otherwise, a new experiment is started unless force_restore is False.
        print("experiment can be restored")
        trainer = TorchTrainer.restore(exp_path, datasets=datasets)
        result = trainer.fit()
    else:
        print("experiment cannot be restored \n starting a new experiment")
        trainer = TorchTrainer(train_loop_per_worker=train_loop_per_worker,
                               train_loop_config=train_loop_config,
                               datasets={"train": train_dataset},
                               scaling_config=scaling_config,
                               run_config=run_config)
        result = trainer.fit()
