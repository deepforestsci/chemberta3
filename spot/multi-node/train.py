import torch
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

if __name__ == '__main__':
    use_gpu = False

    dataset_path = 's3://chemberta3/ray_test/featurized_data/MolGraphConv/zinc250k/'
    train_dataset = RayDataset.read(dataset_path).dataset

    def train_loop_per_worker(config):
        dc_model = dc.models.torch_models.InfoGraphModel(num_features=30,
                                            embedding_dim=11,
                                            num_gc_layers=3,
                                            task='pretraining',
                                            learning_rate=0.0001)
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
            start_epoch = config["num_epochs"]

        for epoch in range(start_epoch):
            losses = []
            for batch in train_dataloader:
                inputs, labels, weights = dc_model._prepare_batch(([batch['x']], None,
                                                          None))
                loss = dc_model.loss_func(inputs, labels, weights)
                loss.backward()
                optimizer.zero_grad()
                optimizer.step()
                losses.append(loss.detach().cpu().item())

            metrics = {"loss": np.mean(losses), "epoch": epoch}

            with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                data = {
                    'model': dc_model.model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'loss': loss
                }
                torch.save(data, "ckpt.pt")
                ray.train.report(
                    metrics,
                    checkpoint=Checkpoint.from_directory(temp_checkpoint_dir))
                if ray.train.get_context().get_world_rank() == 0:
                    print(metrics)

    train_loop_config = {"num_epochs": 20}
    run_config = RunConfig(checkpoint_config=CheckpointConfig(num_to_keep=1))
    trainer = TorchTrainer(train_loop_per_worker=train_loop_per_worker,
                           train_loop_config=train_loop_config,
                           datasets={"train": train_dataset},
                           scaling_config=ScalingConfig(num_workers=2,
                                                        use_gpu=use_gpu),
                           run_config=run_config)
    result = trainer.fit()
