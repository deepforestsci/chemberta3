import torch
import yaml
import tempfile
import os
import deepchem as dc
import ray
import numpy as np
from ray import train
from data_utils import RayDataset
import ray.train.torch
from ray.train.torch import TorchTrainer
from ray.train import Checkpoint, CheckpointConfig, RunConfig, ScalingConfig

from transformers.models.roberta.tokenization_roberta_fast import RobertaTokenizerFast
from transformers.data.data_collator import DataCollatorForLanguageModeling


def train_loop_per_worker(config):
    device = ray.train.torch.get_device()
    dc_model = dc.models.torch_models.Chemberta(task='mlm',
                                                learning_rate=0.0001)
    batch_size = 16

    model = ray.train.torch.prepare_model(dc_model.model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train_data_shard = train.get_dataset_shard("train")
    train_dataloader = train_data_shard.iter_batches(batch_size=batch_size)

    # Chemberta-MLM specific lines
    tokenizer_path = 'seyonec/PubChem10M_SMILES_BPE_60k'
    tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_path)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer)

    checkpoint = train.get_checkpoint()
    if checkpoint:
        print('checkpoint is present')
        with checkpoint.as_directory() as ckpt_dir:
            ckpt_dict = torch.load(os.path.join(ckpt_dir, 'ckpt.pt'),
                                   weights_only=True)
            # We are checkpointing only the PyTorch model and not using DeepChem checkpointing.
            model.load_state_dict(ckpt_dict['model'])
            optimizer.load_state_dict(ckpt_dict['optimizer'])
            start_epoch = int(ckpt_dict['epoch']) + 1
            loss = ckpt_dict['loss']
    else:
        print('checkpoint is not present')
        start_epoch = 0

    # TODO Compile the model here before training for speedup
    for epoch in range(start_epoch, config['num_epochs']):
        for i, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            optimizer.zero_grad()
            inputs = batch['smiles']
            tokens = tokenizer(inputs.tolist(),
                               padding=True,
                               return_tensors="pt")
            inputs, labels = data_collator.torch_mask_tokens(
                tokens['input_ids'])
            inputs = {
                'input_ids': inputs.to(device),
                'labels': labels.to(device),
                'attention_mask': tokens['attention_mask'].to(device),
            }
            outputs = model(**inputs)
            loss = outputs.get("loss")
            loss.backward()
            optimizer.step()
            loss = loss.detach().cpu().item()

            metrics = {"loss": loss, "epoch": epoch, "iteration": i}
            with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                data = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'loss': loss
                }
                torch.save(data, "ckpt.pt")
                ray.train.report(
                    metrics,
                    checkpoint=Checkpoint.from_directory(temp_checkpoint_dir))


if __name__ == '__main__':
    with open('config.yaml', 'r'):
        config = yaml.load(f, Loader=yaml.FullLoader)

    model_parameters = config['model_parameters']
    model_name = config['model_name']
    if config['is_csv']:
        train_dataset = ray.data.read_csv(config['dataset_path'])

    # dataset_path = train_data_dir
    # train_dataset = RayDataset.read(dataset_path).dataset
    train_loop_config = {"num_epochs": config['num_epochs']}
    storage_path, exp_name = config['storage_path'], config['exp_name']
    ckpt_config = CheckpointConfig(num_to_keep=5,
                                   checkpoint_score_attribute='loss',
                                   checkpoint_score_order='min')
    run_config = RunConfig(checkpoint_config=ckpt_config,
                           name=exp_name,
                           storage_path=storage_path)
    exp_path = os.path.join(storage_path, exp_name)
    scaling_config = ScalingConfig(num_workers=config['num_workers'],
                                   use_gpu=True)

    datasets = {"train": train_dataset}
    if TorchTrainer.can_restore(exp_path):
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
