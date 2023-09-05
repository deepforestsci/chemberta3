import torch
import os
from typing import Dict, Union, Tuple, List
from itertools import chain
import numpy as np
from pna import PNA
from metrics import MAE, PositiveSimilarity, NegativeSimilarity, ContrastiveAccuracy, Alignment, TruePositiveRate, TrueNegativeRate
from net3d import Net3D
import rdkit_conformers as rf
from torch.utils.data import DataLoader
from losses import NTXent
import dgl


def contrastive_collate(batch: List[Tuple]):
    # optionally take targets
    graphs, graphs3d, *targets = map(list, zip(*batch))
    batched_graph = dgl.batch(graphs)
    batched_graph3d = dgl.batch(graphs3d)

    if targets:
        return [batched_graph], [batched_graph3d], torch.stack(*targets).float()
    else:
        return [batched_graph], [batched_graph3d]


def move_to_device(element, device):
    '''
    takes arbitrarily nested list and moves everything in it to device if it is a dgl graph or a torch tensor
    :param element: arbitrarily nested list
    :param device:
    :return:
    '''
    if isinstance(element, list):
        return [move_to_device(x, device) for x in element]
    else:
        return element.to(device) if isinstance(element,
                                                (torch.Tensor,
                                                 dgl.DGLGraph)) else element


class SelfSupervisedTrainer:
    """Trainer for training infomax model"""

    def __init__(self,
                 metrics,
                 num_epochs=10,
                 model_type='PNA',
                 logdir='runs',
                 experiment_name='one',
                 main_metric='NTXent',
                 main_metric_goal='min'):
        self.device = torch.device('cpu')
        self.metrics = metrics
        self.main_metric = main_metric
        self.loss_func = NTXent()
        self.model_type = 'PNA'
        self.experiment_name = 'NTXent'
        self.target_dim = 10
        self.model = PNA(hidden_dim=32,
                         target_dim=self.target_dim,
                         readout_aggregators=['mean'],
                         scalers=['identity'],
                         aggregators=['mean'])
        self.model3d = Net3D(node_dim=9,
                             edge_dim=3,
                             hidden_dim=32,
                             target_dim=self.target_dim,
                             readout_aggregators=['mean'])
        self.num_epochs = num_epochs
        self.log_iterations = 2
        self.initialize_optimizer_and_scheduler()
        self.patience = 35
        self.minimum_epochs = 0
        self.eval_per_epochs = 0
        self.config_name = experiment_name
        self.linear_probing_samples = 500
        self.main_metric_goal = main_metric_goal
        self.logdir = logdir
        self.start_epoch = 1
        self.optim_steps = 0
        self.best_val_score = -np.inf if self.main_metric_goal == 'max' else np.inf  # running score to decide whether or not a new model should be saved
        print('Log directory: ', self.logdir)

    def forward_pass(self, batch):
        info2d, info3d, *snorm_n = tuple(batch)
        view2d = self.model(
            *info2d, *snorm_n)  # foward the rest of the batch to the model
        view3d = self.model3d(*info3d)
        loss = self.loss_func(view2d,
                              view3d,
                              nodes_per_graph=info2d[0].batch_num_nodes()
                              if isinstance(info2d[0], dgl.DGLGraph) else None)
        return loss, view2d, view3d

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        epochs_no_improve = 0  # counts every epoch that the validation accuracy did not improve for early stopping
        for epoch in range(self.start_epoch, self.num_epochs +
                           1):  # loop over the dataset multiple times
            self.model.train()
            self.predict(train_loader, epoch, optim=self.optim)

            self.model.eval()
            with torch.no_grad():
                metrics = self.predict(val_loader, epoch)
                val_score = metrics[self.main_metric]

                self.lr_scheduler.step()

                if self.eval_per_epochs > 0 and epoch % self.eval_per_epochs == 0:
                    self.run_per_epoch_evaluations(val_loader)

                val_loss = metrics[type(self.loss_func).__name__]
                print('[Epoch %d] %s: %.6f val loss: %.6f' %
                      (epoch, self.main_metric, val_score, val_loss))

                # save the model with the best main_metric depending on wether we want to maximize or minimize the main metric
                if val_score >= self.best_val_score and self.main_metric_goal == 'max' or val_score <= self.best_val_score and self.main_metric_goal == 'min':
                    epochs_no_improve = 0
                    self.best_val_score = val_score
                    self.save_checkpoint(epoch,
                                         checkpoint_name='best_checkpoint.pt')
                else:
                    epochs_no_improve += 1
                self.save_checkpoint(epoch,
                                     checkpoint_name='last_checkpoint.pt')

                if epochs_no_improve >= self.patience and epoch >= self.minimum_epochs:  # stopping criterion
                    print(
                        f'Early stopping criterion based on -{self.main_metric}- that should be {self.main_metric_goal} reached after {epoch} epochs. Best model checkpoint was in epoch {epoch - epochs_no_improve}.'
                    )
                    break

        # evaluate on best checkpoint
        checkpoint = torch.load(os.path.join(self.logdir, 'best_checkpoint.pt'),
                                map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return self.evaluation(val_loader, data_split='val_best_checkpoint')

    def process_batch(self, batch, optim):
        loss, predictions, targets = self.forward_pass(batch)
        if optim != None:  # run backpropagation if an optimizer is provided
            loss.backward()
            self.optim.step()
            self.lr_scheduler.step()
            self.optim.zero_grad()
            self.optim_steps += 1
        return loss, predictions.detach(), targets.detach()

    def evaluation(self, data_loader: DataLoader, data_split: str = ''):
        self.model.eval()
        metrics = self.predict(data_loader, epoch=2)

        with open(
                os.path.join(self.logdir, 'evaluation_' + data_split + '.txt'),
                'w') as file:
            print('Statistics on ', data_split)
            for key, value in metrics.items():
                file.write(f'{key}: {value}\n')
                print(f'{key}: {value}')
        return metrics

    def initialize_optimizer_and_scheduler(self):
        normal_params = [
            v for k, v in chain(self.model.named_parameters(),
                                self.model3d.named_parameters())
            if not 'batch_norm' in k
        ]
        batch_norm_params = [
            v for k, v in chain(self.model.named_parameters(),
                                self.model3d.named_parameters())
            if 'batch_norm' in k
        ]

        self.optim = torch.optim.Adam([{
            'params': batch_norm_params,
            'weight_decay': 0
        }, {
            'params': normal_params
        }],
                                      lr=8e-5)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optim,
                                                                   gamma=0.99)

    def predict(
        self,
        data_loader: DataLoader,
        epoch: int,
        optim: torch.optim.Optimizer = None,
        return_predictions: bool = False
    ) -> Union[Dict, Tuple[float, Union[torch.Tensor, None], Union[torch.Tensor,
                                                                   None]]]:
        total_metrics = {
            k: 0 for k in list(self.metrics.keys()) + [
                type(self.loss_func).__name__, 'mean_pred', 'std_pred',
                'mean_targets', 'std_targets'
            ]
        }
        epoch_targets = torch.tensor([]).to(self.device)
        epoch_predictions = torch.tensor([]).to(self.device)
        epoch_loss = 0
        for i, batch in enumerate(data_loader):
            batch = move_to_device(list(batch), self.device)
            loss, predictions, targets = self.process_batch(batch, optim)
            with torch.no_grad():
                if self.optim_steps % self.log_iterations == 0 and optim != None:
                    metrics = self.evaluate_metrics(predictions, targets)
                    metrics[type(self.loss_func).__name__] = loss.item()
                    print(
                        '[Epoch %d; Iter %5d/%5d] %s: loss: %.7f' %
                        (epoch, i + 1, len(data_loader), 'train', loss.item()))
                # during validation or testing when we want to average metrics over all the data in that dataloader
                if optim == None:
                    epoch_loss += loss.item()
                    epoch_targets = torch.cat((targets, epoch_targets), 0)
                    epoch_predictions = torch.cat(
                        (predictions, epoch_predictions), 0)

        if optim == None:
            total_metrics = self.evaluate_metrics(epoch_predictions,
                                                  epoch_targets,
                                                  val=True)
            total_metrics[type(
                self.loss_func).__name__] = epoch_loss / len(data_loader)
            return total_metrics

    def evaluate_metrics(self,
                         predictions,
                         targets,
                         batch=None,
                         val=False) -> Dict[str, float]:
        metrics = {}
        metrics[f'mean_pred'] = torch.mean(predictions).item()
        metrics[f'std_pred'] = torch.std(predictions).item()
        metrics[f'mean_targets'] = torch.mean(targets).item()
        metrics[f'std_targets'] = torch.std(targets).item()
        for key, metric in self.metrics.items():
            if not hasattr(metric, 'val_only') or val:
                metrics[key] = metric(predictions, targets).item()
        return metrics

    def run_per_epoch_evaluations(self, data_loader):
        print('fitting linear probe')
        representations = []
        targets = []
        for batch in data_loader:
            batch = [element.to(self.device) for element in batch]
            loss, view2d, view3d = self.process_batch(batch, optim=None)
            representations.append(view2d)
            targets.append(batch[-1])
            if len(representations) * len(
                    view2d) >= self.linear_probing_samples:
                break
        representations = torch.cat(representations, dim=0)
        targets = torch.cat(targets, dim=0)
        if len(representations) >= representations.shape[-1]:
            X, _ = torch.lstsq(targets, representations)
            X, _ = torch.lstsq(targets, representations)
            sol = X[:representations.shape[-1]]
            pred = representations @ sol
            mean_absolute_error = (pred - targets).abs().mean()
        else:
            raise ValueError(
                f'We have less linear_probing_samples {len(representations)} than the metric dimension {representations.shape[-1]}. Linear probing cannot be used.'
            )

        print('finish fitting linear probe')

    def save_checkpoint(self, epoch: int, checkpoint_name: str):
        torch.save(
            {
                'epoch':
                    epoch,
                'best_val_score':
                    self.best_val_score,
                'optim_steps':
                    self.optim_steps,
                'model_state_dict':
                    self.model.state_dict(),
                'model3d_state_dict':
                    self.model3d.state_dict(),
                'optimizer_state_dict':
                    self.optim.state_dict(),
                'scheduler_state_dict':
                    self.lr_scheduler.state_dict()
            }, os.path.join(self.logdir, checkpoint_name))


if __name__ == '__main__':
    valid = rf.RDKitConformers(filepath='data/zinc5k_valid.csv',
                               feature_field='X',
                               device=torch.device('cpu'))
    train = rf.RDKitConformers(filepath='data/zinc5k_train.csv',
                               feature_field='X',
                               device=torch.device('cpu'))
    train_loader = DataLoader(valid,
                              batch_size=64,
                              shuffle=True,
                              collate_fn=contrastive_collate)
    val_loader = DataLoader(valid,
                            batch_size=64,
                            shuffle=True,
                            collate_fn=contrastive_collate)
    metrics = {
        'mae': MAE(),
        'positive_similarity': PositiveSimilarity(),
        'negative_similarity': NegativeSimilarity(),
        'contrastive_accuracy': ContrastiveAccuracy(threshold=0.5009),
        'true_negative_rate': TrueNegativeRate(threshold=0.5009),
        'true_positive_rate': TruePositiveRate(threshold=0.5009),
        'alignment': Alignment(alpha=2),
    }
    trainer = SelfSupervisedTrainer(metrics=metrics)

    trainer.train(train_loader=train_loader, val_loader=val_loader)
