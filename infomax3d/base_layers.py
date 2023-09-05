import torch
import torch.nn as nn


class MLP(nn.Module):

    def __init__(self,
                 in_dim,
                 out_dim,
                 layers,
                 hidden_size=None,
                 mid_activation='relu',
                 last_activation='none',
                 dropout=0.,
                 mid_batch_norm=False,
                 last_batch_norm=False,
                 batch_norm_momentum=0.1,
                 device='cpu'):
        super(MLP, self).__init__()

        self.in_dim = in_dim
        self.hidden_size = hidden_size
        self.out_dim = out_dim

        self.fully_connected = nn.ModuleList()
        if layers <= 1:
            self.fully_connected.append(
                FCLayer(in_dim,
                        out_dim,
                        activation=last_activation,
                        batch_norm=last_batch_norm,
                        device=device,
                        dropout=dropout,
                        batch_norm_momentum=batch_norm_momentum))
        else:
            self.fully_connected.append(
                FCLayer(in_dim,
                        hidden_size,
                        activation=mid_activation,
                        batch_norm=mid_batch_norm,
                        device=device,
                        dropout=dropout,
                        batch_norm_momentum=batch_norm_momentum))
            for _ in range(layers - 2):
                self.fully_connected.append(
                    FCLayer(hidden_size,
                            hidden_size,
                            activation=mid_activation,
                            batch_norm=mid_batch_norm,
                            device=device,
                            dropout=dropout,
                            batch_norm_momentum=batch_norm_momentum))
            self.fully_connected.append(
                FCLayer(hidden_size,
                        out_dim,
                        activation=last_activation,
                        batch_norm=last_batch_norm,
                        device=device,
                        dropout=dropout,
                        batch_norm_momentum=batch_norm_momentum))

    def forward(self, x):
        for fc in self.fully_connected:
            x = fc(x)
        return x
