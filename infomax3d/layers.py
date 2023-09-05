from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F

from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims

full_atom_feature_dims = get_atom_feature_dims()
full_bond_feature_dims = get_bond_feature_dims()

SUPPORTED_ACTIVATION_MAP = {
    'ReLU', 'Sigmoid', 'Tanh', 'ELU', 'SELU', 'GLU', 'LeakyReLU', 'Softplus',
    'SiLU', 'None'
}

def get_activation(activation):
    """ returns the activation function represented by the input string """
    if activation and callable(activation):
        # activation is already a function
        return activation
    # search in SUPPORTED_ACTIVATION_MAP a torch.nn.modules.activation
    activation = [
        x for x in SUPPORTED_ACTIVATION_MAP if activation.lower() == x.lower()
    ]
    assert len(activation) == 1 and isinstance(
        activation[0], str), 'Unhandled activation function'
    activation = activation[0]
    if activation.lower() == 'none':
        return None
    return vars(torch.nn.modules.activation)[activation]()


class AtomEncoder(torch.nn.Module):

    def __init__(self, emb_dim, padding=False):
        """
        :param emb_dim: the dimension that the returned embedding will have
        :param padding: if this is true then -1 will be mapped to padding
        """
        super(AtomEncoder, self).__init__()

        self.atom_embedding_list = torch.nn.ModuleList()
        self.padding = padding

        for i, dim in enumerate(full_atom_feature_dims):
            if padding:
                emb = torch.nn.Embedding(dim + 1, emb_dim, padding_idx=0)
            else:
                emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def reset_parameters(self):
        for i, embedder in enumerate(self.atom_embedding_list):
            embedder.weight.data.uniform_(-sqrt(3), sqrt(3))

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            if self.padding:
                x_embedding += self.atom_embedding_list[i](x[:, i] + 1)
            else:
                x_embedding += self.atom_embedding_list[i](x[:, i])

        return x_embedding


class BondEncoder(torch.nn.Module):

    def __init__(self, emb_dim, padding=False):
        """
        :param emb_dim: the dimension that the returned embedding will have
        :param padding: if this is true then -1 will be mapped to padding
        """
        super(BondEncoder, self).__init__()

        self.bond_embedding_list = torch.nn.ModuleList()
        self.padding = padding

        for i, dim in enumerate(full_bond_feature_dims):
            if padding:
                emb = torch.nn.Embedding(dim + 1, emb_dim, padding_idx=0)
            else:
                emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        bond_embedding = 0
        for i in range(edge_attr.shape[1]):
            if self.padding:
                bond_embedding += self.bond_embedding_list[i](edge_attr[:, i] +
                                                              1)
            else:
                bond_embedding += self.bond_embedding_list[i](edge_attr[:, i])

        return bond_embedding


class FCLayer(nn.Module):

    def __init__(self,
                 in_dim,
                 out_dim,
                 activation='relu',
                 dropout=0.,
                 batch_norm=False,
                 batch_norm_momentum=0.1,
                 bias=True,
                 init_fn=None,
                 device='cpu'):
        super(FCLayer, self).__init__()
        self.__params = locals()
        del self.__params['__class__']
        del self.__params['self']
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.bias = bias
        self.linear = nn.Linear(in_dim, out_dim, bias=bias).to(device)
        self.dropout = None
        self.batch_norm = None
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(
                out_dim, momentum=batch_norm_momentum).to(device)
        self.activation = get_activation(activation)
        self.init_fn = nn.init.xavier_uniform_

        self.reset_parameters()

    def reset_parameters(self, init_fn=None):
        init_fn = init_fn or self.init_fn
        if init_fn is not None:
            init_fn(self.linear.weight, 1 / self.in_dim)
        if self.bias:
            self.linear.bias.data.zero_()

    def forward(self, x):
        h = self.linear(x)
        if self.activation is not None:
            h = self.activation(h)
        if self.dropout is not None:
            h = self.dropout(h)
        if self.batch_norm is not None:
            if h.shape[1] != self.out_dim:
                h = self.batch_norm(h.transpose(1, 2)).transpose(1, 2)
            else:
                h = self.batch_norm(h)
        return h


class MLP(nn.Module):
    """
        Simple multi-layer perceptron, built of a series of FCLayers
    """

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


class MLPReadout(nn.Module):

    def __init__(self, input_dim, output_dim, L=2):  # L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [
            nn.Linear(input_dim // 2**l, input_dim // 2**(l + 1), bias=True)
            for l in range(L)
        ]
        list_FC_layers.append(
            nn.Linear(input_dim // 2**L, output_dim, bias=True))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L

    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.relu(y)
        y = self.FC_layers[self.L](y)
        return y
