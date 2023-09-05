import os

import torch
import dgl
from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector
from rdkit import Chem
from rdkit.Chem import AllChem
from torch.utils.data import Dataset
import pandas as pd
from tqdm import tqdm


class RDKitConformers(Dataset):

    def __init__(self, filepath: str, feature_field: str, device, **kwargs):
        self.feature_field = feature_field
        self.data_directory = os.path.dirname(filepath)
        self.raw_file = os.path.basename(filepath)
        self.processed_file = self.raw_file.split('.')[0] + '.pt'

        self.atom_types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
        self.symbols = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9}
        self.device = device
        self.return_types: list = ['dgl_graph', 'complete_graph3d']

        # load the data and get normalization values
        if not os.path.exists(
                os.path.join(self.data_directory, 'processed_rdkit_conformers',
                             self.processed_file)):
            self.process()
        data_dict = torch.load(
            os.path.join(self.data_directory, 'processed_rdkit_conformers',
                         self.processed_file))

        self.features_tensor = data_dict['atom_features']
        self.e_features_tensor = data_dict['edge_features']
        self.coordinates = data_dict['coordinates']
        self.edge_indices = data_dict['edge_indices']

        self.meta_dict = {
            k: data_dict[k]
            for k in ('mol_id', 'edge_slices', 'atom_slices', 'n_atoms')
        }

        self.dgl_graphs = {}
        self.pairwise = {}  # for memoization
        self.complete_graphs = {}
        self.pairwise_distances = {}
        self.avg_degree = data_dict['avg_degree']

    def __len__(self):
        return len(self.meta_dict['mol_id'])

    def __getitem__(self, idx):
        """

        Parameters
        ----------
        idx: integer between 0 and len(self) - 1

        Returns
        -------
        tuple of all data specified via the return_types parameter of the constructor
        """
        data = []
        e_start = self.meta_dict['edge_slices'][idx].item()
        e_end = self.meta_dict['edge_slices'][idx + 1].item()
        start = self.meta_dict['atom_slices'][idx].item()
        n_atoms = self.meta_dict['n_atoms'][idx].item()

        for return_type in self.return_types:
            data.append(
                self.data_by_type(idx, return_type, e_start, e_end, start,
                                  n_atoms))
        return tuple(data)

    def get_pairwise(self, n_atoms):
        if n_atoms in self.pairwise:
            src, dst = self.pairwise[n_atoms]
            return src.to(self.device), dst.to(self.device)
        else:
            arange = torch.arange(n_atoms, device=self.device)
            src = torch.repeat_interleave(arange, n_atoms - 1)
            dst = torch.cat([
                torch.cat([arange[:idx], arange[idx + 1:]])
                for idx in range(n_atoms)
            ])  # no self loops
            self.pairwise[n_atoms] = (src.to('cpu'), dst.to('cpu'))
            return src, dst

    def get_graph(self, idx, e_start, e_end, n_atoms, start):
        if idx in self.dgl_graphs:
            return self.dgl_graphs[idx].to(self.device)
        else:
            edge_indices = self.edge_indices[:, e_start:e_end]
            g = dgl.graph((edge_indices[0], edge_indices[1]),
                          num_nodes=n_atoms,
                          device=self.device)
            g.ndata['feat'] = self.features_tensor[start:start + n_atoms].to(
                self.device)
            g.ndata['x'] = self.coordinates[start:start + n_atoms].to(
                self.device)
            g.edata['feat'] = self.e_features_tensor[e_start:e_end].to(
                self.device)
            self.dgl_graphs[idx] = g.to('cpu')
            return g

    def get_complete_graph(self, idx, n_atoms, start):
        if idx in self.complete_graphs:
            return self.complete_graphs[idx].to(self.device)
        else:
            src, dst = self.get_pairwise(n_atoms)
            g = dgl.graph((src, dst), device=self.device)
            g.ndata['feat'] = self.features_tensor[start:start + n_atoms].to(
                self.device)
            g.ndata['x'] = self.coordinates[start:start + n_atoms].to(
                self.device)
            g.edata['d'] = torch.norm(g.ndata['x'][g.edges()[0]] -
                                      g.ndata['x'][g.edges()[1]],
                                      p=2,
                                      dim=-1).unsqueeze(-1).detach()
            self.complete_graphs[idx] = g.to('cpu')
            return g

    def data_by_type(self, idx, return_type, e_start, e_end, start, n_atoms):
        if return_type == 'dgl_graph':
            return self.get_graph(idx, e_start, e_end, n_atoms, start)
        elif return_type == 'complete_graph':  # complete graph without self loops
            g = self.get_complete_graph(idx, n_atoms, start)

            # set edge features with padding for virtual edges
            bond_features = self.e_features_tensor[e_start:e_end].to(
                self.device)
            # TODO: replace with -1 padding
            e_features = self.bond_padding_indices.expand(n_atoms * n_atoms, -1)
            edge_indices = self.edge_indices[:, e_start:e_end].to(self.device)
            bond_indices = edge_indices[0] * n_atoms + edge_indices[1]
            # overwrite the bond features
            e_features = e_features.scatter(dim=0,
                                            index=bond_indices[:, None].expand(
                                                -1, bond_features.shape[1]),
                                            src=bond_features)
            src, dst = self.get_pairwise(n_atoms)
            g.edata['feat'] = e_features[src * n_atoms + dst]
            return g
        elif return_type == 'complete_graph3d':
            g = self.get_complete_graph(idx, n_atoms, start)
            return g

    def process(self):
        print('processing data from ({}) and saving it to ({})'.format(
            self.data_directory,
            os.path.join(self.data_directory, 'processed_rdkit_conformers')))

        mol_id = 0
        molecules_df = pd.read_csv(
            os.path.join(self.data_directory, self.raw_file))
        n_molecules = molecules_df.shape[0]

        atom_slices = [0]
        edge_slices = [0]
        mol_ids = []
        n_atoms_list = []
        all_atom_features = []
        all_edge_features = []
        coordinates = []
        edge_indices = []  # edges of each molecule in coo format
        total_atoms = 0
        total_edges = 0
        avg_degree = 0  # average degree in the dataset

        for mol_id, smiles in tqdm(enumerate(molecules_df[self.feature_field])):
            mol = Chem.MolFromSmiles(smiles)
            mol = Chem.AddHs(mol)
            # FIXME Should n_atoms include Hs or not include Hs?
            n_atoms = mol.GetNumAtoms()

            try:
                ps = AllChem.ETKDGv2()
                ps.useRandomCoords = True
                AllChem.EmbedMolecule(mol, ps)
                AllChem.MMFFOptimizeMolecule(mol, confId=0)
                conf = mol.GetConformer()
                coordinates.append(
                    torch.tensor(conf.GetPositions(), dtype=torch.float))
            except:
                print('error processing %d %s' % (mol_id, smiles))
                continue

            atom_features_list = []
            for atom in mol.GetAtoms():
                atom_features_list.append(atom_to_feature_vector(atom))
            all_atom_features.append(
                torch.tensor(atom_features_list, dtype=torch.long))

            edges_list = []
            edge_features_list = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                edge_feature = bond_to_feature_vector(bond)

                # add edges in both directions
                edges_list.append((i, j))
                edge_features_list.append(edge_feature)
                edges_list.append((j, i))
                edge_features_list.append(edge_feature)
            # Graph connectivity in COO format with shape [2, num_edges]
            edge_index = torch.tensor(edges_list, dtype=torch.long).T
            edge_features = torch.tensor(edge_features_list, dtype=torch.long)

            avg_degree += (len(edges_list) / 2) / n_atoms

            edge_indices.append(edge_index)
            all_edge_features.append(edge_features)

            total_edges += len(edges_list)
            total_atoms += n_atoms
            edge_slices.append(total_edges)
            atom_slices.append(total_atoms)
            mol_ids.append(i)
            n_atoms_list.append(n_atoms)

        data_dict = {
            'mol_id': torch.tensor(mol_ids, dtype=torch.long),
            'n_atoms': torch.tensor(n_atoms_list, dtype=torch.long),
            'atom_slices': torch.tensor(atom_slices, dtype=torch.long),
            'edge_slices': torch.tensor(edge_slices, dtype=torch.long),
            'edge_indices': torch.cat(edge_indices, dim=1),
            'atom_features': torch.cat(all_atom_features, dim=0),
            'edge_features': torch.cat(all_edge_features, dim=0),
            'coordinates': torch.cat(coordinates, dim=0),
            'avg_degree': avg_degree / len(mol_ids) 
        }

        if not os.path.exists(
                os.path.join(self.data_directory,
                             'processed_rdkit_conformers')):
            os.mkdir(
                os.path.join(self.data_directory, 'processed_rdkit_conformers'))
        torch.save(
            data_dict,
            os.path.join(self.data_directory, 'processed_rdkit_conformers',
                         self.processed_file))

if __name__ == '__main__':
    dataset = RDKitConformers(filepath='data/zinc16.csv', feature_field='X', device=torch.device('cpu'))
    test_dataset = RDKitConformers(filepath='data/zinc5k_test.csv', feature_field='X', device=torch.device('cpu'))
    valid_dataset = RDKitConformers(filepath='data/zinc5k_valid.csv', feature_field='X', device=torch.device('cpu'))
    train_dataset = RDKitConformers(filepath='data/zinc5k_train.csv', feature_field='X', device=torch.device('cpu'))
