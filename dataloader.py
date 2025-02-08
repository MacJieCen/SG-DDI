import pandas as pd
import torch.utils.data as data
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data
from pandas import DataFrame
from sklearn.decomposition import PCA

def feature_vector(feature_name, df, vector_size):
    # Jaccard Similarity
    def Jaccard(matrix):
        matrix = np.mat(matrix)
        numerator = matrix * matrix.T
        denominator = np.ones(np.shape(matrix)) * matrix.T + matrix * np.ones(np.shape(matrix.T)) - matrix * matrix.T
        denominator = np.array(denominator)
        arrayC = np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=np.float64), where=denominator != 0)
        return arrayC

    all_feature = []
    drug_list = np.array(df[feature_name]).tolist()
    for i in drug_list:
        if i == 'not_have':
            continue
        for each_feature in i.split('|'):
            if each_feature not in all_feature:
                all_feature.append(each_feature)
    feature_matrix = np.zeros((len(drug_list), len(all_feature)), dtype=float)
    df_feature = DataFrame(feature_matrix, columns=all_feature)
    for i in range(len(drug_list)):
        for each_feature in df[feature_name].iloc[i].split('|'):
            if each_feature == 'not_have':
                continue
            df_feature[each_feature].iloc[i] = 1
    sim_matrix = Jaccard(np.array(df_feature))

    pca = PCA(n_components=vector_size)
    sim_matrix = np.array(sim_matrix)
    # sim_matrix[~ np.isfinite(sim_matrix)] = 0
    pca.fit(sim_matrix)
    sim_matrix = pca.transform(sim_matrix)
    return sim_matrix

class PairData(Data):
    def __init__(self, edge_index_s, x_s, edge_index_t, x_t):
        super(PairData, self).__init__()
        self.edge_index_s = edge_index_s
        self.x_s = x_s
        self.edge_index_t = edge_index_t
        self.x_t = x_t

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_s':
            return self.x_s.size(0)
        if key == 'edge_index_t':
            return self.x_t.size(0)
        return super().__inc__(key, value, *args, **kwargs)


class DTIDataset(data.Dataset):
    def __init__(self, list_IDs, df, description_text, smiles_graph, edge_graph, max_drug_nodes = 160, drug_num = 1706):
        self.list_IDs = list_IDs
        self.df = df
        self.description = description_text
        self.smiles_graph = smiles_graph
        self.edge_graph = edge_graph
        self.max_drug_nodes = max_drug_nodes
        self.drug_num = drug_num
        self.df_bio = pd.read_csv('./datasets/D1/Bio_info.csv', encoding='ISO-8859-1')
        self.target_matrix = feature_vector('Targets_DrugBank', self.df_bio, self.drug_num)
        self.smile_matrix = feature_vector('Smile', self.df_bio, self.drug_num)
        self.drug_id_dataset = self.df_bio['DrugID'].to_numpy()

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        index = self.list_IDs[index]

        y = self.df.iloc[index]["Y"]
        label = self.df.iloc[index]["y"]
        y = torch.Tensor([y])
        label = torch.Tensor([label])

        id_1 = self.df.iloc[index]["ID1"]
        id_2 = self.df.iloc[index]["ID2"]

        v_d_id = next((index for index, value in enumerate(self.drug_id_dataset) if value == id_1), None)
        v_p_id = next((index for index, value in enumerate(self.drug_id_dataset) if value == id_2), None)

        v_d = self.smiles_graph[v_d_id]
        v_d_edge_index = self.edge_graph[v_d_id]
        v_d_target = self.target_matrix[v_d_id]
        v_d_smile = self.smile_matrix[v_d_id]
        v_d_description = self.description[v_d_id]

        v_p = self.smiles_graph[v_p_id]
        v_p_edge_index = self.edge_graph[v_p_id]
        v_p_target = self.target_matrix[v_p_id]
        v_p_smile = self.smile_matrix[v_p_id]
        v_p_description = self.description[v_p_id]

        v_d_target = torch.tensor(v_d_target)
        v_d_smile = torch.tensor(v_d_smile)
        v_p_target = torch.tensor(v_p_target)
        v_p_smile = torch.tensor(v_p_smile)

        h_data = Data(edge_index=v_d_edge_index, x=v_d)
        t_data = Data(edge_index=v_p_edge_index, x=v_p)
        graph = (h_data, t_data)
        return graph, y, label, v_d_target, v_d_smile, v_d_description, v_p_target, v_p_smile, v_p_description



# 自定义转换类
class SMILESToGraph(object):
    def __call__(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        G = Data()
        G.edge_index = torch.tensor(AllChem.GetAdjacencyMatrix(mol), dtype=torch.long).t().contiguous()
        G.x = self.featurize_atoms(mol)
        return G

    def featurize_atoms(self, mol):
        feats = []
        for atom in mol.GetAtoms():
            feat = [atom.GetAtomicNum()]
            feats.append(feat)
        return torch.tensor(feats, dtype=torch.float)


class PadNodes(object):
    def __init__(self, max_nodes):
        self.max_nodes = max_nodes

    def __call__(self, data):
        if data is None:
            return None
        if data.num_nodes < self.max_nodes:
            padding = torch.zeros((self.max_nodes - data.num_nodes, data.x.size(1)))
            data.x = torch.cat([data.x, padding], dim=0)
        return data


class MultiDataLoader(object):
    def __init__(self, dataloaders, n_batches):
        if n_batches <= 0:
            raise ValueError("n_batches should be > 0")
        self._dataloaders = dataloaders
        self._n_batches = np.maximum(1, n_batches)
        self._init_iterators()

    def _init_iterators(self):
        self._iterators = [iter(dl) for dl in self._dataloaders]

    def _get_nexts(self):
        def _get_next_dl_batch(di, dl):
            try:
                batch = next(dl)
            except StopIteration:
                new_dl = iter(self._dataloaders[di])
                self._iterators[di] = new_dl
                batch = next(new_dl)
            return batch

        return [_get_next_dl_batch(di, dl) for di, dl in enumerate(self._iterators)]

    def __iter__(self):
        for _ in range(self._n_batches):
            yield self._get_nexts()
        self._init_iterators()

    def __len__(self):
        return self._n_batches
