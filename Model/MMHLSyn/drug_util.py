from torch_geometric.data import InMemoryDataset, Batch
from torch_geometric import data as DATA
import torch
import numpy as np
import pandas as pd

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# device = torch.device('cpu')
class GraphDataset(InMemoryDataset):
    def __init__(self, root='.', dataset='davis', transform=None, pre_transform=None, graphs_dict=None, dttype=None):
        super(GraphDataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        self.dttype = dttype
        self.process(graphs_dict)

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        return [self.dataset + f'_data_{self.dttype}.pt']

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        pass

    #         if not os.path.exists(self.processed_dir):
    #             os.makedirs(self.processed_dir)

    def process(self, graphs_dict):
        data_list = []
        for data_mol in graphs_dict:
            # features, edge_index = data_mol[0],data_mol[1]
            # GCNData = DATA.Data(x=torch.Tensor(features), edge_index=torch.LongTensor(edge_index))
            features = torch.Tensor(data_mol[0]).to(device);
            edge_index = torch.LongTensor(data_mol[1]).to(device)
            GCNData = DATA.Data(x=features, edge_index=edge_index)
            data_list.append(GCNData)
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate(data_list):
    batchA = Batch.from_data_list([data for data in data_list])
    return batchA.to(device)


# -----molecular_graph_feature
def calculate_graph_feat(feat_mat, adj_list):
    assert feat_mat.shape[0] == len(adj_list)#检验
    adj_mat = np.zeros((len(adj_list), len(adj_list)), dtype='float32')
    for i in range(len(adj_list)):
        nodes = adj_list[i]
        for each in nodes:
            adj_mat[i, int(each)] = 1
    assert np.allclose(adj_mat, adj_mat.T)
    x, y = np.where(adj_mat == 1)#当where内只有一个参数时，那个参数表示条件，当条件成立时，where返回的是每个符合condition条件元素的行和列坐标,返回的是以两个数组的元组的形式。
    adj_index = np.array(np.vstack((x, y)))# x,y两个数组拼接起来，形成一个新的数组
    return [feat_mat, adj_index]#该函数返回一个包含特征矩阵 （） 和邻接索引 （） 的列表。feat_matadj_index



def drug_feature_extract(drug_data):
    drug_data = pd.DataFrame(drug_data).T#DataFrame.T属性用于对数据框的索引和列进行转置。该属性T与方法transpose()有一定的关系。
    drug_feat = [[] for _ in range(len(drug_data))]#得到一个二维的列表
    for i in range(len(drug_feat)):
        feat_mat, adj_list = drug_data.iloc[i]#读取第i行的值，并把药物分子的特征和邻接列表分别赋值给两个变量
        drug_feat[i] = calculate_graph_feat(feat_mat, adj_list)#构建分子特征图后，返回这个列表的值
    return drug_feat
