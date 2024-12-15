
import torch
import torch.nn as nn
from torch_geometric.nn import  GATv2Conv, Linear, to_hetero, TransformerConv
from torch_geometric.data import HeteroData
from torch.distributions import Categorical

class GAT(torch.nn.Module):
    def __init__(self, hidden_channels,  num_layers = 2, heads = 3):
        super().__init__()
        self.lin1 = Linear(-1, 8)
        self.s = torch.nn.Softmax(dim=0)
        self.activation = nn.ELU()
        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = TransformerConv(-1, hidden_channels, edge_dim=4, heads = heads)
            self.convs.append(conv)

    def forward(self, x, edge_index, edge_attr_dict):
        x = self.lin1(x)
        x = self.activation(x)
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr_dict)
        x = self.activation(x)
        return x

class Model(torch.nn.Module):
    def __init__(self, hidden_channels, metadata, num_layers = 2, heads = 3):
        super().__init__()
        self.gnn = GAT(hidden_channels, num_layers=num_layers, heads=heads)
        self.gnn = to_hetero(self.gnn, metadata=metadata, aggr='mean')
        self.lin3 = Linear(-1, 1)
        self.lin4 = Linear(-1, hidden_channels)
        self.lin5 = Linear(-1, hidden_channels)
        self.lin6 = Linear(-1, 1)
        self.activation = nn.ELU()

    def forward(self, data: HeteroData):
        res = self.gnn(data.x_dict, data.edge_index_dict, data.edge_attr_dict)
        x_src, x_dst = res['machine'][data.edge_index_dict[('machine','exec','job')][0]], res['job'][data.edge_index_dict['machine','exec','job'][1]]
        edge_feat = torch.cat([x_src,  data.edge_attr_dict[('machine','exec','job')], x_dst], dim=-1)
        res = self.lin4(edge_feat)
        res = self.activation(res)
        res = self.lin3(res)
        return res