import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric import nn as gnn
from torch_geometric import utils
from torch_geometric import datasets
from torch_geometric import transforms as T


class GCNConv(gnn.MessagePassing):
    def __init__(self, in_dim, out_dim, aggr='add'):
        super().__init__(aggr=aggr)
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index):
        # x: num_nodes x in_dim
        # edge: 2 x num_edges

        # Step 1: Add self-loops to the adjacency matrix
        edge_index, _ = utils.add_self_loops(edge_index, num_nodes=x.shape[0])
        # Step 2: Linear transform node feature
        x = self.lin(x)
        # Step 3: Normalization
        row, col = edge_index
        deg = utils.degree(col, x.shape[0], dtype=x.dtype)  # in-degree
        deg_inv_sqrt = deg ** -0.5
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        # Step 4 and 5: Propagate message
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        # x_j: num_edges x out_dim. Node feature of source node of each edge
        return norm.unsqueeze(dim=1) * x_j


class GATLayer(nn.Module):
    '''Single-head attention'''

    def __init__(self, in_dim, out_dim, dropout, concat=True):
        super().__init__()
        self.concat = concat
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.xavier_uniform_(self.W.weight, gain=1.414)

        a = nn.Linear(2 * out_dim, 1, bias=False)
        nn.init.xavier_uniform_(a.weight, gain=1.414)
        self.att1 = nn.Sequential(
            a,
            nn.Flatten(),
            nn.LeakyReLU(0.2),
        )
        self.att2 = nn.Sequential(
            nn.Softmax(dim=1),
            # nn.Dropout(dropout)
        )

    def forward(self, x, edge_index):
        '''This is a naive implementation without using edge_index and sparse tensor.'''
        h = self.W(x)  # num_nodes x out_dim
        a = self.pairwise_cat(h)  # num_nodes x num_nodes x dim * 2
        attention = self.att1(a)  # num_nodes x num_nodes
        # Masked attention
        mask = torch.ones_like(attention, dtype=bool)
        row, col = edge_index
        mask[row, col] = 0
        attention.masked_fill_(mask, -float('inf'))
        attention = self.att2(attention)
        return attention @ h if not self.concat else F.elu(attention @ h)
        # return F.elu(attention @ h)

    def pairwise_cat(self, x):
        # x: num_nodes x dim
        # Return: num_nodes x num_nodes x dim * 2
        num_nodes = x.shape[0]
        left = x.repeat(1, num_nodes).view(num_nodes, num_nodes, -1)
        right = x.repeat(num_nodes, 1).view(num_nodes, num_nodes, -1)
        # left[i, j] = x[i]
        # right[i, j] = x[j]
        return torch.cat([left, right], dim=2)  # num_nodes x num_nodes x dim * 2


class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, in_head, out_head):
        super().__init__()
        p = 0.6

        # self.conv1 = gnn.GATConv(in_dim, hidden_dim, heads=in_head, dropout=p)
        # self.conv2 = gnn.GATConv(hidden_dim * in_head, out_dim, heads=out_head, dropout=p, concat=False)
        self.conv1 = GATLayer(in_dim, hidden_dim, dropout=p)
        self.conv2 = GATLayer(hidden_dim, out_dim, dropout=p, concat=False)
        self.dropout = nn.Dropout(p)
        self.act = nn.ELU()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.act(self.conv1(self.dropout(x), edge_index))
        x = self.conv2(self.dropout(x), edge_index)
        return x


def train(model, data, optimizer, criterion):
    for epoch in range(101):
        model.train()
        pred = model(data)
        loss = criterion(pred[data.train_mask], data.y[data.train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if not epoch % 10:
            with torch.no_grad():
                model.eval()
                pred = model(data).argmax(dim=1)
                acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean()
                print(epoch, 'test_acc', acc)


if __name__ == '__main__':
    DEVICE = 'cuda:9'
    DATASET_NAME = 'Cora'
    dataset = datasets.Planetoid(root=f'/home/tiankang/wusuowei/data/{DATASET_NAME}', name=DATASET_NAME)
    dataset.transform = T.NormalizeFeatures()
    print(type(dataset))
    print('#Classes:', dataset.num_classes)
    print('#Node features:', dataset.num_node_features)
    data = dataset[0]
    model = GAT(dataset.num_features, 30, dataset.num_classes, 1, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    train(model.to(DEVICE), data.to(DEVICE), optimizer, criterion)
