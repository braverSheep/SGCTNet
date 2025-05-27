
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch_geometric.nn import SGConv,global_add_pool,global_mean_pool,global_max_pool
from torch_scatter import scatter_add

def maybe_num_nodes(index, num_nodes=None):
    return index.max().item() + 1 if num_nodes is None else num_nodes

def add_remaining_self_loops(edge_index,
                             edge_weight=None,
                             fill_value=1,
                             num_nodes=None):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    row, col = edge_index   
    mask = row != col   
    row = row.long()
    inv_mask = torch.logical_not(mask) 
    loop_weight = torch.full(    
        (num_nodes, ),
        fill_value,
        dtype=None if edge_weight is None else edge_weight.dtype,
        device=edge_index.device)

    if edge_weight is not None:
        assert edge_weight.numel() == edge_index.size(1) 
                                                         
        remaining_edge_weight = edge_weight[inv_mask]  
        if remaining_edge_weight.numel() > 0:                   
            loop_weight[row[inv_mask]] = remaining_edge_weight  
        edge_weight = torch.cat([edge_weight[mask], loop_weight], dim=0)
        
    loop_index = torch.arange(0, num_nodes, dtype=row.dtype, device=row.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)
    edge_index = torch.cat([edge_index[:, mask], loop_index], dim=1)

    return edge_index, edge_weight

class NewSGConv(SGConv):
    def __init__(self, num_features, num_classes, K=1, cached=False, bias=True):
        super(NewSGConv, self).__init__(num_features, num_classes, K=K, cached=cached, bias=bias)

    #允许边的权重值为负
    def norm(edge_index, num_nodes, edge_weight, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ),
                                     dtype=dtype,
                                     device=edge_index.device
                                     )
        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes
            )
        row, col = edge_index
        row = row.to(torch.int64)

        deg = scatter_add(torch.abs(edge_weight), row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5) # computes D^(-1/2)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0 #将deg_inv_sqrt中等于无穷大（inf）的元素替换为0。

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]#计算S=D^(-1/2)AD^(-1/2)

    #forward是SGConv的重定义done to use the here defined "norm" method for normalization, in substitution of the default "gcn_norm"
    def forward(self, x, edge_index, edge_weight=None):
        if not self.cached or self.cached_result is None:
            edge_index, norm = NewSGConv.norm(# Returns normalized edge_index and edge_weights
                edge_index, x.size(0), edge_weight, dtype=x.dtype)

            edge_index = edge_index.to(torch.int64)
            for k in range(self.K):# see chap 3.1.0, fig.4; we assume K=2
                x = self.propagate(edge_index, x=x, norm=norm)
            self.cached_result = x

        return self.lin(self.cached_result)

    def message(self, x_j, norm):#called by "self.propagate"
        # x_j: (batch_size*num_nodes*num_nodes, num_features)
        # norm: (batch_size*num_nodes*num_nodes, )
        return norm.view(-1, 1) * x_j
          
import Transformer_with_position 
class SGCN_Transformer(torch.nn.Module):
    def __init__(self,num_nodes,learn_edge_weight,edge_weight,num_features,num_hiddens,K):
        super(SGCN_Transformer, self).__init__()
        self.num_hiddens = num_hiddens
        self.num_nodes = num_nodes
        self.SGCN = NewSGConv(num_features=num_features, num_classes=num_hiddens, K=K)
        self.Transformer_encoder = Transformer_with_position.EncoderLayer(n_heads = 5, d_model=num_hiddens, d_k=20, d_v=20)
    
    def forward(self,x, batch_size, edge_index,edge_weight):
        edge_weight_GCN = edge_weight.reshape(-1).repeat(batch_size)
        x = self.SGCN(x , edge_index,edge_weight_GCN)
        x = x.view(batch_size, self.num_nodes, self.num_hiddens)
        x = self.Transformer_encoder(x,edge_weight)
        x = x.reshape(batch_size * self.num_nodes, self.num_hiddens )
        return x
from positional_encodings.torch_encodings import PositionalEncoding1D

class FuseModel(torch.nn.Module):
    def __init__(self, num_nodes, learn_edge_weight, edge_weight, num_features, num_hiddens, num_classes, dropout=0.5):
        super(FuseModel, self).__init__()
        self.num_nodes = num_nodes
        self.num_hiddens = num_hiddens
        self.dropout = dropout
        
        self.SGCN = NewSGConv(num_features=num_features, num_classes=num_hiddens, K=2)
        self.x_transformerAndGCN = torch.nn.ModuleList([SGCN_Transformer(num_nodes,True,edge_weight,num_hiddens,num_hiddens,K=1) for _ in range(12)])
        self.fc = nn.Linear(num_hiddens,num_classes)
        # self.positions = nn.Parameter(torch.zeros(num_nodes, num_hiddens*2))
        
        self.xs, self.ys = torch.tril_indices(self.num_nodes, self.num_nodes, offset=0)
        edge_weight = edge_weight.reshape(self.num_nodes, self.num_nodes)[self.xs, self.ys]
        self.edge_weight = nn.Parameter(edge_weight, requires_grad=learn_edge_weight) 
        self.positions_1d = PositionalEncoding1D(self.num_nodes)
        pass
    
    def forward(self, batch):
        dev = ("cuda" if torch.cuda.is_available() else "cpu")
        batch_psd = batch[0].to(dev)
        batch_de = batch[1].to(dev)
        
        batch_size = batch_psd.batch_size
        edge_index = batch_psd.edge_index
        
        edge_weight = torch.zeros((self.num_nodes, self.num_nodes), device=edge_index.device)
        edge_weight[self.xs.to(edge_weight.device), self.ys.to(edge_weight.device)] = self.edge_weight
        edge_weight = edge_weight + edge_weight.transpose(1,0) - torch.diag(edge_weight.diagonal())
        
        edge_weight_GCN = edge_weight.reshape(-1).repeat(batch_size)
        
        # print(edge_weight_GCN.shape)
        # print(edge_index.shape)
        x=self.SGCN(batch_de.x, edge_index, edge_weight_GCN)#self, x, edge_index, edge_weight=None, 
        
        # x = torch.cat((X_de, X_psd), dim=1)
        #位置嵌入
        x = x.view(batch_size, self.num_nodes, self.num_hiddens)
        x_position = self.positions_1d(x)
        x = x + x_position
        x = x.reshape(batch_size * self.num_nodes, self.num_hiddens )
        
        for x_transformerAndGCN in self.x_transformerAndGCN: 
            x = x_transformerAndGCN(x,batch_size, edge_index,edge_weight) + x
        
        x1 = global_add_pool(x, batch_de.batch, size=batch_size)
        x = F.dropout(x1, p=self.dropout, training=self.training)
        x = self.fc(x)

        return x, edge_weight

