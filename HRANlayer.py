from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
import torch
from torch.nn import Parameter, Linear
from torch.nn.init import xavier_normal_
import torch_scatter.scatter as scatter

def get_param(shape):
    param = Parameter(torch.Tensor(*shape));
    xavier_normal_(param.data)
    return param

class HRANLayer(MessagePassing):
    '''
    Constructing Entity-level Aggregation and Relation-lavel Aggregation Layer

    Parameters
    ----------
    edge_index(Tensor shape of 2 * num_triples): sub_entitye edge and obj_entity index of each edge
    edge_type(Tensor shape of num_triples): relation type of each edge
    params(argparse.Namespace): all hyperparameters for whole project
    in_channels(int): input dim of HRANLayer(Initial dim)
    out_channel(int): output dim of HRANLayer(dim for computing score function)
    num_rels(int): total types of relations
    act(callable function): activation function for output
    head_num(int): num of attention heads
    att_dim(int): dim of attention vector q
    
    Returns
    -------
    Class of HRANLayer
    '''
    def __init__(self, edge_index, edge_type, in_channels, out_channels, num_rels, params=None, head_num=1, att_dim=100):
        super(self.__class__, self).__init__(aggr=params.aggr, flow='target_to_source', node_dim=0)
        self.edge_index = edge_index
        self.edge_type = edge_type
        self.p = params
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.device = None
        self.head_num = head_num
        self.num_rels = num_rels
        self.drop = torch.nn.Dropout(self.p.dropout)  # drop ratio for attention coefficient
        self.bn = torch.nn.BatchNorm1d(out_channels)   # batch normalization for gnn output with each entity embedding
        if self.p.bias: # whether add bias for gnn
            self.register_parameter('bias', Parameter(torch.zeros(out_channels)))
        num_edges = self.edge_index.size(1) // 2
        if self.device is None:
            self.device = self.edge_index.device
        self.in_index, self.out_index = self.edge_index[:, :num_edges], self.edge_index[:, num_edges:]
        self.in_type, self.out_type = self.edge_type[:num_edges], self.edge_type[num_edges:]
        self.loop_index = torch.stack([torch.arange(self.p.num_ent), torch.arange(self.p.num_ent)]).to(self.device)
        self.loop_type = torch.full((self.p.num_ent,), 2 * self.num_rels, dtype=torch.long).to(self.device)
        self.loop_rel = get_param((1, out_channels)).to(self.device)
        self.rel_linear = Linear(out_channels, out_channels, False)
        self.ent_linear = Linear(out_channels, out_channels, False)
        self.rel_attention_transformation_weight = get_param((out_channels, att_dim))
        self.attention_q = get_param((att_dim, 1))
        self.ent_level_aggre_index = torch.zeros(self.edge_index.shape[1] + self.p.num_ent, dtype=torch.int64).to(self.device)
        self.compute_ent_level_aggre_index()
        
        
        
    def forward(self, x, rel_embed, beta):
        rel_embed = torch.cat([rel_embed, self.loop_rel], dim=0)
        edge_index = torch.cat([self.edge_index, self.loop_index], dim=1)
        edge_type = torch.cat([self.edge_type, self.loop_type])
        out = self.propagate(edge_index, size=None, x=x, edge_type=edge_type, rel_embed=rel_embed, attention_q=self.attention_q, rel_attention_transformation_weight=self.rel_attention_transformation_weight)
        if self.p.bias:
            out = out + self.bias
        out = (1 - beta) * out + beta * x
        out = self.ent_linear(out)
        out = torch.nn.functional.relu(out)
        out = self.bn(out)
        rel_embed = self.rel_linear(rel_embed)
        rel_mebed = torch.nn.functional.relu(rel_embed)
        return out, rel_embed[:-1]
    
    def compute_ent_level_aggre_index(self):
        edge_index = torch.cat([self.edge_index, self.loop_index], dim=1)
        edge_type = torch.cat([self.edge_type, self.loop_type])
        edge_index_i = edge_index[0,:]
        temp_index = 0
        ent_rel_type_set = dict()
        for i in range(edge_index_i.shape[0]):
            if (edge_index_i[i].item(), edge_type[i].item()) not in ent_rel_type_set:
                self.ent_level_aggre_index[i] = temp_index
                ent_rel_type_set[(edge_index_i[i].item(), edge_type[i].item())] = temp_index
                temp_index += 1
            else:
                self.ent_level_aggre_index[i] = ent_rel_type_set[(edge_index_i[i].item(), edge_type[i].item())]
        # print(sorted(list(ent_rel_type_set.items()), key=lambda x: x[0][0], reverse=True)[:10])
    
    def message(self, x, edge_index_i, edge_index_j, edge_type, rel_embed, attention_q, rel_attention_transformation_weight):
        x = torch.index_select(x, 0, edge_index_j)
        x = scatter(x, self.ent_level_aggre_index, dim=0, reduce='add')   #entity-level aggregation
        
        edge_type_node = scatter(edge_type, self.ent_level_aggre_index, dim=0, reduce='mean')
        # print('shape after node aggr:', edge_type_node.shape)
        rel_embed = torch.matmul(rel_embed, rel_attention_transformation_weight)
        rel_embed = torch.tanh(rel_embed)
        rel_embed = torch.index_select(rel_embed, dim=0, index=edge_type_node)
        rel_embed = self.drop(rel_embed)
        rel_embed = torch.matmul(rel_embed, attention_q)
        rel_embed = torch.sigmoid(rel_embed)
        return rel_embed * x
    
    def aggregate(self, inputs, index):
        rel_level_aggre_index = scatter(index, self.ent_level_aggre_index, dim=0, reduce='mean')
        out = scatter(inputs, rel_level_aggre_index, dim=self.node_dim, reduce=self.aggr)
        # print('shape after aggregate:', out.shape)
        return out
    
    def __repr__(self):
        return '{}({}, {}, num_rels={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.num_rels)