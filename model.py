from HRANlayer import *
import torch.nn as nn
from torch.nn import functional as F
import numpy as np


class SparseInputLinear(nn.Module):
    
    def __init__(self, inp_dim, out_dim):
        super(SparseInputLinear, self).__init__()
        weight = np.zeros((inp_dim, out_dim), dtype=np.float32)
        weight = nn.Parameter(torch.from_numpy(weight))
        bias = np.zeros(out_dim, dtype=np.float32)
        bias = nn.Parameter(torch.from_numpy(bias))
        self.inp_dim, self.out_dim = inp_dim, out_dim
        self.weight, self.bias = weight, bias
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):  # *nn.Linear* does not accept sparse *x*.
        return torch.mm(x, self.weight) + self.bias


class BaseModel(torch.nn.Module):
    '''
    Constructing BaseModel for represation learning

    Parameters
    ----------
    edge_index(Tensor shape of 2 * num_triples): sub_entitye edge and obj_entity index of each edge
    edge_type(Tensor shape of num_triples): relation type of each edge
    params(argparse.Namespace): all hyperparameters for whole project
    num_rels(int): total types of relations
    
    Returns
    -------
    Class of BaseModel
    '''
    def __init__(self, params, edge_index, edge_type, num_rel):
        super(BaseModel, self).__init__()
        self.p = params
        self.edge_index = edge_index
        self.edge_type = edge_type
        self.device = self.edge_index.device
        self.beta = self.p.beta
        self.init_embed = get_param((self.p.num_ent, self.p.init_dim)).to(self.device)
        self.init_rel = get_param((num_rel * 2, self.p.gcn_dim)).to(self.device)
        self.pca = SparseInputLinear(self.p.init_dim, self.p.gcn_dim).to(self.device)
        conv_ls = []
        for i in range(self.p.gcn_layer):
            conv = HRANLayer(self.edge_index, self.edge_type, self.p.init_dim, self.p.gcn_dim, num_rel, params=self.p)
            self.add_module('conv_{}'.format(i), conv)
            conv_ls.append(conv)
        self.conv_ls = conv_ls
        self.register_parameter('bias', Parameter(torch.zeros(self.p.num_ent)))
        self.p = params
        self.act = torch.tanh
        self.bceloss = torch.nn.BCELoss()   #交叉熵
        
    def forward_base(self, sub, rel, beta, drop1, drop2):
        if not self.p.no_enc:  #whether use gnn layer
            x = self.act(self.pca(self.init_embed)).view(-1, self.p.gcn_dim) # [N F]
            r = self.init_rel
            for conv in self.conv_ls:
                x, r = conv(x, r, beta) # N K F
                x = drop1(x)
                r = drop2(r)
        else:
            x = self.init_embed
            r = self.init_rel
            x = drop1(x)
            r = drop2(r)
        sub_emb = torch.index_select(x, 0, sub)
        rel_emb = torch.index_select(r, 0, rel)
        return sub_emb, rel_emb, x
    
    def loss(self, pred, true_label):
        return self.bceloss(pred, true_label)

class HRAN_TransE(BaseModel):
    '''
    Model HRAN+TransE
    
    Parameters
    ----------
    edge_index(Tensor shape of 2 * num_triples): sub_entitye edge and obj_entity index of each edge
    edge_type(Tensor shape of num_triples): relation type of each edge
    params(argparse.Namespace): all hyperparameters for whole project
    
    Returns
    -------
    Class of BaseModel
    '''
    def __init__(self, edge_index, edge_type, params=None):
        super(self.__class__, self).__init__(params, edge_index, edge_type, params.num_rel)
        self.drop = torch.nn.Dropout(self.p.hid_drop)
        gamma_init = torch.FloatTensor([self.p.init_gamma])
        if not self.p.fix_gamma:
            self.register_parameter('gamma', Parameter(gamma_init))
        
    def forward(self, sub, rel, neg_ents):
        sub_emb, rel_emb, all_ent = self.forward_base(sub, rel, self.beta, self.drop, self.drop)
        obj_emb = sub_emb + rel_emb   #batch * dim
        if neg_ents == None:
            neg_obj_emb = torch.stack([all_ent for _ in range(sub.shape[0])], dim=0)
        else:
            neg_obj_emb = all_ent[neg_ents]  #batch * (neg_num + 1)* dim
        x = self.p.gamma - torch.norm(obj_emb.unsqueeze(1) - neg_obj_emb, p=2, dim=2)  #batch * (neg_num + 1) * 1
        x = x.squeeze(-1)  #batch * (neg_num + 1)
        pred = torch.sigmoid(x)
        return pred


class HRAN_DistMult(BaseModel):
    def __init__(self, edge_index, edge_type, params=None):
        super(self.__class__, self).__init__(params, edge_index, edge_type, params.num_rel)
        self.drop = torch.nn.Dropout(self.p.hid_drop)
    
    def forward(self, sub, rel, neg_ents):
        sub_emb, rel_emb, all_ent = self.forward_base(sub, rel, self.beta, self.drop, self.drop)
        obj_emb = sub_emb * rel_emb  #batch * dim
        if neg_ents == None:
            neg_obj_emb = torch.stack([all_ent for _ in range(sub.shape[0])], dim=0)
        else:
            neg_obj_emb = all_ent[neg_ents]    # batch * neg_num * dim
        x = torch.einsum('bk,bnk->bn', [obj_emb, neg_obj_emb])
        pred = torch.sigmoid(x)
        return pred

class HRAN_ConvE(BaseModel):
    def __init__(self, edge_index, edge_type, params=None):
        super(self.__class__, self).__init__(params, edge_index, edge_type, params.num_rel)
        self.embed_dim = self.p.embed_dim
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(self.p.num_filt)
        self.bn2 = torch.nn.BatchNorm1d(self.embed_dim)
        
        self.hidden_drop = torch.nn.Dropout(self.p.hid_drop)
        self.hidden_drop2 = torch.nn.Dropout(self.p.hid_drop2)
        self.feature_drop = torch.nn.Dropout(self.p.feat_drop)
        self.m_conv1 = torch.nn.Conv2d(1, out_channels=self.p.num_filt, kernel_size=(self.p.ker_sz, self.p.ker_sz),
                                       stride=1, padding=0, bias=self.p.bias)
        
        flat_sz_h = int(2 * self.p.k_w) - self.p.ker_sz + 1
        flat_sz_w = self.p.k_h - self.p.ker_sz + 1
        self.flat_sz = flat_sz_h * flat_sz_w * self.p.num_filt
        self.fc = torch.nn.Linear(self.flat_sz, self.embed_dim)
    
    def concat(self, e1_embed, rel_embed):
        e1_embed = e1_embed.view(-1, 1, self.embed_dim)
        rel_embed = rel_embed.view(-1, 1, self.embed_dim)
        stack_inp = torch.cat([e1_embed, rel_embed], 1)
        stack_inp = torch.transpose(stack_inp, 2, 1).reshape((-1, 1, 2 * self.p.k_w, self.p.k_h))
        return stack_inp
    
    def forward(self, sub, rel, neg_ents):
        sub_emb, rel_emb, all_ent = self.forward_base(sub, rel, self.beta, self.hidden_drop, self.hidden_drop)
        if neg_ents == None:
            neg_obj_emb = torch.stack([all_ent for _ in range(sub.shape[0])], dim=0)
        else:
            neg_obj_emb = all_ent[neg_ents]   # batch * neg_num * dim
        
        stk_inp = self.concat(sub_emb, rel_emb)  #batch * 1 * (2*k_w), k_h
        x = self.bn0(stk_inp)
        x = self.m_conv1(x)    #batch * num_ker * flat_size_h * flat_size_w
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_drop(x)
        x = x.view(-1, self.flat_sz) 
        x = self.fc(x)    #batch * dim
        x = self.hidden_drop2(x)
        x = self.bn2(x)
        x = F.relu(x)  #batch * dim
        x = torch.einsum('bk,bnk->bn', [x, neg_obj_emb])
        pred = torch.sigmoid(x)
        return pred

class HRAN_Conv_TransE(BaseModel):
    def __init__(self, edge_index, edge_type, params=None):
        super(self.__class__, self).__init__(params, edge_index, edge_type, params.num_rel)
        self.embed_dim = self.p.embed_dim
        self.bn0 = torch.nn.BatchNorm1d(2)
        self.bn1 = torch.nn.BatchNorm1d(self.p.num_filt)
        self.bn2 = torch.nn.BatchNorm1d(self.embed_dim)
        
        self.hidden_drop = torch.nn.Dropout(self.p.hid_drop)
        self.hidden_drop2 = torch.nn.Dropout(self.p.hid_drop2)
        self.feature_drop = torch.nn.Dropout(self.p.feat_drop)
        
        padding_size = self.p.ker_sz // 2
        self.m_conv1 = torch.nn.Conv1d(2, out_channels=self.p.num_filt, kernel_size=self.p.ker_sz,
                                       stride=1, padding=padding_size, bias=self.p.bias)
        self.flat_sz = self.p.num_filt * self.embed_dim 
        self.fc = torch.nn.Linear(self.flat_sz, self.embed_dim)
        
    def forward(self, sub, rel, neg_ents):
        sub_emb, rel_emb, all_ent = self.forward_base(sub, rel, self.beta, self.hidden_drop, self.hidden_drop)
        if neg_ents == None:
            neg_obj_emb = torch.stack([all_ent for _ in range(sub.shape[0])], dim=0)
        else:
            neg_obj_emb = all_ent[neg_ents]   # batch * neg_num * dim
        sub_emb = sub_emb.view(-1, 1, self.embed_dim)
        rel_emb = rel_emb.view(-1, 1, self.embed_dim)
        stk_inp = torch.cat([sub_emb, rel_emb], dim=1)  # batch * 2 * dim
        x = self.bn0(stk_inp)
        x = self.m_conv1(x)   #batch * num_filter * dim
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_drop(x)
        x = x.view(-1, self.flat_sz) 
        x = self.fc(x)    #batch * dim
        x = self.hidden_drop2(x)
        x = self.bn2(x)
        x = F.relu(x)  #batch * dim
        x = self.p.gamma - torch.norm(x.unsqueeze(1) - neg_obj_emb, p=2, dim=2)  #batch * (neg_num + 1) * 1
        pred = torch.sigmoid(x)
        return pred

class HRAN_ConvD(BaseModel):
    def __init__(self, edge_index, edge_type, params=None):
        super(self.__class__, self).__init__(params, edge_index, edge_type, params.num_rel)
        self.embed_dim = self.p.embed_dim
        self.num_rel = self.p.num_rel * 2
        self.bn0 = torch.nn.BatchNorm1d(2)
        self.bn1 = torch.nn.BatchNorm1d(self.p.num_filt)
        self.bn2 = torch.nn.BatchNorm1d(self.embed_dim)
        
        self.hidden_drop = torch.nn.Dropout(self.p.hid_drop)
        self.hidden_drop2 = torch.nn.Dropout(self.p.hid_drop2)
        self.feature_drop = torch.nn.Dropout(self.p.feat_drop)
        
        self.m_conv1 = torch.nn.Conv1d(2, out_channels=self.p.num_filt, kernel_size=self.p.ker_sz,
                                       stride=1, padding=0, bias=self.p.bias)
        self.flat_sz_w = self.p.embed_dim - self.p.ker_sz + 1
        self.m_conv_list = [self.m_conv1 for _ in range(self.num_rel)]
        self.flat_sz = self.p.num_filt * self.flat_sz_w
        self.fc = torch.nn.Linear(self.flat_sz, self.embed_dim)
        
    def forward(self, sub, rel, neg_ents):
        sub_emb, rel_emb, all_ent = self.forward_base(sub, rel, self.beta, self.hidden_drop, self.hidden_drop)
        if neg_ents == None:
            neg_obj_emb = torch.stack([all_ent for _ in range(sub.shape[0])], dim=0)
        else:
            neg_obj_emb = all_ent[neg_ents]   # batch * neg_num * dim
        sub_emb = sub_emb.view(-1, 1, self.embed_dim)
        rel_emb = rel_emb.view(-1, 1, self.embed_dim)
        stk_inp = torch.cat([sub_emb, rel_emb], dim=1)  # batch * 2 * dim
        x = self.bn0(stk_inp)
        y = []
        for i in range(x.shape[0]):
            index = torch.tensor([i]).to(self.device)
            temp = self.m_conv_list[rel[i].item()](torch.index_select(x, 0, index))
            y.append(temp)
        y = torch.stack(y).reshape(-1, self.p.num_filt, self.flat_sz_w)  # batch * num_filter * flat_sz_w
        y = self.bn1(y)
        y = F.relu(y)
        y = self.feature_drop(y)
        y = y.view(-1, self.flat_sz) 
        y = self.fc(y)    #batch * dim
        y = self.hidden_drop2(y)
        y = self.bn2(y)
        y = F.relu(y)
        y = torch.einsum('bk,bnk->bn', [y, neg_obj_emb])
        pred = torch.sigmoid(y)
        return pred