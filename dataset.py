from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import torch

class TrainDataset(Dataset):
    """
    Training Dataset class.

    Parameters
    ----------
    triples:    The triples used for training the model
    params:     Parameters for the experiments

    Returns
    -------
    A training Dataset class instance used by DataLoader
    """
    
    def __init__(self, triples, params):
        self.triples = triples
        self.p = params
        self.entities = np.arange(self.p.num_ent, dtype=np.int32)
    
    def __len__(self):
        return len(self.triples)
    
    def __getitem__(self, idx):
        ele = self.triples[idx]
        triple, label = torch.LongTensor(ele['triple']), np.int32(ele['label'])
        trp_label = self.get_label(label)
        if self.p.lbl_smooth != 0.0:
            trp_label = (1.0 - self.p.lbl_smooth) * trp_label + (1.0 / self.p.num_ent)

        if self.p.strategy == 'one_to_n':
            return triple, trp_label, None

        elif self.p.strategy == 'one_to_x':
            neg_ent = torch.LongTensor(self.get_neg_ent(triple, label))
            return triple, trp_label, neg_ent
        else:
            raise NotImplementedError
        
    def get_label(self, label):
        '''
        for one data, creat label for training
        
        Parameters
        ----------
        label : list of correct index
        
        Returns
        -------
        y: vector which lenth equals to num_ent(one_to_n) or neg_num+1(one_to_x)
        '''
        if self.p.strategy == 'one_to_n':
            y = np.zeros([self.p.num_ent], dtype=np.float32)
            for e2 in label: y[e2] = 1.0
        elif self.p.strategy == 'one_to_x':
            y = [1] + [0] * self.p.neg_num
        else:
            raise NotImplementedError
        return torch.FloatTensor(y)
    
    def get_neg_ent(self, triple, label):
        def get(triple, label):
            if self.p.strategy == 'one_to_x':
                pos_obj = triple[2]  #待预测正样本的id
                entities = list(set(self.entities) - set(label))
                neg_ent = np.int32(np.random.choice(entities, self.p.neg_num, replace=False)).reshape([-1])
                neg_ent = np.concatenate(([pos_obj], neg_ent))
            return neg_ent
        
        neg_ent = get(triple, label)
        return neg_ent
    
    @staticmethod
    def collate_fn(data):
        triple = torch.stack([_[0] for _ in data], dim=0)
        trp_label = torch.stack([_[1] for _ in data], dim=0)
        # triple: (batch-size) * 3(sub, rel, -1) trp_label (batch-size) * num entity
        # return triple, trp_label
        if not data[0][2] is None:  # one_to_x
            neg_ent = torch.stack([_[2] for _ in data], dim=0)
            return triple, trp_label, neg_ent
        else:
            return triple, trp_label


class TestDataset(Dataset):
    """
    Evaluation Dataset class.

    Parameters
    ----------
    triples:    The triples used for evaluating the model
    params:     Parameters for the experiments

    Returns
    -------
    An evaluation Dataset class instance used by DataLoader for model evaluation
    """
    def __init__(self, triples, params):
        self.triples = triples
        self.p = params

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        ele = self.triples[idx]
        triple, label = torch.LongTensor(ele['triple']), np.int32(ele['label'])
        label = self.get_label(label)

        return triple, label
    
    def get_label(self, label):
        y = np.zeros([self.p.num_ent], dtype=np.float32)
        for e2 in label: y[e2] = 1.0
        return torch.FloatTensor(y)
    
    @staticmethod
    def collate_fn(data):
        triple = torch.stack([_[0] for _ in data], dim=0)
        label = torch.stack([_[1] for _ in data], dim=0)
        return triple, label