from typing import Dict, Any, List, Tuple, Union
import os
import luigi
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torchbearer import Trial
from mars_gym.meta_config import ProjectConfig
from mars_gym.model.abstract import RecommenderModule
from mars_gym.model.bandit import BanditPolicy
from mars_gym.torch.init import lecun_normal_init
import pickle
import gc 
import random
from typing import Optional, Callable, Tuple
import math


def load_embedding(_n_items, n_factors, path_item_embedding, path_from_index_mapping, index_mapping, freeze_embedding):

    if path_from_index_mapping and path_item_embedding:
        embs = nn.Embedding(_n_items, n_factors)

        # Load extern index mapping
        if os.path.exists(path_from_index_mapping):
            with open(path_from_index_mapping, "rb") as f:
                from_index_mapping = pickle.load(f)    

        # Load weights embs
        extern_weights = np.loadtxt(path_item_embedding)#*100
        intern_weights = embs.weight.detach().numpy()

        # new_x =  g(f-1(x))
        reverse_index_mapping = {value_: key_ for key_, value_ in index_mapping['ItemID'].items()}
        
        embs_weights = np.array([extern_weights[from_index_mapping['ItemID'][reverse_index_mapping[i]]] if from_index_mapping['ItemID'][reverse_index_mapping[i]] > 1 else intern_weights[i] for i in np.arange(_n_items) ])
        #from_index_mapping['ItemID'][reverse_index_mapping[i]]
        #from IPython import embed; embed()
        embs = nn.Embedding.from_pretrained(torch.from_numpy(embs_weights).float(), freeze=freeze_embedding)
        
    elif path_item_embedding:
        # Load extern embs
        extern_weights = torch.from_numpy(np.loadtxt(path_item_embedding)).float()
    
        embs = nn.Embedding.from_pretrained(extern_weights, freeze=freeze_embedding)
    else:
        embs = nn.Embedding(_n_items, n_factors)

    return embs
    
class TimeEmbedding(nn.Module):
    '''
    https://arxiv.org/pdf/1708.00065.pdf
    https://fridayexperiment.com/how-to-encode-time-property-in-recurrent-neutral-networks/
    '''

    def __init__(self, hidden_embedding_size, output_dim):
        super(TimeEmbedding, self).__init__()
        self.emb_weight = nn.Parameter(torch.randn(1, hidden_embedding_size)) # (1, H)
        self.emb_bias = nn.Parameter(torch.randn(hidden_embedding_size)) # (H)
        self.emb_time = nn.Parameter(torch.randn(hidden_embedding_size, output_dim)) # (H, E)

    def forward(self, input):
        # input (B, W, 1)
        x = torch.softmax(input * self.emb_weight + self.emb_bias, dim=2) # (B, W, H)
        x = torch.matmul(x, self.emb_time) # (B, W, E)
        return x

class GRURecModel(RecommenderModule):
    def __init__(
        self,
        project_config: ProjectConfig,
        index_mapping: Dict[str, Dict[Any, int]],
        n_factors: int,
        hidden_size: int,
        n_layers: int,
        path_item_embedding: str,
        from_index_mapping: str,
        dropout: float,
        freeze_embedding: bool
    ):
        super().__init__(project_config, index_mapping)
        self.path_item_embedding = path_item_embedding
        self.dropout = dropout
        self.hidden_size = hidden_size 
        self.n_layers = n_layers
        self.emb_dropout = nn.Dropout(0.25)

        self.item_embeddings = load_embedding(self._n_items, n_factors, path_item_embedding, 
                                                from_index_mapping, index_mapping, freeze_embedding)

        self.gru = nn.GRU(n_factors, self.hidden_size, self.n_layers, dropout=self.dropout)
        self.out = nn.Linear(self.hidden_size, self._n_items)
        self.sf  = nn.Softmax()

        self.weight_init = lecun_normal_init
        self.apply(self.init_weights)

    def init_weights(self, module: nn.Module):
        if type(module) == nn.Linear:
            self.weight_init(module.weight)
            module.bias.data.fill_(0.1)
            
    def flatten(self, input):
        return input.view(input.size(0), -1)

    def normalize(self, x: torch.Tensor, dim: int = 1) -> torch.Tensor:
        if self.use_normalize:
            x = F.normalize(x, p=2, dim=dim)
        return x

    def forward(self, session_ids, item_ids, item_history_ids):
        embs = self.emb_dropout(self.item_embeddings(item_history_ids))
        
        output, hidden = self.gru(embs)
        #output = output.view(-1, output.size(2))  #(B,H)
        
        #out    = torch.softmax(self.out(output[:,-1]), dim=1)
        out    = self.out(output[:,-1])
        return out

    def recommendation_score(self, session_ids, item_ids, item_history_ids):
        
        scores = self.forward(session_ids, item_ids, item_history_ids)
        scores = scores[torch.arange(scores.size(0)),item_ids]

        return scores

class Caser(RecommenderModule):
    '''
    https://github.com/graytowne/caser_pytorch
    https://arxiv.org/pdf/1809.07426v1.pdf
    '''
    def __init__(
        self,
        project_config: ProjectConfig,
        index_mapping: Dict[str, Dict[Any, int]],
        path_item_embedding: str,
        from_index_mapping: str,
        freeze_embedding: bool,
        n_factors: int,
        p_L: int,
        p_d: int,
        p_nh: int,
        p_nv: int,
        dropout: float,
        hist_size: int,
    ):
        super().__init__(project_config, index_mapping)
        self.path_item_embedding = path_item_embedding
        self.hist_size = hist_size
        self.n_factors = n_factors
        
        # init args
        L = p_L
        dims = p_d
        self.n_h = p_nh
        self.n_v = p_nv
        self.drop_ratio = dropout
        self.ac_conv = F.relu#activation_getter[p_ac_conv]
        self.ac_fc = F.relu#activation_getter[p_ac_fc]
        num_items = self._n_items
        dims = n_factors

        # user and item embeddings
        #self.user_embeddings = nn.Embedding(num_users, dims)
        self.item_embeddings = load_embedding(self._n_items, n_factors, path_item_embedding, 
                                                from_index_mapping, index_mapping, freeze_embedding)


        # vertical conv layer
        self.conv_v = nn.Conv2d(1, self.n_v, (L, 1))

        # horizontal conv layer
        lengths = [i + 1 for i in range(L)]
        self.conv_h = nn.ModuleList([nn.Conv2d(1, self.n_h, (i, dims)) for i in lengths])

        # fully-connected layer
        self.fc1_dim_v = self.n_v * dims
        self.fc1_dim_h = self.n_h * len(lengths)
        fc1_dim_in = self.fc1_dim_v + self.fc1_dim_h
        # W1, b1 can be encoded with nn.Linear
        self.fc1 = nn.Linear(fc1_dim_in, dims)
        # W2, b2 are encoded with nn.Embedding, as we don't need to compute scores for all items
        self.W2 = nn.Embedding(num_items, dims) #+dims
        self.b2 = nn.Embedding(num_items, 1)

        # dropout
        self.dropout = nn.Dropout(self.drop_ratio)

        # weight initialization
        #self.user_embeddings.weight.data.normal_(0, 1.0 / self.user_embeddings.embedding_dim)
        self.item_embeddings.weight.data.normal_(0, 1.0 / self.item_embeddings.embedding_dim)
        self.W2.weight.data.normal_(0, 1.0 / self.W2.embedding_dim)
        self.b2.weight.data.zero_()
        self.b = nn.Linear(n_factors, n_factors)

        self.cache_x = None

        self.out = nn.Linear(self.n_factors, self._n_items)

    def forward(self, session_ids, item_ids, item_history_ids): # for training        
        """
        The forward propagation used to get recommendation scores, given
        triplet (user, sequence, targets).
        Parameters
        ----------
        item_history_ids: torch.FloatTensor with size [batch_size, max_sequence_length]
            a batch of sequence
        user_var: torch.LongTensor with size [batch_size]
            a batch of user
        item_ids: torch.LongTensor with size [batch_size]
            a batch of items
        for_pred: boolean, optional
            Train or Prediction. Set to True when evaluation.
        """

        # Embedding Look-up
        item_embs = self.item_embeddings(item_history_ids).unsqueeze(1)  # use unsqueeze() to get 4-D
        #user_emb = self.user_embeddings(user_var).squeeze(1)

        # Convolutional Layers
        out, out_h, out_v = None, None, None
        # vertical conv layer
        if self.n_v:
            out_v = self.conv_v(item_embs)
            out_v = out_v.view(-1, self.fc1_dim_v)  # prepare for fully connect

        # horizontal conv layer
        out_hs = list()
        if self.n_h:
            for conv in self.conv_h:
                conv_out = self.ac_conv(conv(item_embs).squeeze(3))
                pool_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
                out_hs.append(pool_out)
            out_h = torch.cat(out_hs, 1)  # prepare for fully connect

        # Fully-connected Layers
        out = torch.cat([out_v, out_h], 1)
        # apply dropout
        out = self.dropout(out)

        # fully-connected layer
        z = self.ac_fc(self.fc1(out))

        item_embs   = self.item_embeddings(torch.arange(self._n_items).to(item_history_ids.device).long()) # (Dim, E)
        scores      = torch.matmul(z, self.b(item_embs).permute(1, 0)) # (B, dim)

        return scores

    def recommendation_score(self, session_ids, item_ids, item_history_ids):
        
        scores = self.forward(session_ids, item_ids, item_history_ids)
        scores = scores[torch.arange(scores.size(0)),item_ids]

        return scores    

class NARMModel(RecommenderModule):
    '''
    https://github.com/Wang-Shuo/Neural-Attentive-Session-Based-Recommendation-PyTorch.git
    '''
    def __init__(
        self,
        project_config: ProjectConfig,
        index_mapping: Dict[str, Dict[Any, int]],
        n_factors: int,
        n_layers: int,
        hidden_size: int,
        path_item_embedding: str,
        from_index_mapping: str,
        freeze_embedding: bool,        
        dropout: float        
    ):
        super().__init__(project_config, index_mapping)

        self.hidden_size    = hidden_size
        self.n_layers       = n_layers
        self.embedding_dim  = n_factors
        self.n_time_dim     = 100
        self.n_month_dim    = 10

        self.emb         = load_embedding(self._n_items, n_factors, path_item_embedding, 
                                        from_index_mapping, index_mapping, freeze_embedding)
        self.time_emb    = TimeEmbedding(self.n_time_dim, n_factors)
        self.month_emb   = nn.Embedding(13, self.n_month_dim)

        self.emb_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(self.embedding_dim + self.n_month_dim, 
                            self.hidden_size, self.n_layers, bidirectional=True)

        self.a_1 = nn.Linear(self.hidden_size * 2, self.hidden_size * 2, bias=False)
        self.a_2 = nn.Linear(self.hidden_size, self.hidden_size * 2, bias=False)
        self.v_t = nn.Linear(self.hidden_size * 2, 1, bias=False)
        
        self.ct_dropout = nn.Dropout(dropout)
        self.b = nn.Linear(self.embedding_dim, self.hidden_size * 3, bias=False)
        
    def forward(self, session_ids, item_ids, item_history_ids, duration_list, trip_month):
        device = item_ids.device
        seq    = item_history_ids.permute(1,0)  #TODO

        hidden  = self.init_hidden(seq.size(1)).to(device)
        embs    = self.emb_dropout(self.emb(seq))
        
        # Time/Month Embs
        m_embs  = self.emb_dropout(self.month_emb(trip_month.long()))
        m_embs  = m_embs.unsqueeze(0).expand(seq.shape[0], seq.shape[1], self.n_month_dim)

        # Add Time
        t_embs  = self.time_emb(duration_list.float().unsqueeze(2)).permute(1,0,2)  # (H, B, E)
        embs    = (embs + t_embs)/2

        embs    = torch.cat([embs, m_embs], 2)      
        gru_out, hidden = self.gru(embs, hidden)

        # fetch the last hidden state of last timestamp
        ht      = hidden[-1]
        gru_out = gru_out.permute(1, 0, 2)

        c_global = ht
        q1 = self.a_1(gru_out.contiguous().view(-1, self.hidden_size * 2)).view(gru_out.size())  
        q2 = self.a_2(ht)

        mask      = torch.where(seq.permute(1, 0) > 0, torch.tensor([1.], device = device), 
                        torch.tensor([0.], device = device))

        q2_expand = q2.unsqueeze(1).expand_as(q1)
        q2_masked = mask.unsqueeze(2).expand_as(q1) * q2_expand

        alpha   = self.v_t(torch.sigmoid(q1 + q2_masked)\
                    .view(-1, self.hidden_size * 2))\
                    .view(mask.size())
        c_local = torch.sum(alpha.unsqueeze(2).expand_as(gru_out) * gru_out, 1)

        c_t     = torch.cat([c_local, c_global], 1)
        c_t     = self.ct_dropout(c_t)
        
        item_embs = self.emb(torch.arange(self._n_items).to(device).long())
        scores  = torch.matmul(c_t, self.b(item_embs).permute(1, 0))

        return scores

    def recommendation_score(self, session_ids, item_ids, item_history_ids, duration_list, trip_month):
        
        scores = self.forward(session_ids, item_ids, item_history_ids, duration_list, trip_month)
        scores = scores[torch.arange(scores.size(0)),item_ids]

        return scores

    def init_hidden(self, batch_size):
        return torch.zeros((self.n_layers * 2, batch_size, self.hidden_size), requires_grad=True)        


class NARMTimeSpaceModel(RecommenderModule):
    '''
    https://github.com/Wang-Shuo/Neural-Attentive-Session-Based-Recommendation-PyTorch.git
    '''
    def __init__(
        self,
        project_config: ProjectConfig,
        index_mapping: Dict[str, Dict[Any, int]],
        n_factors: int,
        n_layers: int,
        hidden_size: int,
        path_item_embedding: str,
        from_index_mapping: str,
        freeze_embedding: bool,        
        dropout: float        
    ):
        super().__init__(project_config, index_mapping)

        self.hidden_size    = hidden_size
        self.n_layers       = n_layers
        self.embedding_dim  = n_factors
        n_time_dim = 100
        #self.emb = nn.Embedding(self._n_items, self.embedding_dim, padding_idx = 0)

        self.emb = load_embedding(self._n_items, n_factors, path_item_embedding, 
                                                from_index_mapping, index_mapping, freeze_embedding)
        self.time_emb   = TimeEmbedding(n_time_dim, n_factors)
        
        self.project_matrix = nn.Embedding(15, self.embedding_dim*self.embedding_dim)

        self.emb_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(self.embedding_dim, self.hidden_size, self.n_layers)

        self.a_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.a_2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_t = nn.Linear(self.hidden_size, 1, bias=False)
        
        self.ct_dropout = nn.Dropout(dropout)
        self.b = nn.Linear(self.embedding_dim, 2 * self.hidden_size, bias=False)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, session_ids, item_ids, item_history_ids, duration_list, trip_month):
        device = item_ids.device
        seq    = item_history_ids.permute(1,0)  #TODO

        hidden  = self.init_hidden(seq.size(1)).to(device)
        embs    = self.emb_dropout(self.emb(seq))
        matrix_time = self.project_matrix(trip_month.long()).view(
            trip_month.shape[0], self.embedding_dim, self.embedding_dim)


        # Add Time
        t_embs  = self.time_emb(duration_list.float().unsqueeze(2)).permute(1,0,2)  # (H, B, E)
        embs    = (embs + t_embs)/2

        # [locate_shift_ids.reshape(-1).long()]
        embs = torch.matmul(embs.permute(1,0,2), matrix_time).permute(1,0,2)

        gru_out, hidden = self.gru(embs, hidden)

        # fetch the last hidden state of last timestamp
        ht      = hidden[-1]
        gru_out = gru_out.permute(1, 0, 2)

        c_global = ht
        q1 = self.a_1(gru_out.contiguous().view(-1, self.hidden_size)).view(gru_out.size())  
        q2 = self.a_2(ht)

        mask      = torch.where(seq.permute(1, 0) > 0, torch.tensor([1.], device = device), 
                        torch.tensor([0.], device = device))

        q2_expand = q2.unsqueeze(1).expand_as(q1)
        q2_masked = mask.unsqueeze(2).expand_as(q1) * q2_expand

        alpha   = self.v_t(torch.sigmoid(q1 + q2_masked)\
                    .view(-1, self.hidden_size))\
                    .view(mask.size())
        c_local = torch.sum(alpha.unsqueeze(2).expand_as(gru_out) * gru_out, 1)

        c_t     = torch.cat([c_local, c_global], 1)
        c_t     = self.ct_dropout(c_t)
        
        #item_embs   = self.emb(torch.arange(self._n_items).to(device).long()).expand_as(q1)
        
        item_embs = self.emb(torch.arange(self._n_items).to(device).long()).unsqueeze(0).expand(c_t.shape[0], self._n_items, self.embedding_dim)
        item_embs = torch.matmul(item_embs, matrix_time)
        #scores  = torch.matmul(c_t, self.b(item_embs).permute(0, 1, 2))
        scores    = torch.matmul(c_t.unsqueeze(1), self.b(item_embs).permute(0,2, 1)).squeeze(1)#torch.matmul(c_t.unsqueeze(1), self.b(item_embs).permute(0,2, 1))
        return scores

    def recommendation_score(self, session_ids, item_ids, item_history_ids, duration_list, trip_month):
        
        scores = self.forward(session_ids, item_ids, item_history_ids, duration_list, trip_month)
        scores = scores[torch.arange(scores.size(0)),item_ids]

        return scores

    def init_hidden(self, batch_size):
        return torch.zeros((self.n_layers, batch_size, self.hidden_size), requires_grad=True)          