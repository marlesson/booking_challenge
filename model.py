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

IDX_FIX = 3




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

class Attention(nn.Module):
    """ Applies attention mechanism on the `context` using the `query`.

    **Thank you** to IBM for their initial implementation of :class:`Attention`. Here is
    their `License
    <https://github.com/IBM/pytorch-seq2seq/blob/master/LICENSE>`__.

    Args:
        dimensions (int): Dimensionality of the query and context.
        attention_type (str, optional): How to compute the attention score:

            * dot: :math:`score(H_j,q) = H_j^T q`
            * general: :math:`score(H_j, q) = H_j^T W_a q`

    Example:

         >>> attention = Attention(256)
         >>> query = torch.randn(5, 1, 256)
         >>> context = torch.randn(5, 5, 256)
         >>> output, weights = attention(query, context)
         >>> output.size()
         torch.Size([5, 1, 256])
         >>> weights.size()
         torch.Size([5, 1, 5])
    """

    def __init__(self, dimensions, attention_type='general'):
        super(Attention, self).__init__()

        if attention_type not in ['dot', 'general']:
            raise ValueError('Invalid attention type selected.')

        self.attention_type = attention_type
        if self.attention_type == 'general':
            self.linear_in = nn.Linear(dimensions, dimensions, bias=False)

        self.linear_out = nn.Linear(dimensions * 2, dimensions, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def forward(self, query, context):
        """
        Args:
            query (:class:`torch.FloatTensor` [batch size, output length, dimensions]): Sequence of
                queries to query the context.
            context (:class:`torch.FloatTensor` [batch size, query length, dimensions]): Data
                overwhich to apply the attention mechanism.

        Returns:
            :class:`tuple` with `output` and `weights`:
            * **output** (:class:`torch.LongTensor` [batch size, output length, dimensions]):
              Tensor containing the attended features.
            * **weights** (:class:`torch.FloatTensor` [batch size, output length, query length]):
              Tensor containing attention weights.
        """
        batch_size, output_len, dimensions = query.size()
        query_len = context.size(1)

        if self.attention_type == "general":
            query = query.reshape(batch_size * output_len, dimensions)
            query = self.linear_in(query)
            query = query.reshape(batch_size, output_len, dimensions)

        # TODO: Include mask on PADDING_INDEX?

        # (batch_size, output_len, dimensions) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, query_len)
        attention_scores = torch.bmm(query, context.transpose(1, 2).contiguous())

        # Compute weights across every context sequence
        attention_scores = attention_scores.view(batch_size * output_len, query_len)
        attention_weights = self.softmax(attention_scores)
        attention_weights = attention_weights.view(batch_size, output_len, query_len)

        # (batch_size, output_len, query_len) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, dimensions)
        mix = torch.bmm(attention_weights, context)

        # concat -> (batch_size * output_len, 2*dimensions)
        combined = torch.cat((mix, query), dim=2)
        combined = combined.view(batch_size * output_len, 2 * dimensions)

        # Apply linear_out on every 2nd dimension of concat
        # output -> (batch_size, output_len, dimensions)
        output = self.linear_out(combined).view(batch_size, output_len, dimensions)
        output = self.tanh(output)

        return output, attention_weights

# Temporarily leave PositionalEncoding module here. Will be moved somewhere else.
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# --------------------------

class MLTransformerModel(RecommenderModule):
    def __init__(
        self,
        project_config: ProjectConfig,
        index_mapping: Dict[str, Dict[Any, int]],
        n_factors: int,
        n_hid: int,
        n_head: int,
        n_layers: int,
        num_filters: int,
        path_item_embedding: str,
        from_index_mapping: str,
        dropout: float,
        hist_size: int,
        freeze_embedding: bool
    ):
        super().__init__(project_config, index_mapping)
        self.path_item_embedding = path_item_embedding
        
        filter_sizes: List[int] = [3, 4, 5]# [1, 3, 5] #1 3 5
        self.n_time_dim  = 100
        self.n_month_dim = 10
        self.item_emb = load_embedding(self._n_items, n_factors, path_item_embedding, 
                                                from_index_mapping, index_mapping, freeze_embedding)
        self.emb_drop = nn.Dropout(p=dropout)
        self.dropout  = nn.Dropout(p=dropout)

        n_factors_2 = n_factors #+ self.n_month_dim

        self.pos_encoder    = PositionalEncoding(n_factors_2, dropout)
        encoder_layers      =  nn.TransformerEncoderLayer(n_factors_2, n_head, n_hid, dropout)
        self.transformer_encoder =  nn.TransformerEncoder(encoder_layers, n_layers)

        self.time_emb    = TimeEmbedding(self.n_time_dim, n_factors)
        self.month_emb   = nn.Embedding(13, self.n_month_dim)

        # self.convs = nn.ModuleList(
        #     [nn.Conv1d(1, num_filters, K*n_factors_2, stride=n_factors_2) for K in filter_sizes])

        # conv_size_out = len(filter_sizes) * num_filters
        #conv_size_out = n_factors_2*hist_size
        # self.dense = nn.Sequential(
        #     nn.BatchNorm1d(conv_size_out + n_factors),
        #     nn.Linear(conv_size_out + n_factors, conv_size_out + n_factors),
        #     nn.ReLU(),
        #     nn.Linear(conv_size_out + n_factors, n_factors),
        # )
        # self.d1 = nn.Sequential(
        #     nn.BatchNorm1d(n_factors * hist_size),
        #     nn.Linear(n_factors * hist_size, n_factors),
        # )

        # self.d2 = nn.Sequential(
        #     nn.BatchNorm1d(n_factors),
        #     nn.Linear(n_factors, n_factors),
        #     nn.ReLU()
        # )
        self.activate_func = nn.ReLU()
        self.dense = nn.Sequential(
            nn.BatchNorm1d(n_factors_2 * hist_size),
            nn.Linear(n_factors_2 * hist_size, n_factors),
            self.activate_func,
            nn.Linear(n_factors, n_factors),
            self.activate_func
        )
        self.b = nn.Linear(n_factors, n_factors)

        self.use_normalize = False
        self.weight_init = lecun_normal_init
        self.apply(self.init_weights)

    def init_weights(self, module: nn.Module):
        if type(module) == nn.Linear:
            self.weight_init(module.weight)
            module.bias.data.fill_(0.1)
    
    #We can then build a convenient cloning function that can generate multiple layers:
    def get_clones(self, module, N):
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
            
    def layer_transform(self, x, mask=None):
        for i in range(self.transform_n):
            x = self.layers[i](x, mask)
        x = self.norm(x)
        return x            

    def flatten(self, input):
        return input.view(input.size(0), -1)

    def normalize(self, x: torch.Tensor, dim: int = 1) -> torch.Tensor:
        if self.use_normalize:
            x = F.normalize(x, p=2, dim=dim)
        return x
    
    def src_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def layer_transformer(self, src, mask_hist):
        # Create transform mask
        mask          = self.src_mask(len(src)).to(src.device)
        src           = self.pos_encoder(src*mask_hist)
        att_hist_emb  = self.transformer_encoder(src, mask) # (B, H, E)
        
        return att_hist_emb

    def forward(self, session_ids, 
                        item_ids, 
                        item_history_ids, 
                        user_features,
                        hotel_country_list,
                        duration_list, 
                        trip_month, 
                        dense_features):
        # Item History embs
        item_hist_emb   =   self.emb_drop(
                                self.normalize(self.item_emb(item_history_ids), 2)
                            )#[0] # (B, H, E)
        
        # Time/Month Embs
        #m_emb   = self.emb_drop(self.month_emb(trip_month.long()))
        #m_embs  = m_emb.unsqueeze(1).expand(item_history_ids.shape[0], item_history_ids.shape[1], self.n_month_dim)

        # Add Time
        #t_embs  = self.time_emb(duration_list.float().unsqueeze(2))  # (H, B, E)
        #item_hist_emb    = (item_hist_emb + t_embs)/2
        
        #item_hist_emb    = torch.cat([item_hist_emb, m_embs], 2)      

        # Mask history
        mask_hist   = (item_history_ids != 0).to(item_history_ids.device).float()
        mask_hist   = mask_hist.unsqueeze(1).repeat((1,item_hist_emb.size(2),1)).permute(0,2,1)

        # Create transform mask
        att_hist_emb = self.layer_transformer(item_hist_emb, mask_hist) # (B, H, E)

        # Add time emb
        hist_conv   = att_hist_emb
        
        # Last Item
        last_item   = item_hist_emb[:,0]

        #join        = torch.cat([self.d2(last_item), 
        #                        self.flatten(hist_conv)], 1)
        join        = self.flatten(hist_conv)

        pred_emb    = self.dense(join) # (B, E)
        pred_emb    = self.dropout(pred_emb)
        # Predict

        item_embs   = self.item_emb(torch.arange(self._n_items).to(item_history_ids.device).long()) # (Dim, E)
        scores      = torch.matmul(pred_emb, self.b(item_embs).permute(1, 0)) # (B, dim)

        return scores

    def recommendation_score(self, session_ids, 
                        item_ids, 
                        item_history_ids, 
                        user_features,
                        hotel_country_list,
                        duration_list, 
                        trip_month, 
                        dense_features):
        
        scores = self.forward(session_ids, 
                        item_ids, 
                        item_history_ids, 
                        user_features,
                        hotel_country_list,
                        duration_list, 
                        trip_month, 
                        dense_features)
        scores = scores[torch.arange(scores.size(0)),item_ids]

        return scores

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

class RNNAttModel(RecommenderModule):
    def __init__(
        self,
        project_config: ProjectConfig,
        index_mapping: Dict[str, Dict[Any, int]],
        n_factors: int,
        hidden_size: int,
        n_layers: int,
        window_trip: int,
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
        self.n_factors = n_factors

        self.n_time_dim  = 100
        self.n_month_dim = 100
        self.window_trip = window_trip
        self.emb_dropout = nn.Dropout(self.dropout)

        n_booker_country_list_dim   = self.index_mapping_max_value('booker_country_list')
        self.emb_country = nn.Embedding(n_booker_country_list_dim, n_factors)
        self.emb_time    = TimeEmbedding(self.n_time_dim, n_factors)
        self.emb_month   = nn.Embedding(13, self.n_month_dim)
        self.emb_user    = nn.Embedding(self._n_users, n_factors)

        self.emb_item    = load_embedding(self._n_items, n_factors, path_item_embedding, 
                                        from_index_mapping, index_mapping, freeze_embedding)

        self.gru1 = nn.GRU(n_factors, self.hidden_size, self.n_layers, 
                            dropout=self.dropout, bidirectional=True)
        

        self.gru2 = nn.GRU(n_factors, self.hidden_size, self.n_layers, 
                            dropout=self.dropout, bidirectional=True)
                
        self.activate_func = nn.SELU()
        self.mlp_dense = nn.Sequential(
            nn.Linear(self.hidden_size * 2 * self.window_trip + self.hidden_size * 2, self.hidden_size * 2 * self.window_trip),
            self.activate_func,
            nn.Linear(self.hidden_size * 2 * self.window_trip, self.hidden_size * 2 * self.window_trip),
        )
                        
        #self.out = nn.Linear(self.hidden_size * 2, self._n_items)
        self.sf  = nn.Softmax()
        self.att_1 = Attention(n_factors)
        self.att_2 = Attention(n_factors)
        self.b   = nn.Linear(self.n_factors, 
                             self.hidden_size * 2 * self.window_trip, 
                             bias=False)

        self.weight_init = lecun_normal_init
        #self.apply(self.init_weights)

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
    def index_mapping_max_value(self, key: str) -> int:
        return max(self._index_mapping[key].values())+1

    def forward(self, session_ids, 
                        item_ids, 
                        item_history_ids, 
                        affiliate_id_list,
                        device_list,
                        user_features,
                        booker_country_list,
                        duration_list, 
                        start_trip_month, 
                        dense_features):

        embs         = self.emb_dropout(self.emb_item(item_history_ids))
        embs_country = self.emb_dropout(self.emb_country(booker_country_list))

        # # Add Month
        embs_month = self.emb_month(start_trip_month.long())
        embs_month = embs_month.unsqueeze(1).expand(item_history_ids.shape[0], item_history_ids.shape[1], self.n_month_dim)
        #embs    = torch.cat([embs, embs_month], 2)      


        # Add Time
        embs_time   = self.emb_time(duration_list.float().unsqueeze(2))  # (B, H, E)
        embs        = (embs + embs_time)/2

        embs, att_w = self.att_1(embs, embs)
        output, hidden = self.gru1(embs)
        out         = output.contiguous().view(-1, self.hidden_size * 2 *self.window_trip) #output[:,-1] 
        

        # embs_2, att_w_2 = self.att_2(embs_country, embs_country)
        # output_2, hidden_2 = self.gru2(embs_2)
        # out_2       = output_2[:,-1] 
        
        # out       = self.mlp_dense(torch.cat([out, out_2], 1))
        #output.contiguous().view(-1, self.hidden_size * 2)).view(output.size())

        item_embs = self.emb_item(torch.arange(self._n_items).to(item_history_ids.device).long())
        scores    = torch.matmul(out, self.b(item_embs).permute(1, 0))

        return scores

    def recommendation_score(self, session_ids, 
                        item_ids, 
                        item_history_ids, 
                        affiliate_id_list,
                        device_list,
                        user_features,
                        booker_country_list,
                        duration_list, 
                        start_trip_month, 
                        dense_features):
        
        scores = self.forward(session_ids, 
                        item_ids, 
                        item_history_ids, 
                        affiliate_id_list,
                        device_list,
                        user_features,
                        booker_country_list,
                        duration_list, 
                        start_trip_month, 
                        dense_features)
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
        self.n_h = p_nh
        self.n_v = p_nv
        self.drop_ratio = dropout
        self.ac_conv = F.relu# .relu#activation_getter[p_ac_conv]
        self.ac_fc = F.relu#activation_getter[p_ac_fc]
        num_items = self._n_items
        self.n_time_dim  = 100
        self.n_month_dim = 10

        n_booker_country_list_dim   = self.index_mapping_max_value('booker_country_list')
        n_affiliate_id_list_dim    = self.index_mapping_max_value('affiliate_id_list')
        n_device_list_dim         = self.index_mapping_max_value('device_class_list')

        # user and item embeddings
        self.user_embeddings = nn.Embedding(self._n_users, n_factors)
        self.item_embeddings = load_embedding(self._n_items, n_factors, path_item_embedding, 
                                                from_index_mapping, index_mapping, freeze_embedding)
        self.emb_drop = nn.Dropout(p=dropout)

        self.emb_time    = TimeEmbedding(self.n_time_dim, n_factors)
        self.month_emb   = nn.Embedding(13, self.n_month_dim)
        self.emb_country = nn.Embedding(n_booker_country_list_dim, n_factors)
        self.emb_affiliate = nn.Embedding(n_affiliate_id_list_dim, n_factors)
        self.emb_device  = nn.Embedding(n_device_list_dim, 10)        

        self.input_rnn_dim = n_factors * 3 + self.n_month_dim 
        self.att           = Attention(self.input_rnn_dim)

        # vertical conv layer
        self.conv_v = nn.Conv2d(1, self.n_v, (L, 1))
        
        # horizontal conv layer
        lengths = [i + 1 for i in range(L)]
        self.conv_h = nn.ModuleList([nn.Conv2d(1, self.n_h, (i, self.input_rnn_dim)) for i in lengths])

        # fully-connected layer
        self.fc1_dim_v = self.n_v * self.input_rnn_dim
        self.fc1_dim_h = self.n_h * len(lengths)

        self.hidden_size = 300 
        self.n_layers    = 1
        self.gru1 = nn.GRU(self.input_rnn_dim, self.hidden_size, self.n_layers, 
                            dropout=dropout, bidirectional=True)
        
        fc1_dim_in     = self.fc1_dim_v + self.fc1_dim_h 

        # W1, b1 can be encoded with nn.Linear
        #self.fc1 = nn.Linear(fc1_dim_in, self.input_rnn_dim)


        self.activate_func = nn.ReLU()
        self.fc1 = nn.Sequential(
            nn.BatchNorm1d(fc1_dim_in),
            nn.Linear(fc1_dim_in, n_factors),
            self.activate_func,
            #nn.Linear(n_factors, n_factors)
        )
        # W2, b2 are encoded with nn.Embedding, as we don't need to compute scores for all items
        #self.W2 = nn.Embedding(num_items, n_factors) #+dims
        #self.b2 = nn.Embedding(num_items, 1)

        # dropout
        self.dropout = nn.Dropout(self.drop_ratio)

        # weight initialization
        self.user_embeddings.weight.data.normal_(0, 1.0 / self.user_embeddings.embedding_dim)
        self.item_embeddings.weight.data.normal_(0, 1.0 / self.item_embeddings.embedding_dim)
        #self.W2.weight.data.normal_(0, 1.0 / self.W2.embedding_dim)
        #self.b2.weight.data.zero_()
        
        self.b   = nn.Linear(n_factors, n_factors + self.hidden_size * 2)
        self.out = nn.Linear(n_factors, self._n_items)

        #self.cache_x = None
    def index_mapping_max_value(self, key: str) -> int:
        return max(self._index_mapping[key].values())+1
        

    def forward(self, session_ids, 
                        item_ids, 
                        item_history_ids, 
                        affiliate_id_list,
                        device_list,
                        user_features,
                        booker_country_list,
                        duration_list, 
                        start_trip_month, 
                        dense_features): # for training        
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
        item_embs = self.emb_drop(self.item_embeddings(item_history_ids))  # use unsqueeze() to get 4-D
        user_emb = self.emb_drop(self.user_embeddings(session_ids))#.squeeze(1)

        affiliate_emb = self.emb_drop(self.emb_affiliate(affiliate_id_list))
        #device_emb = self.emb_drop(self.emb_device(seq_device))
        country_embs = self.emb_drop(self.emb_country(booker_country_list))

        # Time/Month Embs
        m_emb   = self.emb_drop(self.month_emb(start_trip_month.long()))
        m_embs  = m_emb.unsqueeze(1).expand(item_history_ids.shape[0], item_history_ids.shape[1], self.n_month_dim)

        # Add Time
        t_embs  = self.emb_time(duration_list.float().unsqueeze(2)) # (B, H, E)
        item_embs    = (item_embs + t_embs)/2

        item_embs    = torch.cat([item_embs, affiliate_emb, country_embs, m_embs], 2)
        item_embs, _w    = self.att(item_embs, item_embs)

        item_embs2 = item_embs.unsqueeze(1)

        # Convolutional Layers
        out, out_h, out_v = None, None, None
        # vertical conv layer
        if self.n_v:
            out_v = self.conv_v(item_embs2)
            out_v = out_v.view(-1, self.fc1_dim_v)  # prepare for fully connect

        # horizontal conv layer
        out_hs = list()
        if self.n_h:
            for conv in self.conv_h:
                conv_out = self.ac_conv(conv(item_embs2).squeeze(3))
                pool_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
                out_hs.append(pool_out)
            out_h = torch.cat(out_hs, 1)  # prepare for fully connect

        output, hidden = self.gru1(item_embs)
        out_rnn        = output[:,-1] 

        # Fully-connected Layers
        out = torch.cat([out_v, out_h], 1)

        # apply dropout
        out = self.dropout(out)

        # fully-connected layer
        z = self.ac_fc(self.fc1(out))
        
        y = torch.cat([z, out_rnn], 1)

        item_embs   = self.item_embeddings(torch.arange(self._n_items).to(item_history_ids.device).long()) # (Dim, E)
        scores      = torch.matmul(y, self.b(item_embs).permute(1, 0)) # (B, dim)

        return scores

    def recommendation_score(self, session_ids, 
                        item_ids, 
                        item_history_ids, 
                        affiliate_id_list,
                        device_list,
                        user_features,
                        booker_country_list,
                        duration_list, 
                        start_trip_month, 
                        dense_features):
        
        scores = self.forward(session_ids, 
                        item_ids, 
                        item_history_ids, 
                        user_features,
                        hotel_country_list,
                        duration_list, 
                        trip_month, 
                        dense_features)
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
        
        n_booker_country_list_dim   = self.index_mapping_max_value('booker_country_list')
        n_affiliate_id_list_dim    = self.index_mapping_max_value('affiliate_id_list')
        n_device_list_dim         = self.index_mapping_max_value('device_class_list')

        self.emb_user    = nn.Embedding(self._n_users, n_factors)
        self.emb         = load_embedding(self._n_items, n_factors, path_item_embedding, 
                                        from_index_mapping, index_mapping, freeze_embedding)
        self.time_emb    = TimeEmbedding(self.n_time_dim, n_factors)
        self.month_emb   = nn.Embedding(13, self.n_month_dim)
        self.emb_country = nn.Embedding(n_booker_country_list_dim, n_factors)
        self.emb_affiliate = nn.Embedding(n_affiliate_id_list_dim, n_factors)
        self.emb_device  = nn.Embedding(n_device_list_dim, 10)        

        self.emb_dropout = nn.Dropout(dropout)
        self.input_rnn_dim = self.embedding_dim * 3 + self.n_month_dim 

        self.gru = nn.GRU(self.input_rnn_dim, 
                            self.hidden_size, self.n_layers, bidirectional=True)

        self.a_1 = nn.Linear(self.hidden_size * 2, self.hidden_size * 2, bias=False)
        self.a_2 = nn.Linear(self.hidden_size, self.hidden_size * 2, bias=False)
        self.v_t = nn.Linear(self.hidden_size * 2, 1, bias=False)

        self.activate_func = nn.SELU()
        n_dense_features  = 0
        output_dense_size = self.hidden_size * 3 + n_dense_features 

        # self.mlp_dense = nn.Sequential(
        #     nn.Linear(10, n_dense_features),
        #     self.activate_func,
        #     nn.Linear(n_dense_features, n_factors),
        # )

        self.att = Attention(self.input_rnn_dim)
        # self.mlp_emb_features = nn.Sequential(
        #     nn.Linear(1 * n_factors + self.n_month_dim, n_factors),
        #     self.activate_func,
        #     nn.Linear(n_factors, n_factors),
        # )

        self.ct_dropout = nn.Dropout(dropout)
        self.b = nn.Linear(self.embedding_dim, output_dense_size, bias=False)
        
    def index_mapping_max_value(self, key: str) -> int:
        return max(self._index_mapping[key].values())+1
        
    def flatten(self, input):
        return input.view(input.size(0), -1)

    def normalize(self, x: torch.Tensor, dim: int = 1) -> torch.Tensor:
        if self.use_normalize:
            x = F.normalize(x, p=2, dim=dim)
        return x

    def forward(self,   session_ids, 
                        item_ids, 
                        item_history_ids, 
                        affiliate_id_list,
                        device_list,
                        user_features,
                        booker_country_list,
                        duration_list, 
                        start_trip_month, 
                        dense_features):

        device = item_ids.device
        seq    = item_history_ids.permute(1,0)  
        seq_country = booker_country_list.permute(1,0)
        seq_affiliate = affiliate_id_list.permute(1,0)
        seq_device = device_list.permute(1,0)
        seq_user    = session_ids#.permute(1,0)

        hidden      = self.init_hidden(seq.size(1)).to(device)
        embs        = self.emb_dropout(self.emb(seq))
        affiliate_emb = self.emb_dropout(self.emb_affiliate(seq_affiliate))
        user_emb    = self.emb_dropout(self.emb_user(seq_user))
        device_emb = self.emb_dropout(self.emb_device(seq_device))
        country_embs = self.emb_dropout(self.emb_country(seq_country))

        # e_features  = self.mlp_emb_features(torch.cat([emb_first_hotel_country, 
        #d_features = self.mlp_dense(user_features.float())

        #emb_first_hotel_country = self.emb_country(first_hotel_country)

        # Time/Month Embs
        m_emb   = self.emb_dropout(self.month_emb(start_trip_month.long()))
        m_embs  = m_emb.unsqueeze(0).expand(seq.shape[0], seq.shape[1], self.n_month_dim)

        #m_embs2 = dense_features.float().unsqueeze(0).expand(seq.shape[0], seq.shape[1], self.n_month_dim)

        # Add Time
        t_embs  = self.time_emb(duration_list.float().unsqueeze(2)).permute(1,0,2)  # (H, B, E)
        embs    = (embs + t_embs)/2

        # att
        #_emb, _w    = self.att(embs.permute(1,0,2), m_embs.permute(1,0,2)).permute(1,0,2)
        #embs        = _emb.permute(1,0,2)
        # Join
        embs        = torch.cat([embs, affiliate_emb, country_embs, m_embs], 2)      
        _emb, _w    = self.att(embs.permute(1,0,2), embs.permute(1,0,2))
        embs        = _emb.permute(1,0,2)
                
        gru_out, hidden = self.gru(embs, hidden)

        # fetch the last hidden state of last timestamp
        ht      = hidden[-1]
        gru_out = gru_out.permute(1, 0, 2)

        c_global = ht
        q1 = self.a_1(gru_out.contiguous().view(-1, self.hidden_size * 2)).view(gru_out.size())  
        q2 = self.a_2(ht)

        mask      = torch.where(seq.permute(1, 0) > IDX_FIX, torch.tensor([1.], device = device), 
                        torch.tensor([0.], device = device))

        q2_expand = q2.unsqueeze(1).expand_as(q1)
        q2_masked = mask.unsqueeze(2).expand_as(q1) * q2_expand

        alpha   = self.v_t(torch.sigmoid(q1 + q2_masked)\
                    .view(-1, self.hidden_size * 2))\
                    .view(mask.size())
        c_local = torch.sum(alpha.unsqueeze(2).expand_as(gru_out) * gru_out, 1)


        c_t     = torch.cat([c_local, c_global], 1) #, dense_features.float()
        c_t     = self.ct_dropout(c_t)
        
        item_embs = self.emb(torch.arange(self._n_items).to(device).long())
        scores    = torch.matmul(c_t, self.b(item_embs).permute(1, 0))

        return scores

    def recommendation_score(self, session_ids, 
                                    item_ids, 
                                    item_history_ids, 
                                    affiliate_id_list,
                                    device_list,
                                    user_features,
                                    booker_country_list,
                                    duration_list, 
                                    start_trip_month, 
                                    dense_features):
        
        scores = self.forward(session_ids, 
                                item_ids, 
                                item_history_ids, 
                                affiliate_id_list,
                                device_list,
                                user_features,
                                booker_country_list,
                                duration_list, 
                                start_trip_month, 
                                dense_features)
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