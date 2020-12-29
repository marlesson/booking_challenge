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
    
# mars-gym run supervised \
# --project config.base_rnn \
# --recommender-module-class model.GRURecModel \
# --recommender-extra-params '{
#   "n_factors": 100, 
#   "hidden_size": 100, 
#   "n_layers": 1, 
#   "dropout": 0.2, 
#   "from_index_mapping": false,
#   "path_item_embedding": false, 
#   "freeze_embedding": false}' \
# --data-frames-preparation-extra-params '{
#   "sample_days": 30, 
#   "test_days": 7,
#   "window_trip": 5,
#   "column_stratification": "user_id"}' \
# --early-stopping-min-delta 0.0001 \
# --learning-rate 0.001 \
# --metrics='["loss"]' \
# --batch-size 128 \
# --loss-function ce \
# --epochs 100

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