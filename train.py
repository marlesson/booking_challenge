
from mars_gym.simulation.training import SupervisedModelTraining, DummyTraining
#from loss import RelativeTripletLoss, ContrastiveLoss, CustomCrossEntropyLoss
import torch
import torch.nn as nn
import luigi
import numpy as np
from typing import Type, Dict, List, Optional, Tuple, Union, Any, cast
from mars_gym.utils.files import (
    get_index_mapping_path,
)
import pickle
from tqdm import tqdm

from sklearn import manifold
from time import time
import os
import pandas as pd
from mars_gym.model.agent import BanditAgent
from torch.utils.data.dataset import Dataset, ChainDataset
import numpy as np
import itertools
from scipy.sparse import csr_matrix
from scipy.sparse import csr_matrix
from pandas.api.types import CategoricalDtype
from sklearn.metrics.pairwise import cosine_similarity
#from plot import plot_tsne
from mars_gym.data.dataset import (
    preprocess_interactions_data_frame,
    preprocess_metadata_data_frame,
    literal_eval_array_columns,
    InteractionsDataset,
)
from mars_gym.utils.index_mapping import (
    transform_with_indexing,
)

TORCH_LOSS_FUNCTIONS = dict(
    mse=nn.MSELoss,
    nll=nn.NLLLoss,
    bce=nn.BCELoss,
    ce=nn.CrossEntropyLoss,
    #custom_ce=CustomCrossEntropyLoss,
    mlm=nn.MultiLabelMarginLoss,
    #relative_triplet=RelativeTripletLoss,    
    #contrastive_loss=ContrastiveLoss,
)



class CoOccurrenceTraining(DummyTraining):
    '''
    Most Popular Model
    '''
    def fit(self, df_train: pd.DataFrame):
        print("fit...")
        
        item_idx = np.unique(df_train.last_city_id.values)
        lists    = list(df_train.city_id_list)
        cooc_matrix, to_id = self.create_co_occurences_matrix(item_idx, lists)

        self.columns_coocc = to_id
        self.cooc_matrix   = cooc_matrix


    def create_co_occurences_matrix(self, allowed_words, documents):
        word_to_id = dict(zip(allowed_words, range(len(allowed_words))))
        documents_as_ids = [np.sort([word_to_id[w] for w in doc if w in word_to_id]).astype('uint32') for doc in documents]
        row_ind, col_ind = zip(*itertools.chain(*[[(i, w) for w in doc] for i, doc in enumerate(documents_as_ids)]))
        data = np.ones(len(row_ind), dtype='uint32')  # use unsigned int for better memory utilization
        max_word_id = max(itertools.chain(*documents_as_ids)) + 1
        docs_words_matrix = csr_matrix((data, (row_ind, col_ind)), shape=(len(documents_as_ids), max_word_id))  # efficient arithmetic operations with CSR * CSR
        words_cooc_matrix = docs_words_matrix.T * docs_words_matrix  # multiplying docs_words_matrix with its transpose matrix would generate the co-occurences matrix
        words_cooc_matrix.setdiag(0)

        return words_cooc_matrix, word_to_id 
        
    def get_scores(self, agent: BanditAgent, ob_dataset: Dataset) -> List[float]:
        print("get_scores...")
        #

        last_items = list(ob_dataset._data_frame.city_id_list.apply(lambda l: l[0]))
        next_items = list(ob_dataset._data_frame.last_city_id.values)

        scores = []
        for last_item, next_item in tqdm(zip(last_items, next_items), total=len(last_items)):
            scores.append(self.get_score(last_item, next_item))

        return scores

    def get_score(self, item_a: int, item_b: int):
        try:
            item_a_idx = self.columns_coocc[item_a]
            item_b_idx = self.columns_coocc[item_b]

            return self.cooc_matrix[item_a_idx, item_b_idx]
        except:
            return 0

    # def run_evaluate_task(self) -> None:
    #     os.system(
    #         "PYTHONPATH=. luigi --module mars_gym.evaluation.task EvaluateTestSetPredictions "
    #         f"--model-task-class train.CoOccurrenceTraining --model-task-id {self.task_id} --only-new-interactions --only-exist-items --local-scheduler"
    #     )     