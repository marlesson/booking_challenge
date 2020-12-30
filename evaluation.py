from mars_gym.evaluation.policy_estimator import PolicyEstimatorTraining
from mars_gym.torch.data import FasterBatchSampler, NoAutoCollationDataLoader
from mars_gym.utils.reflection import load_attr, get_attribute_names
from mars_gym.utils.utils import parallel_literal_eval, JsonEncoder
from mars_gym.utils.index_mapping import (
    create_index_mapping,
    create_index_mapping_from_arrays,
    transform_with_indexing,
    map_array,
)
import functools
from multiprocessing.pool import Pool
from mars_gym.evaluation.task import BaseEvaluationTask
import abc
from typing import Type, Dict, List, Optional, Tuple, Union, Any, cast
from torch.utils.data import DataLoader
from mars_gym.torch.data import NoAutoCollationDataLoader, FasterBatchSampler
from torchbearer import Trial
from data import SessionInteractionDataFrame
import gc
import luigi
import pandas as pd
import numpy as np
import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from mars_gym.cuda import CudaRepository
import torchbearer
from tqdm import tqdm
from mars_gym.data.dataset import (
    preprocess_interactions_data_frame,
    InteractionsDataset,
)
from mars_gym.utils.index_mapping import (
    transform_with_indexing,
)
from mars_gym.data.dataset import (
    preprocess_interactions_data_frame,
    InteractionsDataset,
)
from mars_gym.evaluation.metrics.rank import (
    mean_reciprocal_rank,
    average_precision,
    precision_at_k,
    ndcg_at_k,
    personalization_at_k,
    prediction_coverage_at_k,
)
from mars_gym.utils.utils import parallel_literal_eval, JsonEncoder
import pprint
import json
import luigi
import pandas as pd
import functools
import numpy as np
from tqdm import tqdm
import os
from multiprocessing.pool import Pool
from scipy import stats
#from train import MostPopularTraining, CoOccurrenceTraining

def acc(r, k =4):
    r = r[:k]
    return np.sum(r)

def _sort_rank_list(score, index_mapping):
    # UNK, PAD, PAD
    score[0] = score[1] = score[2] = 0

    item_idx  = np.argsort(score)[::-1][:100]
    
    item_id   = [int(index_mapping[item]) for item in item_idx]
    
    return item_id

def _get_moda(arr):
    try:
        return stats.mode(arr)[0][0]
    except:
        return 0 

def _get_count_moda(arr):
    try:
        return stats.mode(arr)[1][0]/len(arr)
    except:
        return 0     

def _create_relevance_list(sorted_actions, expected_action):
    return [1 if str(action) == str(expected_action) else 0 for action in sorted_actions]


# PYTHONPATH="." luigi --module mercado_livre.evaluation MLEvaluationTask \
# --model-task-class "mars_gym.simulation.training.SupervisedModelTraining" \
# --model-task-id SupervisedModelTraining____mars_gym_model_b____e3ae64b091 \
# --normalize-file-path "226cbf7ae2_std_scaler.pkl" \
# --history-window 20 \
# --batch-size 1000 \
# --local-scheduler \
# --file "/media/workspace/triplet_session/output/mercado_livre/dataset/test_0.10_test=random_42_SessionInteractionDataFrame_____SessionID_226cbf7ae2.csv"

class EvaluationTask(BaseEvaluationTask):
    model_task_class: str = luigi.Parameter(
        default="mars_gym.simulation.training.SupervisedModelTraining"
    )
    model_task_id: str = luigi.Parameter()
    offpolicy_eval: bool = luigi.BoolParameter(default=False)
    task_hash: str = luigi.Parameter(default="sub")
    generator_workers: int = luigi.IntParameter(default=0)
    pin_memory: bool = luigi.BoolParameter(default=False)
    batch_size: int = luigi.IntParameter(default=100)
    device: str = luigi.ChoiceParameter(choices=["cpu", "cuda"], default="cuda")
    normalize_dense_features: int = luigi.Parameter(default="min_max")
    normalize_file_path: str = luigi.Parameter(default=None)
    file: str = luigi.Parameter(default="")
    model_eval: str = luigi.ChoiceParameter(choices=["model", "most_popular", "coocorrence"], default="model")

    sample_size: int = luigi.Parameter(default=1000)
    percent_limit: float = luigi.FloatParameter(default=0.2)
    submission_size: int =  luigi.IntParameter(default=10)

    @property
    def task_name(self):
        return self.model_task_id + "_" + self.task_id.split("_")[-1]

    # def requires(self):
    #     if self.local:
    #         return SessionInteractionDataFrame(history_window=self.history_window)
    #     else:
    #         return SplitTrainTestDataset(sample_days=self.history_window)
    # sample_days: int = luigi.IntParameter(default=30)
    # test_days: int = luigi.IntParameter(default=7)
    # window_trip: int = luigi.IntParameter(default=5)

    @property
    def torch_device(self) -> torch.device:
        if not hasattr(self, "_torch_device"):
            if self.device == "cuda":
                self._torch_device = torch.device(f"cuda:{self.device_id}")
            else:
                self._torch_device = torch.device("cpu")
        return self._torch_device

    @property
    def device_id(self):
        if not hasattr(self, "_device_id"):
            if self.device == "cuda":
                self._device_id = CudaRepository.get_avaliable_device()
            else:
                self._device_id = None
        return self._device_id

    def get_test_generator(self, df) -> Optional[DataLoader]:

        dataset = InteractionsDataset(
            data_frame=df,
            embeddings_for_metadata=self.model_training.embeddings_for_metadata,
            project_config=self.model_training.project_config,
            index_mapping=self.model_training.index_mapping
        )

        batch_sampler = FasterBatchSampler(
            dataset, self.batch_size, shuffle=False
        )

        return NoAutoCollationDataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=self.generator_workers,
            pin_memory=self.pin_memory if self.device == "cuda" else False,
        )

    def pos_process(self, rank_list):
        df_item     = pd.read_csv(ITEM_META_PATH, usecols=["item_id", "domain_id", "domain_idx"]).fillna(0)#.head(10)
        domain_map  = df_item[['item_id', 'domain_idx']].set_index("item_id").to_dict()["domain_idx"]

        with Pool(os.cpu_count()) as p:
            _map_domain = list(tqdm(
                p.map(functools.partial(_get_domain, domain_map=domain_map), rank_list),
                total=len(rank_list),
            ))  
            
        arr_moda = list(zip(list(rank_list), 
                            list(_map_domain),  
                            list(map(_get_moda, _map_domain)), 
                            list(map(_get_count_moda, _map_domain))))

        df_moda = pd.DataFrame(arr_moda, columns=["reclist", "domainlist", "domain_moda", "count"])

        df_moda['relevance_list'] = df_moda.apply(lambda row: 
                                                _create_relevance_list_domain(row['domainlist'], row['domain_moda']),  
                                                axis=1)

        df_moda['reclist_2'] = df_moda.apply(lambda row: _sorte_by_domain_moda(
                        row['reclist'], 
                        row['relevance_list'], 
                        row['count'], self.percent_limit)[:self.submission_size],  axis=1)
        

        df_moda['domainlist_2'] = df_moda.apply(lambda row: _sorte_by_domain_moda(
                        row['domainlist'], 
                        row['relevance_list'], 
                        row['count'], self.percent_limit)[:self.submission_size],  axis=1)



        return df_moda

    def run(self):
        os.makedirs(self.output().path)
    
        df: pd.DataFrame = pd.read_csv(self.file)
        target = 'last_city_id'
        
        if target in df.columns:
          df_metric = df[['utrip_id', target]]

        df = preprocess_interactions_data_frame(
            df, 
            self.model_training.project_config
        )

        data = SessionInteractionDataFrame()
                # item_column="",
                # normalize_dense_features=self.normalize_dense_features,
                # normalize_file_path=self.normalize_file_path

        data.transform_data_frame(df, "TEST_GENERATOR")

        df.to_csv(self.output().path+"/dataset.csv")

        transform_with_indexing(
            df, 
            self.model_training.index_mapping, 
            self.model_training.project_config
        )
        # 
        df.to_csv(self.output().path+"/dataset_indexed.csv")
        generator = self.get_test_generator(df)

        print(df.head())
        print(df.shape)

        reverse_index_mapping    = self.model_training.reverse_index_mapping[target]
        reverse_index_mapping[1] = 0
        
        if self.model_eval == "model":
            rank_list = self.model_rank_list(generator, reverse_index_mapping)
        # elif self.model_eval == "most_popular":
        #     rank_list = self.most_popular_rank_list(generator, reverse_index_mapping)
        # elif self.model_eval == "coocorrence":
        #     rank_list = self.coocorrence_rank_list(generator, reverse_index_mapping)

        # has target column - Metric
        if target in df.columns:
          #from IPython import embed; embed()
          df_metric['reclist']  = list(rank_list)
          df_metric['acc@4']    = df_metric.apply(lambda row: row[target] in row.reclist[:4], axis=1)
          
          metric = {
            'task_name': self.task_name,
            'count': len(df_metric),
            'acc@4': df_metric['acc@4'].mean()
          }

          with open(
              os.path.join(self.output().path, "metric.json"), "w"
          ) as params_file:
              json.dump(metric, params_file, default=lambda o: dict(o), indent=4)
          print(json.dumps(metric, indent=4))

          df_metric.to_csv(self.output().path+'/metric.csv', index=False)

        np.savetxt(self.output().path+'/submission_{}.csv'.format(self.task_name), rank_list, fmt='%i', delimiter=',') 

    def most_popular_rank_list(self, generator, reverse_index_mapping):
        

        most_popular = MostPopularTraining(project="mercado_livre.config.mercado_livre_interaction",
                                          data_frames_preparation_extra_params=self.model_training.data_frames_preparation_extra_params,
                                          test_size=self.model_training.test_size,
                                          val_size=self.model_training.val_size,
                                          test_split_type=self.model_training.test_split_type,
                                          dataset_split_method=self.model_training.dataset_split_method)  #
        most_popular.fit(self.model_training.train_data_frame)

        scores    = []
        rank_list = []

        # Inference
        for i, (x, _) in tqdm(enumerate(generator), total=len(generator)):
            input_params = x if isinstance(x, list) or isinstance(x, tuple) else [x]

            scores_batch = [] # nxk
            item_idx = list(range(self.model_training.n_items))

            for i in input_params[1]:
                score = list(most_popular.item_counts.loc[item_idx].fillna(0).values)
                scores_batch.append(score)

            # Test
            _sort_rank_list(scores_batch[0], index_mapping=reverse_index_mapping)

            with Pool(3) as p:
                _rank_list = list(tqdm(
                    p.map(functools.partial(_sort_rank_list, index_mapping=reverse_index_mapping), scores_batch),
                    total=len(scores_batch),
                ))
                rank_list.extend(_rank_list)

            gc.collect()
    
        return rank_list

    def coocorrence_rank_list(self, generator, reverse_index_mapping):
        

        cooccurrence = CoOccurrenceTraining(project="mercado_livre.config.mercado_livre_interaction",
                                          data_frames_preparation_extra_params=self.model_training.data_frames_preparation_extra_params,
                                          test_size=self.model_training.test_size,
                                          val_size=self.model_training.val_size,
                                          test_split_type=self.model_training.test_split_type,
                                          dataset_split_method=self.model_training.dataset_split_method)  #
        cooccurrence.fit(self.model_training.train_data_frame)

        scores    = []
        rank_list = []

        # Inference
        for i, (x, _) in tqdm(enumerate(generator), total=len(generator)):
            input_params = x if isinstance(x, list) or isinstance(x, tuple) else [x]

            scores_batch = [] # nxk
            item_idx = list(range(self.model_training.n_items))
            for item_history in input_params[2]:
                last_item_idx = item_history.detach().cpu().numpy()[0]

                score = [cooccurrence.get_score(last_item_idx, i) for i in  item_idx]
                #from IPython import embed; embed()

                scores_batch.append(score)

            # Test
            _sort_rank_list(scores_batch[0], index_mapping=reverse_index_mapping)

            with Pool(3) as p:
                _rank_list = list(tqdm(
                    p.map(functools.partial(_sort_rank_list, index_mapping=reverse_index_mapping), scores_batch),
                    total=len(scores_batch),
                ))
                rank_list.extend(_rank_list)

            gc.collect()
    
        return rank_list

    def model_rank_list(self, generator, reverse_index_mapping):

        # Gente Model
        model = self.model_training.get_trained_module()
        model.to(self.torch_device)
        model.eval()

        scores = []
        rank_list = []
        # Inference
        with torch.no_grad():
            for i, (x, _) in tqdm(enumerate(generator), total=len(generator)):
                input_params = x if isinstance(x, list) or isinstance(x, tuple) else [x]
                input_params = [t.to(self.torch_device) if isinstance(t, torch.Tensor) else t for t in input_params]

                scores_tensor: torch.Tensor  = model(*input_params)
                scores_batch = scores_tensor.detach().cpu().numpy()
                #scores.extend(scores_batch)

                # Test
                _sort_rank_list(scores_batch[0], index_mapping=reverse_index_mapping)

                with Pool(3) as p:
                    _rank_list = list(tqdm(
                        p.map(functools.partial(_sort_rank_list, index_mapping=reverse_index_mapping), scores_batch),
                        total=len(scores_batch),
                    ))
                    rank_list.extend(_rank_list)

                gc.collect()
        
        return rank_list

# PYTHONPATH="." luigi --module mercado_livre.evaluation EvaluationSubmission \
# --model-task-class "mars_gym.simulation.training.SupervisedModelTraining" \
# --model-task-id SupervisedModelTraining____mars_gym_model_b____e3ae64b091 \
# --normalize-file-path "226cbf7ae2_std_scaler.pkl" \
# --history-window 20 \
# --batch-size 1000 \
# --local-scheduler \
# --file "/media/workspace/triplet_session/output/mercado_livre/dataset/test_0.10_test=random_42_SessionInteractionDataFrame_____SessionID_226cbf7ae2.csv"

class EvaluationSubmission(luigi.Task):
    model_task_class: str = luigi.Parameter(
        default="mars_gym.simulation.training.SupervisedModelTraining"
    )
    model_task_id: str = luigi.Parameter()
    offpolicy_eval: bool = luigi.BoolParameter(default=False)
    task_hash: str = luigi.Parameter(default="sub")
    generator_workers: int = luigi.IntParameter(default=0)
    pin_memory: bool = luigi.BoolParameter(default=False)
    batch_size: int = luigi.IntParameter(default=100)
    device: str = luigi.ChoiceParameter(choices=["cpu", "cuda"], default="cuda")
    history_window: int = luigi.IntParameter(default=30)
    normalize_dense_features: int = luigi.Parameter(default="min_max")
    normalize_file_path: str = luigi.Parameter(default=None)
    local: bool = luigi.BoolParameter(default=False)
    sample_size: int = luigi.Parameter(default=1000)
    percent_limit: float = luigi.FloatParameter(default=0.4)
    model_eval: str = luigi.ChoiceParameter(choices=["model", "most_popular", "coocorrence"], default="model")
    eval_reclist: str = luigi.Parameter(default=None)
    submission_size: int =  luigi.IntParameter(default=10)

    def requires(self):
        return MLEvaluationTask(model_task_class=self.model_task_class,
                                model_task_id=self.model_task_id,
                                normalize_dense_features=self.normalize_dense_features,
                                normalize_file_path=self.normalize_file_path,
                                batch_size=self.batch_size,
                                history_window=self.history_window,
                                local=self.local,
                                model_eval=self.model_eval,
                                sample_size=self.sample_size,
                                percent_limit=self.percent_limit,
                                submission_size=self.submission_size), SessionPrepareLocalTestDataset(history_window=self.history_window)
    

    def output(self):
        if self.eval_reclist:
            return luigi.LocalTarget(os.path.join(self.input()[0].path, self.eval_reclist+"_metrics.json"))
        else:
            return luigi.LocalTarget(os.path.join(self.input()[0].path, "metrics.json"))

    def run(self):
        print("\n==> ", self.input()[0].path, "\n")
        df: pd.DataFrame = pd.read_parquet(self.input()[1][1].path)#.sample(n=self.sample_size, random_state=42, replace=True)#, usecols=['ItemID']).sample(n=self.sample_size, random_state=42, replace=True)
        df_sub: pd.DataFrame = pd.read_csv(self.input()[0].path+'/df_submission.csv')

        if self.eval_reclist is None:
            arr_sub: pd.DataFrame = pd.read_csv(self.input()[0].path+'/submission_{}.csv'.format(self.requires()[0].task_name), header=None)
        else:
            arr_sub: pd.DataFrame = pd.read_csv(self.eval_reclist, header=None)

        if not self.local:
            return
            
        df['reclist']        = list(arr_sub.values)
        df['domainlist']     = list(df_sub.domainlist_2.apply(eval))
        df['relevance_list'] = df.apply(lambda row: _create_relevance_list(row['reclist'], row['ItemID']),  axis=1)
        df['relevance_list_ml'] = df.apply(lambda row: _create_relevance_list_ml(row['reclist'], row['domainlist'], row['ItemID'], row['domain_idx']),  axis=1)


        with Pool(os.cpu_count()) as p:
            print("Calculating average precision...")
            df["average_precision"] = list(
                tqdm(p.map(average_precision, df["relevance_list"]), total=len(df))
            )

            print("Calculating precision at 1...")
            df["precision_at_1"] = list(
                tqdm(
                    p.map(functools.partial(precision_at_k, k=1), df["relevance_list"]),
                    total=len(df),
                )
            )

            print("Calculating MRR at 5 ...")
            df["mrr_at_5"] = list(
                tqdm(
                    p.map(functools.partial(mean_reciprocal_rank, k=5), df["relevance_list"]),
                    total=len(df),
                )
            )


            print("Calculating MRR at 10 ...")
            df["mrr_at_10"] = list(
                tqdm(
                    p.map(functools.partial(mean_reciprocal_rank, k=10), df["relevance_list"]),
                    total=len(df),
                )
            )

            print("Calculating nDCG at 5...")
            df["ndcg_at_5"] = list(
                tqdm(
                    p.map(functools.partial(ndcg_at_k, k=5), df["relevance_list"]),
                    total=len(df),
                )
            )
            print("Calculating nDCG at 10...")
            df["ndcg_at_10"] = list(
                tqdm(
                    p.map(functools.partial(ndcg_at_k, k=10), df["relevance_list"]),
                    total=len(df),
                )
            )
            print("Calculating nDCG at 15...")
            df["ndcg_at_15"] = list(
                tqdm(
                    p.map(functools.partial(ndcg_at_k, k=15), df["relevance_list"]),
                    total=len(df),
                )
            )
            print("Calculating nDCG at 20...")
            df["ndcg_at_20"] = list(
                tqdm(
                    p.map(functools.partial(ndcg_at_k, k=20), df["relevance_list"]),
                    total=len(df),
                )
            )
            print("Calculating nDCG at 50...")
            df["ndcg_at_50"] = list(
                tqdm(
                    p.map(functools.partial(ndcg_at_k, k=50), df["relevance_list"]),
                    total=len(df),
                )
            )

            print("Calculating nDCGML...")
            df["ndcg_ml"] = list(
                tqdm(
                    p.map(functools.partial(ndcg_ml, k=50), df["relevance_list_ml"]),
                    total=len(df),
                )
            )
            

        metrics = {
            "model_task": self.model_task_id,
            "percent_limit": self.percent_limit,
            "count": len(df),
            "mean_average_precision": df["average_precision"].mean(),
            "precision_at_1": df["precision_at_1"].mean(),
            "mrr_at_5": df["mrr_at_5"].mean(),
            "mrr_at_10": df["mrr_at_10"].mean(),
            "ndcg_at_5": df["ndcg_at_5"].mean(),
            "ndcg_at_10": df["ndcg_at_10"].mean(),
            "ndcg_at_15": df["ndcg_at_15"].mean(),
            "ndcg_at_20": df["ndcg_at_20"].mean(),
            "ndcg_at_50": df["ndcg_at_50"].mean(),
            "ndcg_ml": df["ndcg_ml"].mean(),
        }
        pprint.pprint(metrics)
        
        df.to_csv(os.path.join(self.input()[0].path, "eval_dataset.csv"))
        with open(
            os.path.join(self.input()[0].path, "metrics.json"), "w"
        ) as metrics_file:
            json.dump(metrics, metrics_file, cls=JsonEncoder, indent=4)

