import luigi
import pandas as pd
import numpy as np
import os
import pickle

from mars_gym.data.task import BasePrepareDataFrames, BasePySparkTask
from mars_gym.data.dataset import (
    InteractionsDataset,
    InteractionsWithNegativeItemGenerationDataset,
)
import random
from typing import Tuple, List, Union, Callable, Optional, Set, Dict, Any
from mars_gym.meta_config import *
from itertools import chain
from datetime import datetime, timedelta
from tqdm import tqdm
tqdm.pandas()


OUTPUT_PATH: str = os.path.join("output")
BASE_DIR: str = os.path.join(OUTPUT_PATH, "booking")
DATASET_DIR: str = os.path.join(OUTPUT_PATH, "booking", "dataset")

BASE_DATASET_FILE : str = os.path.join("data", "Booking", "booking_train_set.csv")

class SplitTrainTestDataset(luigi.Task):
  sample_days: int = luigi.IntParameter(default=30)
  test_days: int = luigi.IntParameter(default=7)
  window_trip: int = luigi.IntParameter(default=5)

  # def requires(self):

  def output(self):
      return luigi.LocalTarget(os.path.join(DATASET_DIR, "train_{}_{}_{}.csv"\
                  .format(self.sample_days, self.test_days, self.window_trip),)),\
              luigi.LocalTarget(os.path.join(DATASET_DIR, "test_{}_{}_{}.csv"\
                  .format(self.sample_days, self.test_days, self.window_trip),))                    

  def run(self):
    os.makedirs(DATASET_DIR, exist_ok=True)

    df = pd.read_csv(BASE_DATASET_FILE, 
                    dtype={"user_id": str, "city_id": str, 'affiliate_id': str,
                          'utrip_id': str}, date_parser=['checkin', 'checkin'])

    df['checkin']  = pd.to_datetime(df['checkin'])
    df['checkout'] = pd.to_datetime(df['checkout'])
    df['duration'] = (df['checkout'] - df['checkin']).dt.days
    df['checkin_str']   = df['checkin'].astype(str)
    df['checkout_str']  = df['checkout'].astype(str)
    df.head()

    def count_hotel(hotel_country):
        return len(list(np.unique(hotel_country)))

    def list_without_last(itens):
        return list(itens[:-1])

    def list_without_last_and_pad(pad=5, dtype=int):
        def add_pad(items): 
            arr = list_without_last(items)
            arr = list(([dtype(0)] * (pad - len(arr[-pad:])) + arr[-pad:])) 
            
            return arr
            
        return add_pad

    df_trip = df.sort_values(['checkin']).groupby(['utrip_id']).agg(
        user_id=('user_id', 'first'),
        count_unique_city=('city_id', count_hotel),
        trip_size=('checkin', len),
        start_trip=('checkin', 'first'),
        end_trip=('checkin', 'last'),
        checkin_list=('checkin_str', list_without_last_and_pad(self.window_trip, str)),
        checkout_list=('checkout_str', list_without_last_and_pad(self.window_trip, str)),
        duration_list=('duration', list_without_last_and_pad(self.window_trip, int)),
        city_id_list=('city_id', list_without_last_and_pad(self.window_trip, str)),
        device_class_list=('device_class', list_without_last_and_pad(self.window_trip, str)),
        affiliate_id_list=('affiliate_id', list_without_last_and_pad(self.window_trip, str)),
        booker_country_list=('booker_country', list_without_last_and_pad(self.window_trip, str)),
        hotel_country_list=('hotel_country', list_without_last_and_pad(self.window_trip, str)),
        last_city_id=('city_id', 'last')
    )

    #df_trip['end_trip']  = df_trip['checkout_list'].apply(lambda x: x[-1] if len(x) > 1 else None)
    df_trip['duration']  = (df_trip['end_trip'] - df_trip['start_trip']).dt.days

    # Fix results withou last interaction
    df_trip['trip_size'] = df_trip['trip_size'] -1
    df_trip['duration']  = df_trip['duration_list'].apply(sum)

    # Split Data
    max_timestamp = df_trip.start_trip.max()
    init_train_timestamp = max_timestamp - timedelta(days = self.sample_days)
    init_test_timestamp  = max_timestamp - timedelta(days = self.test_days)


    df_train = df_trip[(df_trip.start_trip > init_train_timestamp) & (df_trip.start_trip <= init_test_timestamp)]
    df_test  = df_trip[df_trip.start_trip > init_test_timestamp]    

    df_train.to_csv(self.output()[0].path)
    df_test.to_csv(self.output()[1].path)

class SessionInteractionDataFrame(BasePrepareDataFrames):
    sample_days: int = luigi.IntParameter(default=30)
    test_days: int = luigi.IntParameter(default=7)
    window_trip: int = luigi.IntParameter(default=5)
    item_column: str = luigi.Parameter(default="last_city_id")

    def requires(self):
        return SplitTrainTestDataset(sample_days=self.sample_days, 
                                    test_days=self.test_days, 
                                    window_trip=self.window_trip)

    @property
    def timestamp_property(self) -> str:
        return "start_trip"

    # @property
    # def stratification_property(self) -> str:
    #     return "ItemID"

    # @property
    # def item_property(self) -> str:
    #     return "last_city_id"

    @property
    def dataset_dir(self) -> str:
        return DATASET_DIR

    @property
    def read_data_frame_path(self) -> pd.DataFrame:
        return self.input()[0].path

    # @property
    # def task_name(self):
    #     return self.task_id.split("_")[-1]

    # @property
    # def scaler_file_path(self):
    #     if self.normalize_file_path != None:
    #         return DATASET_DIR+'/'+self.normalize_file_path
    #     return DATASET_DIR+'/{}_std_scaler.pkl'.format(self.task_name)

    def transform_data_frame(self, df: pd.DataFrame, data_key: str) -> pd.DataFrame:

        return df
    
    # def sample_balance_df(self, df, n_samples, state=42):
    #     #col = 'ItemID'
    #     df['sample_weights'] = 1/df['domain_count']
    #     return df.sample(n_samples, weights='sample_weights', random_state=state)