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
from sklearn.model_selection import train_test_split

OUTPUT_PATH: str = os.path.join("output")
BASE_DIR: str = os.path.join(OUTPUT_PATH, "booking")
DATASET_DIR: str = os.path.join(OUTPUT_PATH, "booking", "dataset")

BASE_DATASET_FILE : str = os.path.join("data", "Booking", "booking_train_set.csv")
BASE_DATASET_TEST_FILE : str = os.path.join("data", "Booking", "booking_test_set.csv")

LIMIT_DURATION_SUM = 22
LIMIT_TRIP_SIZE    = 10

def count_hotel(hotel_country):
    return len(list(np.unique(hotel_country)))

def list_without_last(itens):
    l = list(itens[:-1])
    l.append("M") #mask
    return l

def list_and_pad(pad=5, dtype=int, ignore_last_value=True):
    def add_pad(items): 
        if ignore_last_value:
            arr = list_without_last(items)
        else:
            arr = list(items)
        arr = list(([dtype(0)] * (pad - len(arr[-pad:])) + arr[-pad:])) 
        
        return arr
        
    return add_pad
    
class SplitAndPreprocessDataset(luigi.Task):
  test_split: float = luigi.IntParameter(default=0.1)
  window_trip: int = luigi.IntParameter(default=5)

  # def requires(self):

  def output(self):
    return luigi.LocalTarget(os.path.join(DATASET_DIR, "train_{}_{}.csv".format(self.test_split, self.window_trip))),\
            luigi.LocalTarget(os.path.join(DATASET_DIR, "test_{}_{}.csv".format(self.test_split, self.window_trip)))                    

  def add_general_features(self, df):

    df['duration']        = (df['checkout'] - df['checkin']).dt.days
    df['checkin_str']     = df['checkin'].astype(str)
    df['checkout_str']    = df['checkout'].astype(str)
    df['checkin_int']     = df['checkin'].astype(int)/10**9/60/60/24
    df['checkout_int']    = df['checkout'].astype(int)/10**9/60/60/24
    df['days_since_2016'] = pd.to_datetime(df['checkin']).sub(pd.Timestamp('2016-01-01 00:00:00')).dt.days
    df['step'] = 1
    df['step'] = df.groupby(['utrip_id']).step.cumsum()

    df['user_trip_count'] = 1
    df['user_trip_count'] = df.groupby(['user_id']).user_trip_count.cumsum()
    df['is_new_user']     = df['user_trip_count'].apply(lambda x: x > 1).astype(int)

  def group_by_trip(self, df):
    df_trip = df.sort_values(['step']).groupby(['utrip_id']).agg(
        user_id=('user_id', 'first'),
        count_unique_city=('city_id', count_hotel),
        trip_size=('checkin', len),
        start_trip=('checkin', 'first'),
        end_trip=('checkin', 'last'),
        checkin_list=('checkin_int', list_and_pad(self.window_trip, int, False)),
        checkout_list=('checkout_int', list_and_pad(self.window_trip, int, False)),
        days_since_2016_list=('days_since_2016', list_and_pad(self.window_trip, int, False)),
        duration_list=('duration', list_and_pad(self.window_trip, int, False)),
        city_id_list=('city_id', list_and_pad(self.window_trip, str)),
        device_class_list=('device_class', list_and_pad(self.window_trip, str, False)),
        affiliate_id_list=('affiliate_id', list_and_pad(self.window_trip, str, False)),
        booker_country_list=('booker_country', list_and_pad(self.window_trip, str, False)),
        hotel_country_list=('hotel_country', list_and_pad(self.window_trip, str)),
        step_list=('step', list_and_pad(self.window_trip, int, False)),

        last_checkin=('checkin_str', 'last'),
        last_checkout=('checkout_str', 'last'),
        last_days_since_2016=('days_since_2016', 'last'),
        last_duration=('duration', 'last'),
        last_device_class=('device_class', 'last'),
        last_affiliate_id=('affiliate_id', 'last'),
        last_booker_country=('booker_country', 'last'),
        last_step=('step', 'last'),

        first_city_id=('city_id', 'first'),
        first_hotel_country=('hotel_country', 'first'),
        last_city_id=('city_id', 'last'),
        last_hotel_country=('hotel_country', 'last'),
        country_count=('country_count', 'min'),
    )

    #df_trip['end_trip']  = df_trip['checkout_list'].apply(lambda x: x[-1] if len(x) > 1 else None)
    #df_trip['duration']  = (df_trip['end_trip'] - df_trip['start_trip']).dt.days

    # Fix results without last interaction
    df_trip['trip_size']     = df_trip['trip_size'] - 1
    df_trip['duration_sum']  = df_trip['duration_list'].apply(sum)

    return df_trip.reset_index()

  def filter_train_data(self, df):
    # remove outliers
    df_trip = df.groupby(['utrip_id']).agg(
        trip_size=('checkin', 'count'),
        duration_sum=('duration', 'sum')
    )

    # default values cames from EDA.ipynb
    df_trip = df_trip[(df_trip['trip_size'] < LIMIT_TRIP_SIZE) & 
                      (df_trip['trip_size'] > 0) & 
                      (df_trip['duration_sum'] < LIMIT_DURATION_SUM)]

    # Filter
    df = df[df['utrip_id'].isin(list(df_trip.index))]

    return df

  def add_steps_interaction(self, df, max_step):
    # Group By Trip
    data = []
    print("steps... 0 to ", max_step)
    for i in range(1, max_step):
      df_step = df[(df.step <= i) & (df.step >= i-self.window_trip)]
      print("step ", i, "total data ", len(df_step))

      df_step = self.group_by_trip(df_step)
      data.append(df_step)

    df_trip = pd.concat(data)\
                .sort_values(['utrip_id', 'trip_size'])\
                .drop_duplicates(subset = ['utrip_id', 'trip_size'], keep = 'first')

    return df_trip


  def run(self):
    os.makedirs(DATASET_DIR, exist_ok=True)

    df      = pd.read_csv(BASE_DATASET_FILE, 
                    dtype={"user_id": str, "city_id": str, 'affiliate_id': str,'utrip_id': str}, 
                    parse_dates=['checkin', 'checkout']).sort_values('checkin')

    #df      = df.merge(df_user, on='user_id')

    print(df.head())
    print(df.shape)

    if self.test_split > 0:

        df_trip = df[['utrip_id']].drop_duplicates()
        df_train, df_test = train_test_split(df_trip, test_size=self.test_split, random_state=42)
        df_train, df_test = df[df['utrip_id'].isin(df_train['utrip_id'])].sort_values('checkin'), \
                            df[df['utrip_id'].isin(df_test['utrip_id'])].sort_values('checkin')
    else:
        df_train = df
        df_test  = pd.read_csv(BASE_DATASET_TEST_FILE, 
                        dtype={"user_id": str, "city_id": str, 'affiliate_id': str,'utrip_id': str}, 
                        parse_dates=['checkin', 'checkout']).sort_values('checkin')
        

    print(df_train.shape, df_test.shape)
    # Add General Features
    self.add_general_features(df_train)
    self.add_general_features(df_test)

    # Add/Filter  Train Informaion
    # --------------------------------------------------------
    df_train = self.filter_train_data(df_train)
    
    # add country_count
    df_country_count = df_train.groupby(['city_id']).agg(country_count=('user_id','count')).reset_index()
    print(df_country_count.head())
    df_train = df_train.merge(df_country_count, on='city_id', how='left')
    df_test['country_count'] = 1

    print(df_train.head())
    print(df_train.shape)
    # --------------------------------------------------------

    # Group Trip
    df_trip_train = self.group_by_trip(df_train)
    df_trip_test  = self.group_by_trip(df_test)

    # Add Steps Interaction in Train Set
    #df_trip_train = pd.concat([df_trip_train, 
    #                         self.add_steps_interaction(df_train, df_train.step.max()-1)])
    
    print(df_trip_train.head())
    print(df_trip_train.shape)

    # Filter after
    # yes, the trips in test set contain at least 3 reservations
    df_trip_train = df_trip_train[df_trip_train['trip_size'] >= 3]
    df_trip_test  = df_trip_test[df_trip_test['trip_size'] >= 3]

    # Add User Features
    df_user = pd.read_csv(os.path.join(DATASET_DIR, "all_user_features.csv"), 
                    dtype={"user_id": str}, usecols=["user_id", 'user_features'])    
    df_trip_train  = df_trip_train.merge(df_user, on='user_id', how='left')
    df_trip_test   = df_trip_test.merge(df_user, on='user_id', how='left')

    # Remove duplicates
    df_trip_train = df_trip_train.groupby(['utrip_id', 'last_step']).last().reset_index()
    df_trip_test  = df_trip_test.groupby(['utrip_id', 'last_step']).last().reset_index()

    # Save
    df_trip_train.to_csv(self.output()[0].path, index=False)
    df_trip_test.to_csv(self.output()[1].path, index=False)

class SessionInteractionDataFrame(BasePrepareDataFrames):
    test_split: float = luigi.IntParameter(default=0.1)
    window_trip: int = luigi.IntParameter(default=5)
    filter_last_step: bool = luigi.BoolParameter(default=False)
    balance_sample_step: int = luigi.IntParameter(default=0)
    available_arms_size: int = luigi.IntParameter(default=1)
    filter_trip_size: int = luigi.IntParameter(default=0)

    item_column: str = luigi.Parameter(default="last_city_id")

    def requires(self):
        return SplitAndPreprocessDataset(test_split=self.test_split, 
                                          window_trip=self.window_trip)

    @property
    def timestamp_property(self) -> str:
        return "start_trip"

    @property
    def dataset_dir(self) -> str:
        return DATASET_DIR

    @property
    def read_data_frame_path(self) -> pd.DataFrame:
        return self.input()[0].path

    def read_data_frame(self) -> pd.DataFrame:
        df = pd.read_csv(self.read_data_frame_path)#.sample(10000)

        if self.filter_trip_size > 0:
            df = df[df['trip_size'] >= self.filter_trip_size]
        
        print(df.shape)

        return df

    # @property
    # def task_name(self):
    #     return self.task_id.split("_")[-1]

    # @property
    # def scaler_file_path(self):
    #     if self.normalize_file_path != None:
    #         return DATASET_DIR+'/'+self.normalize_file_path
    #     return DATASET_DIR+'/{}_std_scaler.pkl'.format(self.task_name)

    def transform_data_frame(self, df: pd.DataFrame, data_key: str) -> pd.DataFrame:
        #from IPython import embed; embed()

        # transform array
        for c in ['hotel_country_list', 'duration_list']:
            if len(df) > 0 and isinstance(df.iloc[0][c],str):
                df[c] = df[c].apply(eval)

        # add features
        df['start_trip'] = pd.to_datetime(df['start_trip'])
        df['start_trip_quarter']    = df['start_trip'].dt.quarter
        df['start_trip_month']      = df['start_trip'].dt.month
        df['start_trip_day']        = df['start_trip'].dt.dayofyear
        df['start_trip_week']       = df['start_trip'].dt.dayofweek
        df['start_trip_is_weekend'] = df['start_trip'].dt.day_name().apply(lambda x : 1 if x in ['Saturday','Sunday'] else 0)

        # Dense Features
        dense_features = ['is_multiple_country', 'trip_size', 'duration_sum']

        dense_features = [
            'start_trip_is_weekend', 'start_trip_quarter_sin', 'start_trip_quarter_cos', 
            'start_trip_month_sin', 'start_trip_month_cos', 
            'start_trip_day_sin', 'start_trip_day_cos', 
            'start_trip_week_sin', 'start_trip_week_cos']


        df['start_trip_quarter_sin']    = np.sin(2 * np.pi * df['start_trip_quarter']/4)
        df['start_trip_quarter_cos']    = np.cos(2 * np.pi * df['start_trip_quarter']/4)
        df['start_trip_month_sin']      = np.sin(2 * np.pi * df['start_trip_month']/12)
        df['start_trip_month_cos']      = np.cos(2 * np.pi * df['start_trip_month']/12)
        df['start_trip_day_sin']        = np.sin(2 * np.pi * df['start_trip_day']/365)
        df['start_trip_day_cos']        = np.cos(2 * np.pi * df['start_trip_day']/365)
        df['start_trip_week_sin']       = np.sin(2 * np.pi * df['start_trip_week']/6)
        df['start_trip_week_cos']       = np.cos(2 * np.pi * df['start_trip_week']/6)

        df['is_multiple_country'] = df['hotel_country_list'].apply(lambda x: len(list(np.unique(x))) > 1 ).astype(int)
        df['trip_size']     = df['trip_size']/LIMIT_TRIP_SIZE
        df['duration_sum']  = df['duration_list'].apply(sum)/(LIMIT_DURATION_SUM*10)

        df['dense_features'] = df[dense_features].values.tolist()

        # group by u_trip
        df_last_step = df.sort_values(['utrip_id', 'trip_size']).reset_index()\
                          .groupby(['utrip_id']).last().reset_index()

        if data_key == 'TEST_GENERATOR': 
            df = df_last_step
            #TODO
            #df['user_features'] = [[] for i in range(len(df))]            

        elif self.filter_last_step:
            if self.balance_sample_step > 0:
                _val_size = 1.0/self.n_splits if self.dataset_split_method == "k_fold" else self.val_size

                if data_key == 'VALIDATION_DATA':
                    _sample_view_size = int(self.balance_sample_step * _val_size)
                else:
                    _sample_view_size = int(self.balance_sample_step * (1-_val_size))
                
                df_steps = df[~df.index.isin(df_last_step['index'])] # Filter only step midde

                if len(df_steps) > 0:
                    df_steps = self.sample_balance_df(df_steps, _sample_view_size) # view
                    df = pd.concat([df_last_step, df_steps]).drop_duplicates(subset = ['utrip_id', 'trip_size'], keep = 'first')
                else:
                    df = df_last_step
            else:
                df = df_last_step

        return df
    
    def sample_balance_df(self, df, n_samples, state=42):
        df['sample_weights'] = 1/np.log(1+df['country_count'])
        return df.sample(n_samples, weights='sample_weights', random_state=state)