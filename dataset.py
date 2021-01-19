from typing import Tuple, List, Union, Optional, Dict, Any

import functools
import numpy as np
import pandas as pd
import random
from torch.utils.data import Dataset
import pickle

from mars_gym.meta_config import ProjectConfig, IOType, Column
from mars_gym.utils.index_mapping import map_array
from mars_gym.utils.utils import parallel_literal_eval, reduce_df_mem
import gc

NEIGHBOARD_FILE = "/media/workspace/booking_challenge/output/booking/dataset/neighbors_dict_sim_map_c.pkl"


def literal_eval_array_columns(data_frame: pd.DataFrame, columns: List[Column]):
    for column in columns:
        if (
            column.type
            in (IOType.FLOAT_ARRAY, IOType.INT_ARRAY, IOType.INDEXABLE_ARRAY)
            and column.name in data_frame
        ):
            data_frame[column.name] = parallel_literal_eval(data_frame[column.name])


def preprocess_interactions_data_frame(
    data_frame: pd.DataFrame, project_config: ProjectConfig
):
    if len(data_frame) == 0:
        return data_frame

    data_frame[project_config.user_column.name] = data_frame[
        project_config.user_column.name
    ].astype(str)
    data_frame[project_config.item_column.name] = data_frame[
        project_config.item_column.name
    ].astype(str)

    literal_eval_array_columns(
        data_frame,
        [
            project_config.user_column,
            project_config.item_column,
            project_config.output_column,
        ]
        + [input_column for input_column in project_config.other_input_columns]\
        + [input_column for input_column in project_config.auxiliar_output_columns]\
        + [input_column for input_column in project_config.metadata_columns],
    )
    #from IPython import embed; embed()

    if project_config.available_arms_column_name and \
        project_config.available_arms_column_name in data_frame.columns and isinstance(
        data_frame.iloc[0][project_config.available_arms_column_name], str
    ):
        data_frame[project_config.available_arms_column_name] = parallel_literal_eval(
            data_frame[project_config.available_arms_column_name]
        )
    
    return data_frame


def preprocess_metadata_data_frame(
    metadata_data_frame: pd.DataFrame, project_config: ProjectConfig
) -> Dict[str, np.ndarray]:
    NON_UTIL_UID = 3

    metadata_data_frame = metadata_data_frame[
        metadata_data_frame[project_config.item_column.name] != 0
    ]  # Remove unkown

    metadata_data_frame = metadata_data_frame.set_index(
        project_config.item_column.name, drop=False
    ).sort_index()
    
    if not (
        np.arange(NON_UTIL_UID, len(metadata_data_frame.index) + NON_UTIL_UID) == metadata_data_frame.index
    ).all():
        raise ValueError("The item index is not contiguous")

    embeddings_for_metadata: Dict[str, np.ndarray] = {}
    for metadata_column in project_config.metadata_columns:

        emb = metadata_data_frame[metadata_column.name].values.tolist()
        embedding = np.array(emb, dtype=metadata_column.type.dtype)
        pad = np.zeros((NON_UTIL_UID,) + embedding.shape[1:])
        #
        embedding = np.concatenate((pad, embedding))
        embeddings_for_metadata[metadata_column.name] = embedding
        # from IPython import embed; embed()
    return embeddings_for_metadata


def _rand_int_except(low: int, high: int, exception: int) -> int:
    while True:
        number = np.random.randint(low, high)
        if number != exception:
            return number


def _choose_except(values: list, exception: Any) -> int:
    while True:
        value = random.choice(values)
        if value != exception:
            return value

def _change_idx(values: list, neighbors_dict = dict, r = 1):

    if random.random() <= r:
        number = np.random.randint(0, len(values))

        if values[number] in neighbors_dict:
            neighbors_dict[values[number]].append(values[number])
            value  = random.choice(neighbors_dict[values[number]])

            values[number] = value

    return values

class InteractionsDataset(Dataset):
    def __init__(
        self,
        data_frame: pd.DataFrame,
        embeddings_for_metadata: Optional[Dict[Any, np.ndarray]],
        project_config: ProjectConfig,
        index_mapping: Dict[str, Dict[Any, int]],
        *args,
        **kwargs
    ) -> None:
        self._project_config = project_config
        self._index_mapping  = index_mapping
        self._input_columns: List[Column] = project_config.input_columns
        input_column_names = [input_column.name for input_column in self._input_columns]

        if project_config.item_is_input:
            self._item_input_index = self._input_columns.index(
                project_config.item_column
            )

        auxiliar_output_column_names = [
            auxiliar_output_column.name
            for auxiliar_output_column in project_config.auxiliar_output_columns
        ]
        self._data_frame = data_frame[
            set(
                input_column_names
                + [project_config.output_column.name]
                + auxiliar_output_column_names
            ).intersection(data_frame.columns)
        ]
        self._embeddings_for_metadata = embeddings_for_metadata

    def __len__(self) -> int:
        return self._data_frame.shape[0]

    def _convert_dtype(self, value: np.ndarray, type: IOType) -> np.ndarray:
        if type == IOType.INDEXABLE:
            return value.astype(np.int64)
        if type == IOType.NUMBER:
            return value.astype(np.float64)
        if type in (IOType.INT_ARRAY, IOType.INDEXABLE_ARRAY):
            return np.array([np.array(v, dtype=np.int64) for v in value])
        if type == IOType.FLOAT_ARRAY:
            return np.array([np.array(v, dtype=np.float64) for v in value])
        return value

    def __getitem__(
        self, indices: Union[int, List[int], slice]
    ) -> Tuple[Tuple[np.ndarray, ...], Union[np.ndarray, Tuple[np.ndarray, ...]]]:
        if isinstance(indices, int):
            indices = [indices]
        rows: pd.Series = self._data_frame.iloc[indices]

        inputs = tuple(
            self._convert_dtype(rows[column.name].values, column.type)
            for column in self._input_columns if column.name in self._data_frame.columns
        )
        
        if (
            self._project_config.item_is_input
            and self._embeddings_for_metadata is not None
        ):
            item_indices = inputs[self._item_input_index]
            inputs += tuple(
                self._embeddings_for_metadata[column.name][item_indices]
                for column in self._project_config.metadata_columns if column.name not in self._data_frame.columns
            )
        #from IPython import embed; embed()
        #
        output = self._convert_dtype(
            rows[self._project_config.output_column.name].values,
            self._project_config.output_column.type,
        )
        if self._project_config.auxiliar_output_columns:
            output = tuple([output]) + tuple(
                self._convert_dtype(rows[column.name].values, column.type)
                for column in self._project_config.auxiliar_output_columns
            )
        # print(inputs)
        return inputs, output

class InteractionsDatasetWithMask(Dataset):
    def __init__(
        self,
        data_frame: pd.DataFrame,
        embeddings_for_metadata: Optional[Dict[Any, np.ndarray]],
        project_config: ProjectConfig,
        index_mapping: Dict[str, Dict[Any, int]],
        data_key: str,
        *args,
        **kwargs
    ) -> None:
        self._project_config = project_config
        self._index_mapping  = index_mapping
        self._input_columns: List[Column] = project_config.input_columns
        self._data_key = data_key
        input_column_names = [input_column.name for input_column in self._input_columns]

        neighbors_file = NEIGHBOARD_FILE
        neighbors_dict = None
        if neighbors_file:
            print("load neighbors...")
            with open(neighbors_file, "rb") as pkl_handle:
                neighbors_file = pickle.load(pkl_handle)
                neighbors_dict = {}
                for key, values in neighbors_file.items():
                    neighbors_dict[index_mapping['city_id_list'][key]] = [index_mapping['city_id_list'][k] for k in values]
                neighbors_dict[0] = []
                neighbors_dict[1] = []
                neighbors_dict[2] = []
                neighbors_dict[3] = []
        
        self.neighbors_dict = neighbors_dict
        self.prob_neighbors = 0.1

        if project_config.item_is_input:
            self._item_input_index = self._input_columns.index(
                project_config.item_column
            )

        auxiliar_output_column_names = [
            auxiliar_output_column.name
            for auxiliar_output_column in project_config.auxiliar_output_columns
        ]
        self._data_frame = data_frame[
            set(
                input_column_names
                + [project_config.output_column.name]
                + auxiliar_output_column_names
            ).intersection(data_frame.columns)
        ]
        self._embeddings_for_metadata = embeddings_for_metadata
        self._list_input_index = 2

    def __len__(self) -> int:
        return self._data_frame.shape[0]

    def _convert_dtype(self, value: np.ndarray, type: IOType) -> np.ndarray:
        if type == IOType.INDEXABLE:
            return value.astype(np.int64)
        if type == IOType.NUMBER:
            return value.astype(np.float64)
        if type in (IOType.INT_ARRAY, IOType.INDEXABLE_ARRAY):
            return np.array([np.array(v, dtype=np.int64) for v in value])
        if type == IOType.FLOAT_ARRAY:
            return np.array([np.array(v, dtype=np.float64) for v in value])
        return value

    def __getitem__(
        self, indices: Union[int, List[int], slice]
    ) -> Tuple[Tuple[np.ndarray, ...], Union[np.ndarray, Tuple[np.ndarray, ...]]]:
        if isinstance(indices, int):
            indices = [indices]
        rows: pd.Series = self._data_frame.iloc[indices]

        inputs = tuple(
            self._convert_dtype(rows[column.name].values, column.type)
            for column in self._input_columns if column.name in self._data_frame.columns
        )
        
        if (
            self._project_config.item_is_input
            and self._embeddings_for_metadata is not None
        ):
            item_indices = inputs[self._item_input_index]
            inputs += tuple(
                self._embeddings_for_metadata[column.name][item_indices]
                for column in self._project_config.metadata_columns if column.name not in self._data_frame.columns
            )
        #from IPython import embed; embed()
        #
        output = self._convert_dtype(
            rows[self._project_config.output_column.name].values,
            self._project_config.output_column.type,
        )
        if self._project_config.auxiliar_output_columns:
            output = tuple([output]) + tuple(
                self._convert_dtype(rows[column.name].values, column.type)
                for column in self._project_config.auxiliar_output_columns
            )
        # print(inputs)
        if self._data_key == 'train_data':
            _neighbors_dict = self.neighbors_dict.copy()
            list_modif = np.array([_change_idx(l,_neighbors_dict) for l in inputs[self._list_input_index]])
        #inputs[self._list_input_index]

        return inputs, output

class InteractionsWithNegativeItemGenerationDataset(InteractionsDataset):
    def __init__(
        self,
        data_frame: pd.DataFrame,
        embeddings_for_metadata: Optional[Dict[str, np.ndarray]],
        project_config: ProjectConfig,
        index_mapping: Dict[str, Dict[Any, int]],
        negative_proportion: float = 0.8,
        *args,
        **kwargs
    ) -> None:
        # data_frame = data_frame[data_frame[project_config.output_column.name] > 0]

        super().__init__(
            data_frame,
            embeddings_for_metadata,
            project_config,
            index_mapping,
            *args,
            **kwargs
        )
        self._negative_proportion = negative_proportion
        self._max_item_idx = data_frame[project_config.item_column.name].max()

    def __len__(self) -> int:
        return super().__len__() + int(
            (1 / (1 - self._negative_proportion) - 1) * super().__len__()
        )

    def __getitem__(
        self, indices: Union[int, List[int], slice]
    ) -> Tuple[Tuple[np.ndarray, ...], np.ndarray]:
        if isinstance(indices, int):
            indices = [indices]
        if isinstance(indices, slice):
            indices = list(range(len(self))[indices])

        n = super().__len__()

        positive_indices = [index for index in indices if index < n]
        num_of_negatives = len(indices) - len(positive_indices)
        positive_input, positive_output = super().__getitem__(positive_indices)

        if num_of_negatives > 0:
            sample_positive_indices = list(
                np.random.randint(0, n, size=num_of_negatives)
            )

            negative_input, _ = super().__getitem__(sample_positive_indices)
            negative_output = self._convert_dtype(
                np.zeros(num_of_negatives), self._project_config.output_column.type
            )

            negative_input = list(negative_input)
            negative_input[self._item_input_index] = np.array(
                [
                    _rand_int_except(0, self._max_item_idx + 1, exception=item_idx)
                    for item_idx in negative_input[self._item_input_index]
                ]
            )
            negative_input = tuple(negative_input)

            if positive_indices:
                input_ = tuple(
                    np.concatenate([positive_array, negative_array])
                    for positive_array, negative_array in zip(
                        positive_input, negative_input
                    )
                )
                output = np.concatenate([positive_output, negative_output])
            else:
                input_ = negative_input
                output = negative_output
        else:
            input_ = positive_input
            output = positive_output

        return input_, output

