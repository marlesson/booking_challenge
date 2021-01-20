from mars_gym.data.dataset import (
    InteractionsDataset,
    InteractionsWithNegativeItemGenerationDataset,
)
from dataset import InteractionsDatasetWithMask

from mars_gym.meta_config import *
import data
import warnings
warnings.filterwarnings("ignore")


base_rnn = ProjectConfig(
    base_dir=data.BASE_DIR,
    prepare_data_frames_task=data.SessionInteractionDataFrame,
    dataset_class=InteractionsDatasetWithMask,
    user_column=Column("user_id", IOType.INDEXABLE),
    item_column=Column("last_city_id", IOType.INDEXABLE),
    timestamp_column_name="start_trip",
    other_input_columns=[
        Column("city_id_list", IOType.INDEXABLE_ARRAY, same_index_as="last_city_id"),
    ],
    output_column=Column("last_city_id", IOType.INDEXABLE),
    recommender_type=RecommenderType.USER_BASED_COLLABORATIVE_FILTERING,
)


conf1_rnn = ProjectConfig(
    base_dir=data.BASE_DIR,
    prepare_data_frames_task=data.SessionInteractionDataFrame,
    dataset_class=InteractionsDatasetWithMask,
    user_column=Column("user_id", IOType.INDEXABLE),
    item_column=Column("last_city_id", IOType.INDEXABLE, same_index_as="city_id_list"),
    timestamp_column_name="start_trip",
    other_input_columns=[
        Column("city_id_list", IOType.INDEXABLE_ARRAY),
        Column("affiliate_id_list", IOType.INDEXABLE_ARRAY),
        Column("device_class_list", IOType.INDEXABLE_ARRAY),
        Column("checkin_list", IOType.INT_ARRAY),
        Column("user_features", IOType.FLOAT_ARRAY), 
        Column("booker_country_list", IOType.INDEXABLE_ARRAY, same_index_as="last_hotel_country"),
        Column("duration_list", IOType.INT_ARRAY), #days_since_2016_list
        Column("start_trip_month", IOType.NUMBER),
        Column("dense_features", IOType.FLOAT_ARRAY),

    ],
    output_column=Column("last_city_id", IOType.INDEXABLE, same_index_as="city_id_list"),
    auxiliar_output_columns=[Column("last_hotel_country", IOType.INDEXABLE)],

    recommender_type=RecommenderType.USER_BASED_COLLABORATIVE_FILTERING,
)


