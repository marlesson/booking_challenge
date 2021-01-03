from mars_gym.data.dataset import (
    InteractionsDataset,
    InteractionsWithNegativeItemGenerationDataset,
)
from mars_gym.meta_config import *
import data
import warnings
warnings.filterwarnings("ignore")


base_rnn = ProjectConfig(
    base_dir=data.BASE_DIR,
    prepare_data_frames_task=data.SessionInteractionDataFrame,
    dataset_class=InteractionsDataset,
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
    dataset_class=InteractionsDataset,
    user_column=Column("user_id", IOType.INDEXABLE),
    item_column=Column("last_city_id", IOType.INDEXABLE),
    timestamp_column_name="start_trip",
    other_input_columns=[
        Column("city_id_list", IOType.INDEXABLE_ARRAY, same_index_as="last_city_id"),
        Column("hotel_country_list", IOType.INDEXABLE_ARRAY),
        Column("duration_list", IOType.INT_ARRAY), #days_since_2016_list
        Column("start_trip_month", IOType.NUMBER),
        Column("dense_features", IOType.FLOAT_ARRAY),

    ],
    output_column=Column("last_city_id", IOType.INDEXABLE),
    recommender_type=RecommenderType.USER_BASED_COLLABORATIVE_FILTERING,
)


