import luigi
from mars_gym.simulation.training import SupervisedModelTraining

if __name__ == '__main__':
  job = SupervisedModelTraining(
    project="config.conf1_rnn",
    recommender_module_class="model.RNNAttModel",
    recommender_extra_params={
      "n_factors": 100, 
      "hidden_size": 100, 
      "n_layers": 1, 
      "dropout": 0.25, 
      "from_index_mapping": False,
      "path_item_embedding": False, 
      "freeze_embedding": False},
    data_frames_preparation_extra_params={
      "sample_days": 500, 
      "test_days": 30,
      "window_trip": 5,
      "column_stratification": "user_id",
      "filter_last_step": True,
      "balance_sample_step": 200000},
    test_size= 0.0,
    metrics=['loss'],
    loss_function="ce", 
    batch_size=5,
    epochs=2
  )

  # job = SupervisedModelTraining(
  #   project="config.conf1_rnn",
  #   recommender_module_class="model.NARMModel",
  #   recommender_extra_params={
  #     "n_factors": 100, 
  #     "hidden_size": 100, 
  #     "n_layers": 1, 
  #     "dropout": 0.25, 
  #     "from_index_mapping": False,
  #     "path_item_embedding": False, 
  #     "freeze_embedding": False},
  #   data_frames_preparation_extra_params={
  #     "sample_days": 500, 
  #     "test_days": 30,
  #     "window_trip": 5,
  #     "column_stratification": "user_id",
  #     "filter_last_step": True,
  #     "balance_sample_step": 200000},
  #   test_size= 0.0,
  #   metrics=['loss'],
  #   loss_function="ce", 
  #   batch_size=5,
  #   epochs=2
  # )


  # job = SupervisedModelTraining(
  #   project="config.conf1_rnn",
  #   recommender_module_class="model.MLTransformerModel",
  #   recommender_extra_params={
  #     "n_factors": 100, 
  #     "n_hid": 50,
  #     "n_head": 2,
  #     "n_layers": 1,
  #     "num_filters": 100,
  #     "dropout": 0.2, 
  #     "hist_size": 5,
  #     "from_index_mapping": False,
  #     "path_item_embedding": False, 
  #     "freeze_embedding": False},
  #   data_frames_preparation_extra_params={
  #     "sample_days": 500, 
  #     "test_days": 30,
  #     "window_trip": 5,
  #     "column_stratification": "user_id",
  #     "filter_last_step": True,
  #     "balance_sample_step": 200000},
  #   test_size= 0.0,
  #   metrics=['loss'],
  #   loss_function="ce", 
  #   batch_size=2,
  #   epochs=2
  # )


  job.run()
