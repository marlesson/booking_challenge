## Geral


* https://github.com/mhjabreel/STF-RNN/blob/master/models.py


###
```bash
PYTHONPATH="."  luigi  \
--module train CoOccurrenceTraining  \
--project config.base_rnn \
--local-scheduler  \
--data-frames-preparation-extra-params '{
  "sample_days": 500, 
  "test_days": 30,
  "window_trip": 5,
  "column_stratification": "user_id",
  "filter_last_step": true,
  "balance_sample_step": 200000,
  "filter_trip_size": 0 }' \
--test-size 0 \
--early-stopping-min-delta 0.0001 \
--learning-rate 0.001 \
--metrics='["loss"]' \
--batch-size 128 \
--loss-function ce \
--epochs 100


PYTHONPATH="." luigi --module evaluation EvaluationTask \
--model-task-class "train.CoOccurrenceTraining" \
--model-task-id CoOccurrenceTraining____mars_gym_model_b____563c1ab497 \
--file "/media/workspace/booking_challenge/output/booking/dataset/test_500_30_5.csv"  \
--local-scheduler \
--model-eval "coocorrence"
```

### RNNAttModel

```bash

mars-gym run supervised --project config.conf1_rnn \
  --recommender-module-class model.RNNAttModel \
  --recommender-extra-params '{
    "n_factors": 100, 
    "hidden_size": 300, 
    "n_layers": 1, 
    "dropout": 0.2, 
    "from_index_mapping": false,
    "path_item_embedding": false, 
    "freeze_embedding": false}' \
  --data-frames-preparation-extra-params '{
    "sample_days": 500, 
    "test_days": 30,
    "window_trip": 5,
    "column_stratification": "user_id",
    "filter_last_step": true,
    "balance_sample_step": 200000,
    "filter_trip_size": 0 }' \
  --test-size 0 \
  --early-stopping-min-delta 0.0001 \
  --learning-rate 0.001 \
  --metrics='["loss"]' \
  --batch-size 128 \
  --loss-function ce \
  --epochs 100
```


mars-gym evaluate supervised --model-task-id SupervisedModelTraining____mars_gym_model_b____f3425bd577


PYTHONPATH="." luigi --module evaluation EvaluationTask \
--model-task-class "mars_gym.simulation.training.SupervisedModelTraining" \
--model-task-id SupervisedModelTraining____mars_gym_model_b____1ff373b221 \
--file "/media/workspace/booking_challenge/output/booking/dataset/test_500_30_5.csv"  \
--local-scheduler

{
    "task_name": "SupervisedModelTraining____mars_gym_model_b____e0f98a21d5_ac72640b98",
    "count": 11860,
    "acc@4": 0.49232715008431704
}


# NARMModel


```bash
mars-gym run supervised --project config.conf1_rnn \
  --recommender-module-class model.NARMModel \
  --recommender-extra-params '{
    "n_factors": 100, 
    "hidden_size": 300, 
    "n_layers": 1, 
    "dropout": 0.2, 
    "from_index_mapping": false,
    "path_item_embedding": false, 
    "freeze_embedding": false}' \
  --data-frames-preparation-extra-params '{
    "sample_days": 500, 
    "test_days": 30,
    "window_trip": 5,
    "column_stratification": "user_id",
    "filter_last_step": true,
    "balance_sample_step": 200000,
    "filter_trip_size": 0 }' \
  --test-size 0 \
  --early-stopping-min-delta 0.0001 \
  --learning-rate 0.001 \
  --metrics='["loss"]' \
  --batch-size 128 \
  --loss-function ce \
  --epochs 100
```


PYTHONPATH="." luigi --module evaluation EvaluationTask \
--model-task-class "mars_gym.simulation.training.SupervisedModelTraining" \
--model-task-id SupervisedModelTraining____mars_gym_model_b____9d1e31dd5a \
--file "/media/workspace/booking_challenge/output/booking/dataset/test_500_30_5.csv"  \
--local-scheduler


{
    "task_name": "SupervisedModelTraining____mars_gym_model_b____852bbd76fd_0a12286915",
    "count": 11860,
    "acc@4": 0.510539629005059
}

PYTHONPATH="." luigi --module evaluation EvaluationTask \
--model-task-class "mars_gym.simulation.training.SupervisedModelTraining" \
--model-task-id SupervisedModelTraining____mars_gym_model_b____9d1e31dd5a \
--file "/media/workspace/booking_challenge/output/booking/dataset/test_500_30_5.csv"  \
--neighbors-file "/media/workspace/booking_challenge/output/booking/dataset/neighbors_dict_sim_map.pkl" \
--local-scheduler


# Caser

```bash
mars-gym run supervised --project config.conf1_rnn \
  --recommender-module-class model.Caser \
  --recommender-extra-params '{
      "n_factors": 100, 
      "p_L": 10, 
      "p_nh": 16,
      "p_nv": 4,  
      "dropout": 0.2, 
      "hist_size": 10, 
      "from_index_mapping": false,
      "path_item_embedding": false, 
      "freeze_embedding": false}' \
  --data-frames-preparation-extra-params '{
    "sample_days": 500, 
    "test_days": 30,
    "window_trip": 10,
    "column_stratification": "user_id",
    "filter_last_step": true,
    "balance_sample_step": 0,
    "filter_trip_size": 0 }' \
  --test-size 0 \
  --early-stopping-min-delta 0.0001 \
  --learning-rate 0.001 \
  --metrics='["loss"]' \
  --loss-function ce \
  --batch-size 128 \
  --epochs 100
```
PYTHONPATH="." luigi --module evaluation EvaluationTask \
--model-task-class "mars_gym.simulation.training.SupervisedModelTraining" \
--model-task-id SupervisedModelTraining____mars_gym_model_b____1c9179030a \
--file "/media/workspace/booking_challenge/output/booking/dataset/test_500_30_10.csv"  \
--local-scheduler

PYTHONPATH="." luigi --module evaluation EvaluationTask \
--model-task-class "mars_gym.simulation.training.SupervisedModelTraining" \
--model-task-id SupervisedModelTraining____mars_gym_model_b____6a588b261e \
--file "/media/workspace/booking_challenge/output/booking/dataset/test_500_30_5.csv"  \
--local-scheduler

{
    "task_name": "SupervisedModelTraining____mars_gym_model_b____1316eac6fa_399ec6a778",
    "count": 11860,
    "acc@4": 0.5043001686340641
}





# MLTransformerModel2

```bash
mars-gym run supervised --project config.conf1_rnn \
  --recommender-module-class model.MLTransformerModel \
  --recommender-extra-params '{
      "n_factors": 100, 
      "n_hid": 50,
      "n_head": 2,
      "n_layers": 1,
      "num_filters": 100,
      "dropout": 0.2, 
      "hist_size": 5, 
      "from_index_mapping": false,
      "path_item_embedding": false, 
      "freeze_embedding": false}' \
  --data-frames-preparation-extra-params '{
    "sample_days": 500, 
    "test_days": 30,
    "window_trip": 5,
    "column_stratification": "user_id",
    "filter_last_step": true,
    "balance_sample_step": 200000,
    "filter_trip_size": 0 }' \
  --test-size 0 \
  --early-stopping-min-delta 0.0001 \
  --learning-rate 0.001 \
  --metrics='["loss"]' \
  --loss-function ce \
  --batch-size 128 \
  --epochs 100
```
PYTHONPATH="." luigi --module evaluation EvaluationTask \
--model-task-class "mars_gym.simulation.training.SupervisedModelTraining" \
--model-task-id SupervisedModelTraining____mars_gym_model_b____c18805d18b \
--file "/media/workspace/booking_challenge/output/booking/dataset/test_500_30_5.csv"  \
--local-scheduler