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
    "window_trip": 5,
    "from_index_mapping": false,
    "path_item_embedding": false, 
    "freeze_embedding": false}' \
  --data-frames-preparation-extra-params '{
    "sample_days": 500, 
    "test_days": 30,
    "window_trip": 5,
    "column_stratification": "user_id",
    "filter_last_step": true,
    "balance_sample_step": 500000,
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
--model-task-id SupervisedModelTraining____mars_gym_model_b____4bd8cd44d1 \
--file "/media/workspace/booking_challenge/output/booking/dataset/test_500_30_5.csv"  \
--local-scheduler

{
    "task_name": "SupervisedModelTraining____mars_gym_model_b____4bd8cd44d1_2b74728525",
    "count": 11860,
    "acc@4": 0.4992411467116358
}


# NARMModel


```bash
mars-gym run supervised --project config.conf1_rnn \
  --recommender-module-class model.NARMModel \
  --recommender-extra-params '{
    "n_factors": 100, 
    "hidden_size": 300, 
    "n_layers": 2, 
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
  --metrics='["loss", "top_k_acc"]' \
  --batch-size 128 \
  --loss-function ce \
  --loss-function-class loss.FocalLoss \
  --epochs 100
```

PYTHONPATH="." luigi --module evaluation EvaluationTask \
--model-task-class "mars_gym.simulation.training.SupervisedModelTraining" \
--model-task-id SupervisedModelTraining____mars_gym_model_b____ad7cf2835e \
--file "/media/workspace/booking_challenge/output/booking/dataset/test_500_30_5.csv"  \
--local-scheduler

{
    "task_name": "SupervisedModelTraining____mars_gym_model_b____2d2e241d4b_8b6b581e5d",
    "count": 11860,
    "acc@4": 0.5233558178752108
}



PYTHONPATH="." luigi --module evaluation EvaluationTask \
--model-task-class "mars_gym.simulation.training.SupervisedModelTraining" \
--model-task-id SupervisedModelTraining____mars_gym_model_b____ad7cf2835e \
--file "/media/workspace/booking_challenge/output/booking/dataset/test_500_30_10.csv"  \
--local-scheduler

{
    "task_name": "SupervisedModelTraining____mars_gym_model_b____ad7cf2835e_b3d288b0be",
    "count": 11860,
    "acc@4": 0.5182967959527824
}


# Caser

```bash
mars-gym run supervised --project config.conf1_rnn \
  --recommender-module-class model.Caser \
  --recommender-extra-params '{
      "n_factors": 100, 
      "p_L": 5, 
      "p_nh": 16,
      "p_nv": 4,  
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
    "balance_sample_step": 500000,
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
--model-task-id SupervisedModelTraining____mars_gym_model_b____f841e27287 \
--file "/media/workspace/booking_challenge/output/booking/dataset/test_500_30_5_.csv"  \
--local-scheduler

{
    "task_name": "SupervisedModelTraining____mars_gym_model_b____f841e27287_2882c0a0a4",
    "count": 11860,
    "acc@4": 0.5112984822934232
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
    "balance_sample_step": 500000,
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
--model-task-id SupervisedModelTraining____mars_gym_model_b____48bad5060c \
--file "/media/workspace/booking_challenge/output/booking/dataset/test_500_30_5.csv"  \
--local-scheduler

{
    "task_name": "SupervisedModelTraining____mars_gym_model_b____48bad5060c_89fa1a1958",
    "count": 11860,
    "acc@4": 0.4301011804384486
}
