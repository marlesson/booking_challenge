## Geral


* https://github.com/mhjabreel/STF-RNN/blob/master/models.py

###


mars-gym run supervised --project config.base_rnn \
  --recommender-module-class model.GRURecModel \
  --recommender-extra-params '{
    "n_factors": 100, 
    "hidden_size": 100, 
    "n_layers": 1, 
    "dropout": 0.2, 
    "from_index_mapping": false,
    "path_item_embedding": false, 
    "freeze_embedding": false}' \
  --data-frames-preparation-extra-params '{
    "sample_days": 500, 
    "test_days": 30,
    "window_trip": 5,
    "column_stratification": "user_id"}' \
  --test-size 0 \
  --early-stopping-min-delta 0.0001 \
  --learning-rate 0.001 \
  --metrics='["loss"]' \
  --batch-size 128 \
  --loss-function ce \
  --epochs 100



mars-gym evaluate supervised --model-task-id SupervisedModelTraining____mars_gym_model_b____cf9405841a


PYTHONPATH="." luigi --module evaluation EvaluationTask \
--model-task-class "mars_gym.simulation.training.SupervisedModelTraining" \
--model-task-id SupervisedModelTraining____mars_gym_model_b____1a02839e77 \
--file "/media/workspace/booking_challenge/output/booking/dataset/test_500_30_5.csv"  \
--local-scheduler


mars-gym run supervised --project config.base_rnn \
  --recommender-module-class model.GRURecModel \
  --recommender-extra-params '{
    "n_factors": 100, 
    "hidden_size": 100, 
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
    "filter_last_step": true}' \
  --test-size 0 \
  --early-stopping-min-delta 0.0001 \
  --learning-rate 0.001 \
  --metrics='["loss"]' \
  --batch-size 128 \
  --loss-function ce \
  --epochs 100

PYTHONPATH="." luigi --module evaluation EvaluationTask \
--model-task-class "mars_gym.simulation.training.SupervisedModelTraining" \
--model-task-id SupervisedModelTraining____mars_gym_model_b____072c21cf90 \
--file "/media/workspace/booking_challenge/output/booking/dataset/test_500_30_5.csv"  \
--local-scheduler


# NARMModel


```bash
mars-gym run supervised --project config.conf1_rnn \
  --recommender-module-class model.NARMModel \
  --recommender-extra-params '{
    "n_factors": 100, 
    "hidden_size": 100, 
    "n_layers": 1, 
    "dropout": 0.25, 
    "from_index_mapping": false,
    "path_item_embedding": false, 
    "freeze_embedding": false}' \
  --data-frames-preparation-extra-params '{
    "sample_days": 500, 
    "test_days": 30,
    "window_trip": 5,
    "column_stratification": "user_id",
    "filter_last_step": true,
    "balance_sample_step": 200000 }' \
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
--model-task-id SupervisedModelTraining____mars_gym_model_b____6d427e2852 \
--file "/media/workspace/booking_challenge/output/booking/dataset/test_500_30_5.csv"  \
--local-scheduler

# Caser

```bash
mars-gym run supervised --project config.base_rnn \
  --recommender-module-class model.Caser \
  --recommender-extra-params '{
      "n_factors": 100, 
      "p_L": 5, 
      "p_d": 50, 
      "p_nh": 16,
      "p_nv": 4,  
      "dropout": 0.1, 
      "hist_size": 5, 
      "from_index_mapping": false,
      "path_item_embedding": false, 
      "freeze_embedding": false}' \
  --data-frames-preparation-extra-params '{
    "sample_days": 500, 
    "test_days": 30,
    "window_trip": 5,
    "column_stratification": "user_id",
    "filter_last_step": false}' \
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
--model-task-id SupervisedModelTraining____mars_gym_model_b____0140abfecc \
--file "/media/workspace/booking_challenge/output/booking/dataset/test_500_30_5.csv"  \
--local-scheduler
