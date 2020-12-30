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
--model-task-id SupervisedModelTraining____mars_gym_model_b____cf9405841a \
--local-scheduler \
--file "/media/workspace/booking_challenge/output/booking/dataset/test_500_30_5.csv"
