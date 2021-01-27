# NARMModel


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
    "test_split": 0.0, 
    "window_trip": 5,
    "column_stratification": "user_id",
    "filter_last_step": true,
    "balance_sample_step": 200000,
    "filter_trip_size": 0 }' \
  --test-size 0 \
  --val-size 0.1 \
  --early-stopping-min-delta 0.0001 \
  --learning-rate 0.001 \
  --metrics='["loss", "top_k_acc"]' \
  --batch-size 128 \
  --loss-function ce \
  --loss-function-class loss.FocalLoss \
  --epochs 100

PYTHONPATH="." luigi --module evaluation EvaluationTask \
--model-task-class "mars_gym.simulation.training.SupervisedModelTraining" \
--model-task-id SupervisedModelTraining____mars_gym_model_b____2b5e1f0d72 \
--file "/media/workspace/booking_challenge/output/booking/dataset/test_0.0_5.csv"  \
--submission-size 4 \
--local-scheduler

# PYTHONPATH="." luigi --module evaluation EvaluationTask \
# --model-task-class "mars_gym.simulation.training.SupervisedModelTraining" \
# --model-task-id SupervisedModelTraining____mars_gym_model_b____2b5e1f0d72 \
# --file "/media/workspace/booking_challenge/output/booking/dataset/test_0.1_5.csv"  \
# --local-scheduler