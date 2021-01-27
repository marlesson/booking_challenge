# bin/bash!

# # Step 1

# ## Train Big Model

# * Split and add features

# ```bash
#   mars-gym run data 
# ```

# * Build User Features

# * Build Neibohods

# * Train Model

# * Evaluate/Submission

# ## Train 4-folds models

# * Train Model k-folds

# * Make Submission k-Foods

# ## Ensamble 

# * Run Notebook Ensamble File


# Params

# std_scaler="7cef5bca66_std_scaler.pkl"
# models=( 
#   SupervisedModelTraining____mars_gym_model_b____c179ab54fa 
#   SupervisedModelTraining____mars_gym_model_b____c179ab54fa 
#   SupervisedModelTraining____mars_gym_model_b____c179ab54fa 
#   SupervisedModelTraining____mars_gym_model_b____c179ab54fa 
#   SupervisedModelTraining____mars_gym_model_b____c179ab54fa
#   SupervisedModelTraining____mars_gym_model_b____c179ab54fa )

# Training

# All

mars-gym run supervised --project config.conf1_rnn \
  --recommender-module-class model.NARMModel \
  --recommender-extra-params '{
    "n_factors": 50, 
    "hidden_size": 300, 
    "n_layers": 1, 
    "dropout": 0.2, 
    "n_user_features": 10,
    "n_dense_features": 5,
    "from_index_mapping": false,
    "path_item_embedding": false, 
    "freeze_embedding": false}' \
  --data-frames-preparation-extra-params '{
    "test_split": 0.1, 
    "window_trip": 10,
    "user_features_file": "all_user_features_10.csv",
    "column_stratification": "user_id",
    "filter_last_step": true,
    "balance_sample_step": 100000,
    "filter_trip_size": 0 }' \
  --test-size 0 \
  --val-size 0.1 \
  --early-stopping-min-delta 0.0001 \
  --early-stopping-patience 5 \
  --learning-rate 0.001 \
  --metrics='["loss", "top_k_acc", "top_k_acc2"]' \
  --batch-size 64 \
  --optimizer "radam" \
  --optimizer-params '{
    "weight_decay": 0.01
    }' \
  --loss-function ce \
  --loss-function-class loss.FocalLoss \
  --loss-function-params '{
    "alpha":1,
    "gamma":3,
    "c": 0.5,
    "epsilon": 0.1
    }' \
  --epochs 100 \
  --monitor-metric val_top_k_acc \
  --monitor-mode max \
  --generator-workers 2 \
  --obs "All"

echo "..."
# k-fold@0

for f in 0 1 2 3 4; do
  echo "kfold: $f"

mars-gym run supervised --project config.conf1_rnn \
  --recommender-module-class model.NARMModel \
  --recommender-extra-params '{
    "n_factors": 50, 
    "hidden_size": 300, 
    "n_layers": 1, 
    "dropout": 0.2, 
    "n_user_features": 10,
    "n_dense_features": 5,
    "from_index_mapping": false,
    "path_item_embedding": false, 
    "freeze_embedding": false}' \
  --data-frames-preparation-extra-params '{
    "test_split": 0.1, 
    "window_trip": 10,
    "user_features_file": "all_user_features_10.csv",
    "column_stratification": "user_id",
    "filter_last_step": true,
    "balance_sample_step": 10000,
    "filter_trip_size": 0 }' \
  --test-size 0 \
  --val-size 0.1 \
  --early-stopping-min-delta 0.0001 \
  --early-stopping-patience 5 \
  --learning-rate 0.001 \
  --metrics='["loss", "top_k_acc", "top_k_acc2"]' \
  --batch-size 64 \
  --optimizer "radam" \
  --optimizer-params '{
    "weight_decay": 0.01
    }' \
  --loss-function ce \
  --loss-function-class loss.FocalLoss \
  --loss-function-params '{
    "alpha":1,
    "gamma":3,
    "c": 0.5,
    "epsilon": 0.1
    }' \
  --epochs 100 \
  --monitor-metric val_top_k_acc \
  --monitor-mode max \
  --generator-workers 2 \
  --n-splits 5 \
  --dataset-split-method k_fold \
  --split-index $f  \
  --obs "kfold@ $f"

  echo "..."
done

## Submission

# for u in "${models[@]}"
# do
#   echo ""
#   echo "Make Submission for $u"

#   PYTHONPATH="." luigi --module mercado_livre.evaluation EvaluationSubmission \
#   --local-scheduler \
#   --model-task-class "mars_gym.simulation.training.SupervisedModelTraining" \
#   --model-task-id $u \
#   --normalize-file-path $std_scaler \
#   --history-window 20 \
#   --batch-size 1000 \
#   --percent-limit 1 \
#   --submission-size 100 \
#   --model-eval "model" 
# done