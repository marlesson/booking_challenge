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

mars-gym run supervised \
--project mercado_livre.config.mercado_livre_narm \
--recommender-module-class model.MLNARMModel2 \
--recommender-extra-params '{
  "n_factors": 100, 
  "hidden_size": 200, 
  "dense_size": 19,
  "n_layers": 1, 
  "dropout": 0.2, 
  "history_window": 20, 
  "history_word_window": 3,
  "from_index_mapping": false,
  "path_item_embedding": "~/triplet_session/output/mercado_livre/assets/mercadolivre-100d.bin", 
  "freeze_embedding": true}' \
--data-frames-preparation-extra-params '{
  "sample_days": 60, 
  "history_window": 20, 
  "column_stratification": "SessionID",
  "normalize_dense_features": "min_max",
  "min_interactions": 5,
  "filter_only_buy": true,
  "sample_balance": true,
  "sample_view": 300000}' \
--optimizer adam \
--optimizer-params '{"weight_decay": 1e-4}' \
--test-size 0.0 \
--val-size 0.1 \
--early-stopping-min-delta 0.0001 \
--test-split-type random \
--dataset-split-method column \
--learning-rate 0.001 \
--metrics='["loss"]' \
--generator-workers 5  \
--batch-size 512 \
--loss-function ce \
--epochs 1000 \
--obs "All"

echo "..."
# k-fold@0

for f in 0 1 2 3 4; do
  echo "kfold: $f"

  mars-gym run supervised \
  --project mercado_livre.config.mercado_livre_narm \
  --recommender-module-class model.MLNARMModel2 \
  --recommender-extra-params '{
    "n_factors": 100, 
    "hidden_size": 200, 
    "dense_size": 19,
    "n_layers": 1, 
    "dropout": 0.2, 
    "history_window": 20, 
    "history_word_window": 3,
    "from_index_mapping": false,
    "path_item_embedding": "/media/workspace/triplet_session/output/mercado_livre/assets/mercadolivre-100d.bin", 
    "freeze_embedding": true}' \
  --data-frames-preparation-extra-params '{
    "sample_days": 60, 
    "history_window": 20, 
    "column_stratification": "SessionID",
    "normalize_dense_features": "min_max",
    "min_interactions": 5,
    "filter_only_buy": true,
    "sample_balance": true,
    "sample_view": 300000}' \
  --optimizer adam \
  --optimizer-params '{"weight_decay": 1e-4}' \
  --test-size 0.0 \
  --val-size 0.1 \
  --early-stopping-min-delta 0.0001 \
  --test-split-type random \
  --dataset-split-method k_fold \
  --n-splits 5 \
  --split-index $f \
  --learning-rate 0.001 \
  --metrics='["loss"]' \
  --generator-workers 5  \
  --batch-size 512 \
  --loss-function ce \
  --epochs 1000 \
  --obs "kfold@ $f"

  echo "..."
done

## Submission

for u in "${models[@]}"
do
  echo ""
  echo "Make Submission for $u"

  PYTHONPATH="." luigi --module mercado_livre.evaluation EvaluationSubmission \
  --local-scheduler \
  --model-task-class "mars_gym.simulation.training.SupervisedModelTraining" \
  --model-task-id $u \
  --normalize-file-path $std_scaler \
  --history-window 20 \
  --batch-size 1000 \
  --percent-limit 1 \
  --submission-size 100 \
  --model-eval "model" 
done