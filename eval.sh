# bin/bash!

# Params

std_scaler="7cef5bca66_std_scaler.pkl"
models=( 
  SupervisedModelTraining____mars_gym_model_b____6b5c4e0f46
  SupervisedModelTraining____mars_gym_model_b____3c881def21
  SupervisedModelTraining____mars_gym_model_b____57e987a1ff
  SupervisedModelTraining____mars_gym_model_b____d6ea2403f2
  SupervisedModelTraining____mars_gym_model_b____0c49038918
  SupervisedModelTraining____mars_gym_model_b____88fed69335
  SupervisedModelTraining____mars_gym_model_b____822330452c
  SupervisedModelTraining____mars_gym_model_b____16a8189b8b
  SupervisedModelTraining____mars_gym_model_b____fe94ffde36
  SupervisedModelTraining____mars_gym_model_b____37ce2fdd57
  SupervisedModelTraining____mars_gym_model_b____5c19dd615c
  SupervisedModelTraining____mars_gym_model_b____291ff27fbc )

# Training

## Submission

for u in "${models[@]}"
do
  echo ""
  echo "Eval $u"

  PYTHONPATH="." luigi --module evaluation EvaluationTask \
  --model-task-class "mars_gym.simulation.training.SupervisedModelTraining" \
  --model-task-id $u \
  --file "/media/workspace/booking_challenge/output/booking/dataset/test_0.1_10.csv"  \
  --local-scheduler

done