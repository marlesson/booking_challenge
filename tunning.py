import luigi
from mars_gym.simulation.training import SupervisedModelTraining
import numpy as np

# PYTHONPATH="."  luigi  --module tunning ModelTunning --local-scheduler --seed 12
class ModelTunning(luigi.WrapperTask):
  '''
  https://luigi.readthedocs.io/en/stable/luigi_patterns.html
  '''
  
  seed: int = luigi.IntParameter(default=42)
  
  experiments: int = luigi.IntParameter(default=100)

  def requires(self):
    random_state = np.random.RandomState(self.seed)

    tasks = []

    _n_factors = [10, 50, 100, 200]

    _hidden_size  = [10, 100, 300, 500]

    _n_layers     = [1, 2, 4, 6]

    _hist_size    = [10]

    _weight_decay = [0, 1e-5, 1e-3, 1e-2]

    _dropout   = [0, 0.3, 0.6, 0.8]

    _learning_rate = [0.001]

    _batch_size = [64, 128, 256]

    _balance_loss = [1, 0.9, 0.8, 0.6, 0.5]

    _epsilon   = [0.1, 0.2, 0.3, 0.5]

    _alpha     = [1, 2, 5, 10]

    _gamma     = [2, 3, 10, 100]

    _optimizer = ['radam']

    # emb_path = "/media/workspace/triplet_session/output/mercado_livre/assets/mercadolivre-100d.bin"

    for i in range(self.experiments): 
      n_factors     = int(random_state.choice(_n_factors))
      hidden_size   = int(random_state.choice(_hidden_size))
      n_layers      = int(random_state.choice(_n_layers))
      dropout       = float(random_state.choice(_dropout))
      
      weight_decay  = float(random_state.choice(_weight_decay))
      hist_size      = int(random_state.choice(_hist_size))
      learning_rate = float(random_state.choice(_learning_rate))
      batch_size = int(random_state.choice(_batch_size))
      balance_loss = float(random_state.choice(_balance_loss))
      epsilon= float(random_state.choice(_epsilon))
      alpha= float(random_state.choice(_alpha))
      gamma= float(random_state.choice(_gamma))
      optimizer = str(random_state.choice(_optimizer))


      job = SupervisedModelTraining(
            project="config.conf1_rnn",
            recommender_module_class="model.NARMModel",
            recommender_extra_params={
              "n_factors": n_factors, 
              "hidden_size": hidden_size, 
              "n_layers": n_layers, 
              "dropout": dropout, 
              "n_user_features": 10,
              "from_index_mapping": False,
              "path_item_embedding": False, 
              "freeze_embedding": False},
            data_frames_preparation_extra_params={
              "test_split": 0.1, 
              "window_trip": hist_size,
              "user_features_file": "all_user_features_10.csv",
              "column_stratification": "user_id",
              "filter_last_step": True,
              "balance_sample_step": 200000,
              "filter_trip_size": 0
            },
            test_size=0.0,
            val_size=0.1,
            early_stopping_min_delta=0.0001,
            early_stopping_patience=10,    
            learning_rate=learning_rate,
            metrics=["loss", "top_k_acc", "top_k_acc2"],
            batch_size=batch_size,
            optimizer=optimizer,
            optimizer_params={
              "weight_decay": weight_decay
            },
            loss_function="ce",
            loss_function_class="loss.FocalLoss",
            loss_function_params={
                "alpha":alpha,
                "gamma":gamma,
                "c": balance_loss,
                "epsilon": epsilon
            },
            epochs=500,
            monitor_metric='val_top_k_acc',
            monitor_mode='max',
            generator_workers=10
          )      

      yield job


if __name__ == '__main__':
  print("..........")
  #MLTransformerModelExperimentRuns().run()  
  ModelTunning().run()
