

Experimentos (base:0.51797)

* Checkin como modelo de tempo no emb                           (0.511420)
* Balancear as duas saídas do modelo loss  with 0.8             (0.52337)
* Balancear as duas saídas do modelo loss (mesmo MLP final)     (0.51917)
* Remover usuario emb de da RNN                                 (0.5062)
* Remover usuario emb de da End MLP e deixar RNN                (0.51825)
* Remover usuario emb all                                       (0.5099903)
* normalizar c_t                                                (0.52161)
* normalizar c_t depois de uma MLP                              (0.52023)
* normalizar c_t depois de uma MLP Tanh                         (0.48705)
* MLP Tanh sem normalizar                                       (0.49836)
* mask PAD zero rnn                                             (0.50569)
* mask PAD zero rnn before att                                  (0.504360)
* without mask PAD                                              (0.520372)
* Weight decay Geral (1e-2)                                     (0.52697)
* L2 Regularization no Emb  (1e-2)                              (0.5199)
* L2 Regularization no Emb and NLP (1e-2)                       (0.485948)
* L2 Regularization no Emb and Norm  (1e-2) 
* L2 Regularization no Emb and NLP with norm (1e-2) 
* Tying Word Classify                                           (0.50237)
* Usar focal loss nas duas saidas                               (0.52761)
* Pegar a loss apenas de quem é vizinho.                        (0.447233 - funciona bem no treino mas não na validação)
* Label Smoth (0.5357851)

* User with 100D and normalized
* Predict Neighbors to