===== Ridge =====
2026-02-23 11:43:28,946  INFO  CV  MAE=0.42±0.01  RMSE=1.13±0.03  R2=0.2714±0.0123
2026-02-23 11:43:28,950  INFO  Fitting on full training set …
2026-02-23 11:43:29,349  INFO  Val  MAE=0.41  RMSE=1.15  R2=0.2075  MAPE=524180185205002.8125
2026/02/23 11:43:29 WARNING mlflow.models.model: `artifact_path` is deprecated. Please use `name` instead.
2026/02/23 11:43:29 WARNING mlflow.sklearn: Saving scikit-learn models in the pickle or cloudpickle format requires exercising caution because these formats rely on Python's object serialization mechanism, which can execute arbitrary code during deserialization. The recommended safe alternative is the 'skops' format. For more information, see: https://scikit-learn.org/stable/model_persistence.html
2026-02-23 11:43:31,418  INFO  Run 8dfe27aef80f47efa95403a4b21bc2d9 finished.  Model saved to ../models/Ridge.pkl.
2026-02-23 11:43:31,451  INFO    MAE    0.4128
2026-02-23 11:43:31,451  INFO    RMSE   1.1470
2026-02-23 11:43:31,452  INFO    R2     0.2075
2026-02-23 11:43:31,452  INFO    MAPE   524180185205002.8125
2026-02-23 11:43:31,470  INFO  Starting 5-fold cross-validation …

===== RandomForest =====
2026-02-23 11:45:22,535  INFO  CV  MAE=0.21±0.00  RMSE=0.64±0.03  R2=0.7666±0.0151
2026-02-23 11:45:22,537  INFO  Fitting on full training set …
2026-02-23 11:45:31,940  INFO  Val  MAE=0.20  RMSE=0.60  R2=0.7835  MAPE=139409572944421.2188
2026/02/23 11:45:32 WARNING mlflow.models.model: `artifact_path` is deprecated. Please use `name` instead.
2026/02/23 11:45:32 WARNING mlflow.sklearn: Saving scikit-learn models in the pickle or cloudpickle format requires exercising caution because these formats rely on Python's object serialization mechanism, which can execute arbitrary code during deserialization. The recommended safe alternative is the 'skops' format. For more information, see: https://scikit-learn.org/stable/model_persistence.html
2026-02-23 11:45:34,393  INFO  Run c61003aa99054087901285b7993d92ba finished.  Model saved to ../models/RandomForest.pkl.
2026-02-23 11:45:34,478  INFO    MAE    0.1986
2026-02-23 11:45:34,479  INFO    RMSE   0.5995
2026-02-23 11:45:34,479  INFO    R2     0.7835
2026-02-23 11:45:34,479  INFO    MAPE   139409572944421.2188
2026-02-23 11:45:34,508  INFO  Starting 5-fold cross-validation …

===== XGBoost =====
2026-02-23 11:45:41,963  INFO  CV  MAE=0.20±0.00  RMSE=0.53±0.03  R2=0.8363±0.0177
2026-02-23 11:45:41,965  INFO  Fitting on full training set …
2026-02-23 11:45:43,728  INFO  Val  MAE=0.19  RMSE=0.50  R2=0.8489  MAPE=116087022654501.5625
2026/02/23 11:45:43 WARNING mlflow.models.model: `artifact_path` is deprecated. Please use `name` instead.
2026/02/23 11:45:43 WARNING mlflow.sklearn: Saving scikit-learn models in the pickle or cloudpickle format requires exercising caution because these formats rely on Python's object serialization mechanism, which can execute arbitrary code during deserialization. The recommended safe alternative is the 'skops' format. For more information, see: https://scikit-learn.org/stable/model_persistence.html
2026-02-23 11:45:45,627  INFO  Run b447ed5da88c45ff856b181c47988888 finished.  Model saved to ../models/XGBoost.pkl.
2026-02-23 11:45:45,666  INFO    MAE    0.1885
2026-02-23 11:45:45,667  INFO    RMSE   0.5009
2026-02-23 11:45:45,667  INFO    R2     0.8489
2026-02-23 11:45:45,667  INFO    MAPE   116087022654501.5625
2026-02-23 11:45:45,688  INFO  Starting 5-fold cross-validation …

===== LightGBM =====
2026-02-23 11:47:01,723  INFO  CV  MAE=0.19±0.00  RMSE=0.53±0.03  R2=0.8392±0.0185
2026-02-23 11:47:01,724  INFO  Fitting on full training set …
2026-02-23 11:47:08,471  INFO  Val  MAE=0.18  RMSE=0.52  R2=0.8374  MAPE=118639987895993.3281
2026/02/23 11:47:08 WARNING mlflow.models.model: `artifact_path` is deprecated. Please use `name` instead.
2026/02/23 11:47:08 WARNING mlflow.sklearn: Saving scikit-learn models in the pickle or cloudpickle format requires exercising caution because these formats rely on Python's object serialization mechanism, which can execute arbitrary code during deserialization. The recommended safe alternative is the 'skops' format. For more information, see: https://scikit-learn.org/stable/model_persistence.html
2026-02-23 11:47:10,651  INFO  Run bb3608284e734616b1a885df316ac869 finished.  Model saved to ../models/LightGBM.pkl.
2026-02-23 11:47:10,731  INFO    MAE    0.1799
2026-02-23 11:47:10,731  INFO    RMSE   0.5196
2026-02-23 11:47:10,731  INFO    R2     0.8374
2026-02-23 11:47:10,731  INFO    MAPE   118639987895993.3281

--- Validation Results Summary ---
              MAE  RMSE   R2               MAPE
LightGBM     0.18  0.52 0.84 118639987895993.33
XGBoost      0.19  0.50 0.85 116087022654501.56
RandomForest 0.20  0.60 0.78 139409572944421.22
Ridge        0.41  1.15 0.21 524180185205002.81
Best model: LightGBM