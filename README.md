# Tabular ML Toolkit
> A super fast helper library to jumpstart your machine learning project based on tabular or structured data.


## Install

`pip install -U tabular_ml_toolkit`

## How to use

Start with your favorite model and then just simply create MLPipeline with one API.

*For example, Here we are using RandomForestRegressor from Scikit-Learn, on  [Melbourne Home Sale price data](https://www.kaggle.com/estrotococo/home-data-for-ml-course)*


*No need to install scikit-learn as it comes preinstall with Tabular_ML_Toolkit*

```
from tabular_ml_toolkit.MLPipeline import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np
import time
```

```
# Dataset file names and Paths
DIRECTORY_PATH = "https://raw.githubusercontent.com/psmathur/tabular_ml_toolkit/master/input/home_data/"
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
SAMPLE_SUB_FILE = "sample_submission.csv"
```

```
# create scikit-learn ml model
scikit_model = RandomForestRegressor(random_state=42)

# createm ml pipeline for scikit-learn model
tmlt = MLPipeline().prepare_data_for_training(
    train_file_path= DIRECTORY_PATH+TRAIN_FILE,
    test_file_path= DIRECTORY_PATH+TEST_FILE,
    idx_col="Id",
    target="SalePrice",
    model=scikit_model,
    random_state=42)

# visualize scikit-pipeline
# tmlt.spl
```

```
start = time.time()

# Now do cross_validation
scores = tmlt.do_cross_validation(cv=10, scoring='neg_mean_absolute_error')

end = time.time()
print("Cross Validation Time:", end - start)

print("scores:", scores)
print("Average MAE score:", scores.mean())
```

    Cross Validation Time: 14.697277069091797
    scores: [16871.87979452 18135.86726027 16032.48842466 19186.10719178
     19341.86143836 14970.44808219 15863.47123288 16053.91267123
     20180.36609589 17375.76856164]
    Average MAE score: 17401.217075342465


#### You can also use XGBoost model on same pipeline

*Just make sure to install XGBooost first depending upon your OS.*

*After that all steps remains same. Here is example using XGBRegressor with [Melbourne Home Sale price data](https://www.kaggle.com/estrotococo/home-data-for-ml-course)*

```
#!pip install -U xgboost
```

```
from xgboost import XGBRegressor
xgb_params = {
    'n_estimators':250,
    'learning_rate':0.05,
    'random_state':42,
    # for GPU
#     'tree_method': 'gpu_hist',
#     'predictor': 'gpu_predictor',
}

# create xgb model
xgb_model = XGBRegressor(**xgb_params)
```

```
# Make sure to update pipeline with xgb model
tmlt.update_model(xgb_model)

# visualize scikit-pipeline
# tmlt.spl
```

```
start = time.time()

# Now do cross_validation
scores = tmlt.do_cross_validation(cv=10, scoring='neg_mean_absolute_error')

end = time.time()
print("Cross Validation Time:", end - start)

print("scores:", scores)
print("Average MAE score:", scores.mean())
```

    Cross Validation Time: 6.754866123199463
    scores: [14655.50866866 15914.2911494  15265.50021404 17544.02900257
     18052.48084332 15120.70601455 14776.67005565 13291.54387842
     17425.94231592 16157.00203339]
    Average MAE score: 15820.36741759418


**XGB model looks more promising!**

In background `prepare_data_for_training` method loads your input data into Pandas DataFrame, seprates X(features) and y(target), 

Then it preprocess all numerical and categorical type data found in these dataframes.

Then it bundle preprocessed data with your given model and return an MLPipeline object which contains dataframeloader, preprocessor and scikit-learn pipeline.


Please see tutorials for more features from ML Tabular Toolkit.

#### Let's do Hyper Parameters Optimization and find the best params for XGB Model

 Let's give our Grid Search 2 minute time budget, Because you don't have eternity to wait for hyperparam tunning!

```
# let's do tune grid search for faster hyperparams tuning for data preprocessing and model

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer


# let's tune data preprocessing and model hyperparams
param_grid = {
    "preprocessor__num_cols__scaler": [StandardScaler(), MinMaxScaler()],
    "preprocessor__cat_cols__imputer": [SimpleImputer(strategy='constant'),
                                                 SimpleImputer(strategy='most_frequent')],
    'model__n_estimators': [500,1000],
    'model__learning_rate': [0.02,0.05],
    'model__max_depth': [5,10]
}

start = time.time()
# Now do tune grid search
tune_search = tmlt.do_tune_grid_search(param_grid=param_grid,
                                       cv=5,
                                       scoring='neg_mean_absolute_error',
                                      early_stopping=False,
                                      time_budget_s=120)
end = time.time()
print("Grid Search Time:", end - start)

print("Best params:")
print(tune_search.best_params_)

print(f"Internal CV Metrics score: {-1*(tune_search.best_score_):.3f}")
```

    /Users/pamathur/miniconda3/envs/nbdev_env/lib/python3.9/site-packages/tune_sklearn/tune_basesearch.py:400: UserWarning: max_iters is set > 1 but incremental/partial training is not enabled. To enable partial training, ensure the estimator has `partial_fit` or `warm_start` and set `early_stopping=True`. Automatically setting max_iters=1.
      warnings.warn(
    /Users/pamathur/miniconda3/envs/nbdev_env/lib/python3.9/site-packages/ray/tune/tune.py:368: UserWarning: The `loggers` argument is deprecated. Please pass the respective `LoggerCallback` classes to the `callbacks` argument instead. See https://docs.ray.io/en/latest/tune/api_docs/logging.html
      warnings.warn(


If you want to customize data and preprocessing steps you can do so by using `DataFrameLoader` and `PreProessor` classes. Please Check other Tutorials and detail documentations for these classes for more options. 

**Amazing our 5 Fold CV MAE has reduced to 15700.401 within 2 minutes by HyperParamss tunning!

If we can continue doing hyperparmas tunning, may be we can even do better, You can also try early_stopping, take that as challenge!

###### Let's use our newly found params for k-fold training and update preprocessor and model

##### Update PreProcessor on MLPipeline

```
pp_params = tmlt.get_preprocessor_best_params(tune_search)

# Update pipeline with updated preprocessor
tmlt.update_preprocessor(**pp_params)
tmlt.spl
```

##### Update Model on MLPipeline

```
xgb_params = tmlt.get_model_best_params(tune_search)

# create xgb ml model
xgb_model = XGBRegressor(**xgb_params)

# Update pipeline with xgb model
tmlt.update_model(xgb_model)
tmlt.spl
```

```
# k-fold training
xgb_model_k_fold, xgb_model_metrics_score = tmlt.do_k_fold_training(n_splits=10, metrics=mean_absolute_error)
print("mean metrics score:", np.mean(xgb_model_metrics_score))
```

**Yay, we have much better 10 K-Fold MAE**

```
# predict on test dataset which was given initially
xgb_model_preds = tmlt.do_k_fold_prediction(k_fold=xgb_model_k_fold)
print(xgb_model_preds.shape)
```
