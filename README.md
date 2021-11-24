# Getting Started Tutorial with TMLT (Tabular ML Toolkit)
> A tutorial on getting started with TMLT (Tabular ML Toolkit)


## Install

`pip install -U tabular_ml_toolkit`

## How to Best Use tabular_ml_toolkit

Start with your favorite model and then just simply create tmlt with one API

*For example, Here we are using XGBRegressor on  [Melbourne Home Sale price data](https://www.kaggle.com/estrotococo/home-data-for-ml-course)*

```
from tabular_ml_toolkit.tmlt import *
from sklearn.metrics import mean_absolute_error
import numpy as np

# Just to compare fit times
import time
```

    /Users/pankajmathur/anaconda3/envs/nbdev_env/lib/python3.9/site-packages/redis/connection.py:77: UserWarning: redis-py works best with hiredis. Please consider installing
      warnings.warn(msg)


```
# Dataset file names and Paths
DIRECTORY_PATH = "input/home_data/"
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
SAMPLE_SUB_FILE = "sample_submission.csv"
OUTPUT_PATH = "output/"
```

##### Create a base xgb classifier model with your best guess params

```
from xgboost import XGBRegressor
xgb_params = {
    'learning_rate':0.1,
    'use_label_encoder':False,
    'eval_metric':'rmse',
    'random_state':42,
    # for GPU
#     'tree_method': 'gpu_hist',
#     'predictor': 'gpu_predictor',
}
# create xgb ml model
xgb_model = XGBRegressor(**xgb_params)
```

##### Just point in the direction of your data, let tmlt know what are idx and target columns in your tabular data and what kind of problem type you are trying to resolve

```
# tmlt
tmlt = TMLT().prepare_data_for_training(
    train_file_path= DIRECTORY_PATH+TRAIN_FILE,
    test_file_path= DIRECTORY_PATH+TEST_FILE,
    idx_col="Id", target="SalePrice",
    model=xgb_model,
    random_state=42,
    problem_type="regression")

# TMLT currently only supports below problem_type:

# "binary_classification"
# "multi_label_classification"
# "multi_class_classification"
# "regression"
```

    2021-11-23 21:17:45,075 INFO 8 cores found, model and data parallel processing should worked!
    2021-11-23 21:17:45,112 INFO DataFrame Memory usage decreased to 0.58 Mb (35.5% reduction)
    2021-11-23 21:17:45,148 INFO DataFrame Memory usage decreased to 0.58 Mb (34.8% reduction)
    2021-11-23 21:17:45,172 INFO Both Numerical & Categorical columns found, Preprocessing will done accordingly!


```
# create train, valid split to evaulate model on valid dataset
tmlt.dfl.create_train_valid(valid_size=0.2)

start = time.time()
# Now fit
tmlt.spl.fit(tmlt.dfl.X_train, tmlt.dfl.y_train)
end = time.time()
print("Fit Time:", end - start)

#predict
preds = tmlt.spl.predict(tmlt.dfl.X_valid)
print('X_valid MAE:', mean_absolute_error(tmlt.dfl.y_valid, preds))
```

    Fit Time: 0.23742175102233887
    X_valid MAE: 15936.53249411387


In background `prepare_data_for_training` method loads your input data into Pandas DataFrame, seprates X(features) and y(target).

The `prepare_data_for_training` methods prepare X and y DataFrames, preprocess all numerical and categorical type data found in these DataFrames using scikit-learn pipelines. Then it bundle preprocessed data with your given model and return an MLPipeline object, this class instance has dataframeloader, preprocessor and scikit-lean pipeline instances.

The `create_train_valid` method use valid_size to split X(features) into X_train, y_train, X_valid and y_valid DataFrames, so you can call fit methods on X_train and y_train and predict methods on X_valid or X_test.


Please check detail documentation and source code for more details.

*NOTE: If you want to customize data and preprocessing steps you can do so by using `DataFrameLoader` and `PreProessor` classes. Check detail documentations for these classes for more options.*



#### To see more clear picture of model performance, Let's do a quick Cross Validation on our Pipeline

```
start = time.time()
# Now do cross_validation
scores = tmlt.do_cross_validation(cv=5, scoring='neg_mean_absolute_error')
end = time.time()
print("Cross Validation Time:", end - start)

print("scores:", scores)
print("Average MAE score:", scores.mean())
```

*MAE did came out slightly bad with cross validation*

*Let's see if we can improve our cross validation score with hyperparams tunning*

**we are using optuna based hyperparameter search here, make sure to supply a new directory path so search is saved**

```
study = tmlt.do_xgb_optuna_optimization(optuna_db_path=OUTPUT_PATH)
print(study.best_trial)
```

#### Let's use our newly found best params to update the model on sklearn pipeline

```
xgb_params.update(study.best_trial.params)
print("xgb_params", xgb_params)
xgb_model = XGBRegressor(**xgb_params)
tmlt.update_model(xgb_model)
tmlt.spl
```

#### Now, Let's use 5 K-Fold Training on this Updated XGB model with best params found from Optuna search

```
# k-fold training
xgb_model_metrics_score, xgb_model_test_preds = tmlt.do_kfold_training(n_splits=5, test_preds_metric=mean_absolute_error)
```

    /Users/pankajmathur/anaconda3/envs/nbdev_env/lib/python3.9/site-packages/sklearn/model_selection/_split.py:676: UserWarning: The least populated class in y has only 1 members, which is less than n_splits=5.
      warnings.warn(
    2021-11-23 21:17:59,130 INFO fold: 1 mean_absolute_error : 19024.810065282534
    2021-11-23 21:17:59,131 INFO fold: 1 mean_squared_error : 1751242762.4598274
    2021-11-23 21:17:59,132 INFO fold: 1 r2_score : 0.6977529569233543
    2021-11-23 21:17:59,474 INFO fold: 2 mean_absolute_error : 15865.40580854024
    2021-11-23 21:17:59,475 INFO fold: 2 mean_squared_error : 999468295.7551991
    2021-11-23 21:17:59,476 INFO fold: 2 r2_score : 0.8218766594977651
    2021-11-23 21:17:59,829 INFO fold: 3 mean_absolute_error : 16107.65785530822
    2021-11-23 21:17:59,830 INFO fold: 3 mean_squared_error : 800398659.0890251
    2021-11-23 21:17:59,831 INFO fold: 3 r2_score : 0.894853077195726
    2021-11-23 21:18:00,192 INFO fold: 4 mean_absolute_error : 15299.691553403254
    2021-11-23 21:18:00,193 INFO fold: 4 mean_squared_error : 566410890.4061956
    2021-11-23 21:18:00,193 INFO fold: 4 r2_score : 0.8984592069829627
    2021-11-23 21:18:00,559 INFO fold: 5 mean_absolute_error : 17103.836445847603
    2021-11-23 21:18:00,560 INFO fold: 5 mean_squared_error : 962383122.8550748
    2021-11-23 21:18:00,560 INFO fold: 5 r2_score : 0.8596639193010006
    2021-11-23 21:18:00,622 INFO  Mean Metrics Results from all Folds are: {'mean_absolute_error': 16680.28034567637, 'mean_squared_error': 1015980746.1130643, 'r2_score': 0.8345211639801617}


```
# predict on test dataset
if xgb_model_test_preds is not None:
    print(xgb_model_test_preds.shape)
```


##### You can even improve metrics score further by running Optuna search for longer time or rerunning the study, check documentation for more details
