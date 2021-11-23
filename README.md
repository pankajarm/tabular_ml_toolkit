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
```

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

```
# predict on test dataset
if xgb_model_test_preds is not None:
    print(xgb_model_test_preds.shape)
```


##### You can even improve metrics score further by running Optuna search for longer time or rerunning the study, check documentation for more details
