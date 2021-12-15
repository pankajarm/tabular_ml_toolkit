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
from xgboost import XGBRegressor
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

#### Just point tmlt in the direction of your data, let it know what are idx and target columns in your tabular data and what kind of problem type you are trying to resolve

```
%%time
# tmlt
tmlt = TMLT().prepare_data(
    train_file_path= DIRECTORY_PATH+TRAIN_FILE,
    test_file_path= DIRECTORY_PATH+TEST_FILE,
    idx_col="Id", target="SalePrice",
    random_state=42,
    problem_type="regression")

# TMLT currently only supports below problem_type:

# "binary_classification"
# "multi_label_classification"
# "multi_class_classification"
# "regression"
```

    2021-12-09 21:33:30,292 INFO 8 cores found, model and data parallel processing should worked!
    2021-12-09 21:33:30,330 INFO DataFrame Memory usage decreased to 0.58 Mb (35.5% reduction)
    2021-12-09 21:33:30,365 INFO DataFrame Memory usage decreased to 0.58 Mb (34.8% reduction)
    2021-12-09 21:33:30,388 INFO Both Numerical & Categorical columns found, Preprocessing will done accordingly!


    CPU times: user 186 ms, sys: 50.4 ms, total: 237 ms
    Wall time: 258 ms


```
print(type(tmlt.dfl.X))
print(tmlt.dfl.X.shape)
print(type(tmlt.dfl.y))
print(tmlt.dfl.y.shape)
print(type(tmlt.dfl.X_test))
print(tmlt.dfl.X_test.shape)
```

    <class 'pandas.core.frame.DataFrame'>
    (1460, 79)
    <class 'numpy.ndarray'>
    (1460,)
    <class 'pandas.core.frame.DataFrame'>
    (1459, 79)


```
tmlt.dfl.X
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MSSubClass</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>MasVnrArea</th>
      <th>BsmtFinSF1</th>
      <th>BsmtFinSF2</th>
      <th>...</th>
      <th>GarageCond</th>
      <th>PavedDrive</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
      <th>Neighborhood</th>
      <th>Exterior1st</th>
      <th>Exterior2nd</th>
    </tr>
    <tr>
      <th>Id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>60</td>
      <td>65.0</td>
      <td>8450</td>
      <td>7</td>
      <td>5</td>
      <td>2003</td>
      <td>2003</td>
      <td>196.0</td>
      <td>706</td>
      <td>0</td>
      <td>...</td>
      <td>TA</td>
      <td>Y</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>WD</td>
      <td>Normal</td>
      <td>CollgCr</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20</td>
      <td>80.0</td>
      <td>9600</td>
      <td>6</td>
      <td>8</td>
      <td>1976</td>
      <td>1976</td>
      <td>0.0</td>
      <td>978</td>
      <td>0</td>
      <td>...</td>
      <td>TA</td>
      <td>Y</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>WD</td>
      <td>Normal</td>
      <td>Veenker</td>
      <td>MetalSd</td>
      <td>MetalSd</td>
    </tr>
    <tr>
      <th>3</th>
      <td>60</td>
      <td>68.0</td>
      <td>11250</td>
      <td>7</td>
      <td>5</td>
      <td>2001</td>
      <td>2002</td>
      <td>162.0</td>
      <td>486</td>
      <td>0</td>
      <td>...</td>
      <td>TA</td>
      <td>Y</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>WD</td>
      <td>Normal</td>
      <td>CollgCr</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
    </tr>
    <tr>
      <th>4</th>
      <td>70</td>
      <td>60.0</td>
      <td>9550</td>
      <td>7</td>
      <td>5</td>
      <td>1915</td>
      <td>1970</td>
      <td>0.0</td>
      <td>216</td>
      <td>0</td>
      <td>...</td>
      <td>TA</td>
      <td>Y</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>WD</td>
      <td>Abnorml</td>
      <td>Crawfor</td>
      <td>Wd Sdng</td>
      <td>Wd Shng</td>
    </tr>
    <tr>
      <th>5</th>
      <td>60</td>
      <td>84.0</td>
      <td>14260</td>
      <td>8</td>
      <td>5</td>
      <td>2000</td>
      <td>2000</td>
      <td>350.0</td>
      <td>655</td>
      <td>0</td>
      <td>...</td>
      <td>TA</td>
      <td>Y</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>WD</td>
      <td>Normal</td>
      <td>NoRidge</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1456</th>
      <td>60</td>
      <td>62.0</td>
      <td>7917</td>
      <td>6</td>
      <td>5</td>
      <td>1999</td>
      <td>2000</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>TA</td>
      <td>Y</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>WD</td>
      <td>Normal</td>
      <td>Gilbert</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
    </tr>
    <tr>
      <th>1457</th>
      <td>20</td>
      <td>85.0</td>
      <td>13175</td>
      <td>6</td>
      <td>6</td>
      <td>1978</td>
      <td>1988</td>
      <td>119.0</td>
      <td>790</td>
      <td>163</td>
      <td>...</td>
      <td>TA</td>
      <td>Y</td>
      <td>NaN</td>
      <td>MnPrv</td>
      <td>NaN</td>
      <td>WD</td>
      <td>Normal</td>
      <td>NWAmes</td>
      <td>Plywood</td>
      <td>Plywood</td>
    </tr>
    <tr>
      <th>1458</th>
      <td>70</td>
      <td>66.0</td>
      <td>9042</td>
      <td>7</td>
      <td>9</td>
      <td>1941</td>
      <td>2006</td>
      <td>0.0</td>
      <td>275</td>
      <td>0</td>
      <td>...</td>
      <td>TA</td>
      <td>Y</td>
      <td>NaN</td>
      <td>GdPrv</td>
      <td>Shed</td>
      <td>WD</td>
      <td>Normal</td>
      <td>Crawfor</td>
      <td>CemntBd</td>
      <td>CmentBd</td>
    </tr>
    <tr>
      <th>1459</th>
      <td>20</td>
      <td>68.0</td>
      <td>9717</td>
      <td>5</td>
      <td>6</td>
      <td>1950</td>
      <td>1996</td>
      <td>0.0</td>
      <td>49</td>
      <td>1029</td>
      <td>...</td>
      <td>TA</td>
      <td>Y</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>WD</td>
      <td>Normal</td>
      <td>NAmes</td>
      <td>MetalSd</td>
      <td>MetalSd</td>
    </tr>
    <tr>
      <th>1460</th>
      <td>20</td>
      <td>75.0</td>
      <td>9937</td>
      <td>5</td>
      <td>6</td>
      <td>1965</td>
      <td>1965</td>
      <td>0.0</td>
      <td>830</td>
      <td>290</td>
      <td>...</td>
      <td>TA</td>
      <td>Y</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>WD</td>
      <td>Normal</td>
      <td>Edwards</td>
      <td>HdBoard</td>
      <td>HdBoard</td>
    </tr>
  </tbody>
</table>
<p>1460 rows × 79 columns</p>
</div>



### Training

##### create train valid dataframes for quick preprocessing and training

```
%%time
# create train, valid split to evaulate model on valid dataset
X_train, X_valid,  y_train, y_valid =  tmlt.dfl.create_train_valid(valid_size=0.2)
```

    CPU times: user 4.67 ms, sys: 1.66 ms, total: 6.34 ms
    Wall time: 4.95 ms


```
print(X_train.shape)
print(y_train.shape)
print(X_valid.shape)
print(y_valid.shape)
```

    (1168, 79)
    (1168,)
    (292, 79)
    (292,)


```
# X_train.columns.to_list()
```

##### Now PreProcess X_train, X_valid

NOTE: Preprocessing gives back numpy arrays for pandas dataframe

```
%%time
X_train_np,  X_valid_np = tmlt.pp_fit_transform(X_train, X_valid)
```

    CPU times: user 29.8 ms, sys: 2.95 ms, total: 32.8 ms
    Wall time: 31.9 ms


```
print(type(X_train_np))
print(X_train_np.shape)
# print(X_train_np)
print(type(X_valid_np))
print(X_valid_np.shape)
# print(X_valid_np)
print(type(y_valid))
print(type(y_train))
```

    <class 'numpy.ndarray'>
    (1168, 302)
    <class 'numpy.ndarray'>
    (292, 302)
    <class 'numpy.ndarray'>
    <class 'numpy.ndarray'>


#### Training

##### Create a base xgb classifier model with your best guess params

```
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

```
%%time
# Now do model training
xgb_model.fit(X_train_np, y_train,
              verbose=False,
              #detect & avoid overfitting
              eval_set=[(X_train_np, y_train), (X_valid_np, y_valid)],
              eval_metric="mae",
              early_stopping_rounds=300
             )

#predict
preds = xgb_model.predict(X_valid_np)
print('X_valid MAE:', mean_absolute_error(y_valid, preds))
```

    X_valid MAE: 15915.75480254709
    CPU times: user 4.48 s, sys: 158 ms, total: 4.64 s
    Wall time: 778 ms


In background `prepare_data` method loads your input data into Pandas DataFrame, seprates X(features) and y(target), preprocess all numerical and categorical type data found in these DataFrames using scikit-learn pipelines. Then it bundle preprocessor and data return a TMLT object, this class instance has dataframeloader, preprocessor instances.

The `create_train_valid` method use valid_size to split X(features) into X_train, y_train, X_valid and y_valid DataFrames, so you can call fit methods on X_train and y_train and predict methods on X_valid or X_test.

Please check detail documentation and source code for more details.

*NOTE: If you want to customize data and preprocessing steps you can do so by using `DataFrameLoader` and `PreProessor` classes. Check detail documentations for these classes for more options.*



#### To see more clear picture of model performance, Let's do a quick Cross Validation on our Pipeline

##### Make sure to PreProcess the data

```
%%time
X_np, X_test_np = tmlt.pp_fit_transform(tmlt.dfl.X, tmlt.dfl.X_test)
y_np = tmlt.dfl.y
```

    CPU times: user 323 ms, sys: 37.8 ms, total: 361 ms
    Wall time: 65.9 ms


```
%%time
# Now do cross_validation
scores = tmlt.do_cross_validation(X_np, y_np, xgb_model, scoring='neg_mean_absolute_error', cv=5)

print("scores:", scores)
print("Average MAE score:", scores.mean())
```

    scores: [15733.51983893 16386.18366064 16648.82777718 14571.39875856
     17295.16245719]
    Average MAE score: 16127.018498501711
    CPU times: user 190 ms, sys: 101 ms, total: 291 ms
    Wall time: 5.01 s


*MAE did came out slightly bad with cross validation*

*Let's see if we can improve our cross validation score with hyperparams tunning*

**We are using optuna based hyperparameter search here!**

**TMLT has inbuilt xgb optuna optimization helper method!**

```
# **Just make sure to supply an output directory path so hyperparameter search is saved**
study = tmlt.do_xgb_optuna_optimization(optuna_db_path=OUTPUT_PATH, opt_timeout=60)
print(study.best_trial)
```

    2021-12-09 21:33:36,363 INFO Optimization Direction is: minimize
    [32m[I 2021-12-09 21:33:36,451][0m Using an existing study with name 'tmlt_autoxgb' instead of creating a new one.[0m
    2021-12-09 21:33:36,726 INFO Training Started!


    [21:33:36] WARNING: /Users/travis/build/dmlc/xgboost/src/learner.cc:541: 
    Parameters: { early_stopping_rounds, eval_set } might not be used.
    
      This may not be accurate due to some parameters are only used in language bindings but
      passed down to XGBoost core.  Or some parameters are not used but slip through this
      verification. Please open an issue if you find above cases.
    
    


    2021-12-09 21:34:11,305 INFO Training Ended!
    2021-12-09 21:34:11,344 INFO mean_absolute_error: 14927.047182684075
    2021-12-09 21:34:11,345 INFO mean_squared_error: 808477254.3879791
    [32m[I 2021-12-09 21:34:11,408][0m Trial 26 finished with value: 808477254.3879791 and parameters: {'learning_rate': 0.021675111991183195, 'n_estimators': 7000, 'reg_lambda': 8.715919175070972e-05, 'reg_alpha': 1.879676344414822e-07, 'subsample': 0.4092274050958187, 'colsample_bytree': 0.7122518032823463, 'max_depth': 8, 'early_stopping_rounds': 284, 'tree_method': 'exact', 'booster': 'gbtree', 'gamma': 0.03865572813384841, 'grow_policy': 'depthwise'}. Best is trial 0 with value: 607032267.1056623.[0m
    2021-12-09 21:34:11,680 INFO Training Started!


    [21:34:11] WARNING: /Users/travis/build/dmlc/xgboost/src/learner.cc:541: 
    Parameters: { early_stopping_rounds, eval_set } might not be used.
    
      This may not be accurate due to some parameters are only used in language bindings but
      passed down to XGBoost core.  Or some parameters are not used but slip through this
      verification. Please open an issue if you find above cases.
    
    


    2021-12-09 21:34:40,866 INFO Training Ended!
    2021-12-09 21:34:40,905 INFO mean_absolute_error: 15264.263217037671
    2021-12-09 21:34:40,905 INFO mean_squared_error: 654883443.0486244
    [32m[I 2021-12-09 21:34:40,956][0m Trial 27 finished with value: 654883443.0486244 and parameters: {'learning_rate': 0.04965238838358245, 'n_estimators': 20000, 'reg_lambda': 0.6817845210980734, 'reg_alpha': 0.00017829240858671677, 'subsample': 0.7881206302407813, 'colsample_bytree': 0.5948264911895165, 'max_depth': 3, 'early_stopping_rounds': 359, 'tree_method': 'hist', 'booster': 'gbtree', 'gamma': 0.0008403183968780732, 'grow_policy': 'lossguide'}. Best is trial 0 with value: 607032267.1056623.[0m


    FrozenTrial(number=0, values=[607032267.1056623], datetime_start=datetime.datetime(2021, 12, 6, 20, 42, 39, 63725), datetime_complete=datetime.datetime(2021, 12, 6, 20, 42, 55, 16274), params={'booster': 'gbtree', 'colsample_bytree': 0.8467533640596729, 'early_stopping_rounds': 156, 'gamma': 0.048829460890126776, 'grow_policy': 'lossguide', 'learning_rate': 0.14978041444389834, 'max_depth': 4, 'n_estimators': 7000, 'reg_alpha': 4.069576449804004e-05, 'reg_lambda': 0.00014406316350951595, 'subsample': 0.4839769602908782, 'tree_method': 'hist'}, distributions={'booster': CategoricalDistribution(choices=('gbtree', 'gblinear')), 'colsample_bytree': UniformDistribution(high=1.0, low=0.1), 'early_stopping_rounds': IntUniformDistribution(high=500, low=100, step=1), 'gamma': LogUniformDistribution(high=1.0, low=1e-08), 'grow_policy': CategoricalDistribution(choices=('depthwise', 'lossguide')), 'learning_rate': LogUniformDistribution(high=0.25, low=0.01), 'max_depth': IntUniformDistribution(high=9, low=1, step=1), 'n_estimators': CategoricalDistribution(choices=(7000, 15000, 20000)), 'reg_alpha': LogUniformDistribution(high=100.0, low=1e-08), 'reg_lambda': LogUniformDistribution(high=100.0, low=1e-08), 'subsample': UniformDistribution(high=1.0, low=0.1), 'tree_method': CategoricalDistribution(choices=('exact', 'approx', 'hist'))}, user_attrs={}, system_attrs={}, intermediate_values={}, trial_id=1, state=TrialState.COMPLETE, value=None)


#### Let's use our newly found best params to update the model on sklearn pipeline

```
xgb_params.update(study.best_trial.params)
print("xgb_params", xgb_params)
updated_xgb_model = XGBRegressor(**xgb_params)
```

    xgb_params {'learning_rate': 0.14978041444389834, 'use_label_encoder': False, 'eval_metric': 'rmse', 'random_state': 42, 'booster': 'gbtree', 'colsample_bytree': 0.8467533640596729, 'early_stopping_rounds': 156, 'gamma': 0.048829460890126776, 'grow_policy': 'lossguide', 'max_depth': 4, 'n_estimators': 7000, 'reg_alpha': 4.069576449804004e-05, 'reg_lambda': 0.00014406316350951595, 'subsample': 0.4839769602908782, 'tree_method': 'hist'}


#### Now, Let's use 5 K-Fold Training on this Updated XGB model with best params found from Optuna search

```
# # k-fold training
# xgb_model_metrics_score, xgb_model_test_preds = tmlt.do_kfold_training(X_np, y_np, n_splits=5, model=xgb_model, test_preds_metric=mean_absolute_error)
```

```
%%time
# k-fold training
xgb_model_metrics_score, xgb_model_test_preds = tmlt.do_kfold_training(X_np, y_np, X_test=X_test_np, n_splits=5, model=updated_xgb_model)
```

    2021-12-09 21:34:40,989 INFO  model class:<class 'xgboost.sklearn.XGBRegressor'>
    /Users/pankajmathur/anaconda3/envs/nbdev_env/lib/python3.9/site-packages/sklearn/model_selection/_split.py:676: UserWarning: The least populated class in y has only 1 members, which is less than n_splits=5.
      warnings.warn(
    2021-12-09 21:34:41,000 INFO Training Started!


    [21:34:41] WARNING: /Users/travis/build/dmlc/xgboost/src/learner.cc:541: 
    Parameters: { early_stopping_rounds } might not be used.
    
      This may not be accurate due to some parameters are only used in language bindings but
      passed down to XGBoost core.  Or some parameters are not used but slip through this
      verification. Please open an issue if you find above cases.
    
    


    2021-12-09 21:34:58,839 INFO Training Finished!
    2021-12-09 21:34:58,840 INFO Predicting Val Score!
    2021-12-09 21:34:58,848 INFO fold: 1 mean_absolute_error : 20715.247458261987
    2021-12-09 21:34:58,848 INFO fold: 1 mean_squared_error : 2957095332.886548
    2021-12-09 21:34:58,849 INFO Predicting Test Scores!
    2021-12-09 21:34:58,875 INFO Training Started!


    [21:34:58] WARNING: /Users/travis/build/dmlc/xgboost/src/learner.cc:541: 
    Parameters: { early_stopping_rounds } might not be used.
    
      This may not be accurate due to some parameters are only used in language bindings but
      passed down to XGBoost core.  Or some parameters are not used but slip through this
      verification. Please open an issue if you find above cases.
    
    


    2021-12-09 21:35:17,029 INFO Training Finished!
    2021-12-09 21:35:17,030 INFO Predicting Val Score!
    2021-12-09 21:35:17,040 INFO fold: 2 mean_absolute_error : 16294.310145547945
    2021-12-09 21:35:17,040 INFO fold: 2 mean_squared_error : 830363243.5137496
    2021-12-09 21:35:17,041 INFO Predicting Test Scores!
    2021-12-09 21:35:17,071 INFO Training Started!


    [21:35:17] WARNING: /Users/travis/build/dmlc/xgboost/src/learner.cc:541: 
    Parameters: { early_stopping_rounds } might not be used.
    
      This may not be accurate due to some parameters are only used in language bindings but
      passed down to XGBoost core.  Or some parameters are not used but slip through this
      verification. Please open an issue if you find above cases.
    
    


    2021-12-09 21:35:34,190 INFO Training Finished!
    2021-12-09 21:35:34,191 INFO Predicting Val Score!
    2021-12-09 21:35:34,199 INFO fold: 3 mean_absolute_error : 16816.43455693493
    2021-12-09 21:35:34,200 INFO fold: 3 mean_squared_error : 710149427.4676694
    2021-12-09 21:35:34,200 INFO Predicting Test Scores!
    2021-12-09 21:35:34,230 INFO Training Started!


    [21:35:34] WARNING: /Users/travis/build/dmlc/xgboost/src/learner.cc:541: 
    Parameters: { early_stopping_rounds } might not be used.
    
      This may not be accurate due to some parameters are only used in language bindings but
      passed down to XGBoost core.  Or some parameters are not used but slip through this
      verification. Please open an issue if you find above cases.
    
    


    2021-12-09 21:35:51,177 INFO Training Finished!
    2021-12-09 21:35:51,177 INFO Predicting Val Score!
    2021-12-09 21:35:51,185 INFO fold: 4 mean_absolute_error : 15824.842572773972
    2021-12-09 21:35:51,186 INFO fold: 4 mean_squared_error : 523051432.0483315
    2021-12-09 21:35:51,186 INFO Predicting Test Scores!
    2021-12-09 21:35:51,218 INFO Training Started!


    [21:35:51] WARNING: /Users/travis/build/dmlc/xgboost/src/learner.cc:541: 
    Parameters: { early_stopping_rounds } might not be used.
    
      This may not be accurate due to some parameters are only used in language bindings but
      passed down to XGBoost core.  Or some parameters are not used but slip through this
      verification. Please open an issue if you find above cases.
    
    


    2021-12-09 21:36:08,860 INFO Training Finished!
    2021-12-09 21:36:08,861 INFO Predicting Val Score!
    2021-12-09 21:36:08,869 INFO fold: 5 mean_absolute_error : 16750.133160316782
    2021-12-09 21:36:08,870 INFO fold: 5 mean_squared_error : 843596693.0681775
    2021-12-09 21:36:08,870 INFO Predicting Test Scores!
    2021-12-09 21:36:08,901 INFO  Mean Metrics Results from all Folds are: {'mean_absolute_error': 17280.193578767125, 'mean_squared_error': 1172851225.7968953}


    CPU times: user 10min 53s, sys: 4.01 s, total: 10min 57s
    Wall time: 1min 27s


```
# predict on test dataset
if xgb_model_test_preds is not None:
    print(xgb_model_test_preds.shape)
```

    (1459,)



##### You can even improve metrics score further by running Optuna search for longer time or rerunning the study, check documentation for more details

```
#fin
```
