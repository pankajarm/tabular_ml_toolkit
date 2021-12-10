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

    2021-12-06 20:40:13,230 INFO 8 cores found, model and data parallel processing should worked!
    2021-12-06 20:40:13,267 INFO DataFrame Memory usage decreased to 0.58 Mb (35.5% reduction)
    2021-12-06 20:40:13,305 INFO DataFrame Memory usage decreased to 0.58 Mb (34.8% reduction)
    2021-12-06 20:40:13,331 INFO Both Numerical & Categorical columns found, Preprocessing will done accordingly!


    CPU times: user 122 ms, sys: 16.6 ms, total: 139 ms
    Wall time: 138 ms


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

    CPU times: user 6.11 ms, sys: 1.88 ms, total: 7.99 ms
    Wall time: 6.28 ms


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

    CPU times: user 35.3 ms, sys: 3.43 ms, total: 38.7 ms
    Wall time: 37.3 ms


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
    CPU times: user 3.67 s, sys: 28.5 ms, total: 3.7 s
    Wall time: 484 ms


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

    CPU times: user 49.3 ms, sys: 3.5 ms, total: 52.8 ms
    Wall time: 52.2 ms


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
    CPU times: user 37.9 ms, sys: 59.7 ms, total: 97.6 ms
    Wall time: 3.3 s


*MAE did came out slightly bad with cross validation*

*Let's see if we can improve our cross validation score with hyperparams tunning*

**We are using optuna based hyperparameter search here!**

**TMLT has inbuilt xgb optuna optimization helper method!**

```
# **Just make sure to supply an output directory path so hyperparameter search is saved**
study = tmlt.do_xgb_optuna_optimization(optuna_db_path=OUTPUT_PATH, opt_timeout=360)
print(study.best_trial)
```

    2021-12-06 21:00:14,542 INFO Optimization Direction is: minimize
    [32m[I 2021-12-06 21:00:14,569][0m Using an existing study with name 'tmlt_autoxgb' instead of creating a new one.[0m
    2021-12-06 21:00:14,734 INFO Training Started!


    [21:00:14] WARNING: /Users/travis/build/dmlc/xgboost/src/learner.cc:541: 
    Parameters: { early_stopping_rounds, eval_set } might not be used.
    
      This may not be accurate due to some parameters are only used in language bindings but
      passed down to XGBoost core.  Or some parameters are not used but slip through this
      verification. Please open an issue if you find above cases.
    
    


    2021-12-06 21:00:35,108 INFO Training Ended!
    2021-12-06 21:00:35,152 INFO mean_absolute_error: 15009.969084439212
    2021-12-06 21:00:35,153 INFO mean_squared_error: 655351882.5350126
    2021-12-06 21:00:35,153 INFO r2_score: 0.914560102812714
    [32m[I 2021-12-06 21:00:35,197][0m Trial 11 finished with value: 655351882.5350126 and parameters: {'learning_rate': 0.010270418302430515, 'n_estimators': 7000, 'reg_lambda': 0.11208993634908683, 'reg_alpha': 1.2523750509850092e-08, 'subsample': 0.7654682326285074, 'colsample_bytree': 0.3202600058643555, 'max_depth': 6, 'early_stopping_rounds': 311, 'tree_method': 'approx', 'booster': 'gbtree', 'gamma': 3.0032310337571614e-08, 'grow_policy': 'lossguide'}. Best is trial 0 with value: 607032267.1056623.[0m
    2021-12-06 21:00:35,344 INFO Training Started!


    [21:00:35] WARNING: /Users/travis/build/dmlc/xgboost/src/learner.cc:541: 
    Parameters: { early_stopping_rounds, eval_set } might not be used.
    
      This may not be accurate due to some parameters are only used in language bindings but
      passed down to XGBoost core.  Or some parameters are not used but slip through this
      verification. Please open an issue if you find above cases.
    
    


    2021-12-06 21:00:57,999 INFO Training Ended!
    2021-12-06 21:00:58,043 INFO mean_absolute_error: 15042.68758026541
    2021-12-06 21:00:58,044 INFO mean_squared_error: 692181925.3312689
    2021-12-06 21:00:58,044 INFO r2_score: 0.9097584761541573
    [32m[I 2021-12-06 21:00:58,084][0m Trial 12 finished with value: 692181925.3312689 and parameters: {'learning_rate': 0.01031476516405966, 'n_estimators': 7000, 'reg_lambda': 0.06291237692238871, 'reg_alpha': 1.000769636493865e-08, 'subsample': 0.7266594563237431, 'colsample_bytree': 0.36116785096889154, 'max_depth': 6, 'early_stopping_rounds': 310, 'tree_method': 'approx', 'booster': 'gbtree', 'gamma': 1.0514451034987692e-08, 'grow_policy': 'lossguide'}. Best is trial 0 with value: 607032267.1056623.[0m
    2021-12-06 21:00:58,236 INFO Training Started!


    [21:00:58] WARNING: /Users/travis/build/dmlc/xgboost/src/learner.cc:541: 
    Parameters: { early_stopping_rounds, eval_set } might not be used.
    
      This may not be accurate due to some parameters are only used in language bindings but
      passed down to XGBoost core.  Or some parameters are not used but slip through this
      verification. Please open an issue if you find above cases.
    
    


    2021-12-06 21:01:16,142 INFO Training Ended!
    2021-12-06 21:01:16,188 INFO mean_absolute_error: 15168.345836900686
    2021-12-06 21:01:16,189 INFO mean_squared_error: 646714033.6624396
    2021-12-06 21:01:16,190 INFO r2_score: 0.9156862412114277
    [32m[I 2021-12-06 21:01:16,226][0m Trial 13 finished with value: 646714033.6624396 and parameters: {'learning_rate': 0.011755163749377023, 'n_estimators': 7000, 'reg_lambda': 0.3206755620341041, 'reg_alpha': 1.0765871018494073e-08, 'subsample': 0.701867431660848, 'colsample_bytree': 0.2745782478450856, 'max_depth': 6, 'early_stopping_rounds': 298, 'tree_method': 'approx', 'booster': 'gbtree', 'gamma': 3.1057087219579735e-08, 'grow_policy': 'lossguide'}. Best is trial 0 with value: 607032267.1056623.[0m
    2021-12-06 21:01:16,377 INFO Training Started!


    [21:01:16] WARNING: /Users/travis/build/dmlc/xgboost/src/learner.cc:541: 
    Parameters: { early_stopping_rounds, eval_set } might not be used.
    
      This may not be accurate due to some parameters are only used in language bindings but
      passed down to XGBoost core.  Or some parameters are not used but slip through this
      verification. Please open an issue if you find above cases.
    
    


    2021-12-06 21:01:30,116 INFO Training Ended!
    2021-12-06 21:01:30,134 INFO mean_absolute_error: 16569.805610552226
    2021-12-06 21:01:30,135 INFO mean_squared_error: 757096281.7109799
    2021-12-06 21:01:30,135 INFO r2_score: 0.9012954258709913
    [32m[I 2021-12-06 21:01:30,169][0m Trial 14 finished with value: 757096281.7109799 and parameters: {'learning_rate': 0.10138738039382836, 'n_estimators': 7000, 'reg_lambda': 0.2816742259414267, 'reg_alpha': 0.00016554411913750615, 'subsample': 0.6563278374163238, 'colsample_bytree': 0.6352242840674879, 'max_depth': 2, 'early_stopping_rounds': 227, 'tree_method': 'approx', 'booster': 'gbtree', 'gamma': 0.00620127368783362, 'grow_policy': 'lossguide'}. Best is trial 0 with value: 607032267.1056623.[0m
    2021-12-06 21:01:30,329 INFO Training Started!


    [21:01:30] WARNING: /Users/travis/build/dmlc/xgboost/src/learner.cc:541: 
    Parameters: { early_stopping_rounds, eval_set } might not be used.
    
      This may not be accurate due to some parameters are only used in language bindings but
      passed down to XGBoost core.  Or some parameters are not used but slip through this
      verification. Please open an issue if you find above cases.
    
    


    2021-12-06 21:02:07,556 INFO Training Ended!
    2021-12-06 21:02:07,599 INFO mean_absolute_error: 15009.079743685788
    2021-12-06 21:02:07,600 INFO mean_squared_error: 626148008.2534
    2021-12-06 21:02:07,600 INFO r2_score: 0.9183674864223249
    [32m[I 2021-12-06 21:02:07,638][0m Trial 15 finished with value: 626148008.2534 and parameters: {'learning_rate': 0.020642957404846188, 'n_estimators': 7000, 'reg_lambda': 1.7513064452033746, 'reg_alpha': 1.7170396831267702e-05, 'subsample': 0.6242954147289631, 'colsample_bytree': 0.8202870433196692, 'max_depth': 6, 'early_stopping_rounds': 195, 'tree_method': 'hist', 'booster': 'gbtree', 'gamma': 0.49885898603515555, 'grow_policy': 'lossguide'}. Best is trial 0 with value: 607032267.1056623.[0m
    2021-12-06 21:02:07,781 INFO Training Started!


    [21:02:07] WARNING: /Users/travis/build/dmlc/xgboost/src/learner.cc:541: 
    Parameters: { early_stopping_rounds, eval_set } might not be used.
    
      This may not be accurate due to some parameters are only used in language bindings but
      passed down to XGBoost core.  Or some parameters are not used but slip through this
      verification. Please open an issue if you find above cases.
    
    


    2021-12-06 21:02:19,643 INFO Training Ended!
    2021-12-06 21:02:19,665 INFO mean_absolute_error: 15922.791136023116
    2021-12-06 21:02:19,665 INFO mean_squared_error: 652899854.8858695
    2021-12-06 21:02:19,666 INFO r2_score: 0.9148797799141711
    [32m[I 2021-12-06 21:02:19,697][0m Trial 16 finished with value: 652899854.8858695 and parameters: {'learning_rate': 0.023785205795892477, 'n_estimators': 7000, 'reg_lambda': 2.3962635699611746e-06, 'reg_alpha': 1.7345589108628806e-05, 'subsample': 0.2253464414439063, 'colsample_bytree': 0.849342044824962, 'max_depth': 3, 'early_stopping_rounds': 190, 'tree_method': 'hist', 'booster': 'gbtree', 'gamma': 0.9605947921182305, 'grow_policy': 'lossguide'}. Best is trial 0 with value: 607032267.1056623.[0m
    2021-12-06 21:02:19,844 INFO Training Started!


    [21:02:19] WARNING: /Users/travis/build/dmlc/xgboost/src/learner.cc:541: 
    Parameters: { early_stopping_rounds, eval_set } might not be used.
    
      This may not be accurate due to some parameters are only used in language bindings but
      passed down to XGBoost core.  Or some parameters are not used but slip through this
      verification. Please open an issue if you find above cases.
    
    


    2021-12-06 21:03:22,928 INFO Training Ended!
    2021-12-06 21:03:23,056 INFO mean_absolute_error: 15691.25409353596
    2021-12-06 21:03:23,056 INFO mean_squared_error: 621107020.9367895
    2021-12-06 21:03:23,057 INFO r2_score: 0.9190246928018773
    [32m[I 2021-12-06 21:03:23,133][0m Trial 17 finished with value: 621107020.9367895 and parameters: {'learning_rate': 0.030318257026273988, 'n_estimators': 20000, 'reg_lambda': 3.575493353987708, 'reg_alpha': 0.0006892576471908247, 'subsample': 0.6074312166546988, 'colsample_bytree': 0.709262955470817, 'max_depth': 7, 'early_stopping_rounds': 185, 'tree_method': 'hist', 'booster': 'gbtree', 'gamma': 0.018834748728647963, 'grow_policy': 'depthwise'}. Best is trial 0 with value: 607032267.1056623.[0m
    2021-12-06 21:03:23,268 INFO Training Started!


    [21:03:23] WARNING: /Users/travis/build/dmlc/xgboost/src/learner.cc:541: 
    Parameters: { early_stopping_rounds, eval_set } might not be used.
    
      This may not be accurate due to some parameters are only used in language bindings but
      passed down to XGBoost core.  Or some parameters are not used but slip through this
      verification. Please open an issue if you find above cases.
    
    


    2021-12-06 21:04:29,536 INFO Training Ended!
    2021-12-06 21:04:29,693 INFO mean_absolute_error: 15795.27316994863
    2021-12-06 21:04:29,694 INFO mean_squared_error: 652118209.6186093
    2021-12-06 21:04:29,694 INFO r2_score: 0.9149816850021878
    [32m[I 2021-12-06 21:04:29,774][0m Trial 18 finished with value: 652118209.6186093 and parameters: {'learning_rate': 0.03271758718111387, 'n_estimators': 20000, 'reg_lambda': 0.005692270491178403, 'reg_alpha': 0.0023408373512426913, 'subsample': 0.8199008446138789, 'colsample_bytree': 0.6886538208392597, 'max_depth': 8, 'early_stopping_rounds': 177, 'tree_method': 'hist', 'booster': 'gbtree', 'gamma': 0.003797884287025332, 'grow_policy': 'depthwise'}. Best is trial 0 with value: 607032267.1056623.[0m
    2021-12-06 21:04:29,915 INFO Training Started!


    [21:04:29] WARNING: /Users/travis/build/dmlc/xgboost/src/learner.cc:541: 
    Parameters: { early_stopping_rounds, eval_set } might not be used.
    
      This may not be accurate due to some parameters are only used in language bindings but
      passed down to XGBoost core.  Or some parameters are not used but slip through this
      verification. Please open an issue if you find above cases.
    
    


    2021-12-06 21:04:58,518 INFO Training Ended!
    2021-12-06 21:04:58,603 INFO mean_absolute_error: 19889.809463291953
    2021-12-06 21:04:58,603 INFO mean_squared_error: 908790344.0741059
    2021-12-06 21:04:58,604 INFO r2_score: 0.8815186838830185
    [32m[I 2021-12-06 21:04:58,648][0m Trial 19 finished with value: 908790344.0741059 and parameters: {'learning_rate': 0.076049158907957, 'n_estimators': 20000, 'reg_lambda': 0.006228486140956925, 'reg_alpha': 90.60432556749008, 'subsample': 0.11043073454912561, 'colsample_bytree': 0.5759546613129808, 'max_depth': 9, 'early_stopping_rounds': 253, 'tree_method': 'hist', 'booster': 'gbtree', 'gamma': 0.0221527943168504, 'grow_policy': 'depthwise'}. Best is trial 0 with value: 607032267.1056623.[0m
    2021-12-06 21:04:58,789 INFO Training Started!


    [21:04:58] WARNING: /Users/travis/build/dmlc/xgboost/src/learner.cc:541: 
    Parameters: { early_stopping_rounds, eval_set } might not be used.
    
      This may not be accurate due to some parameters are only used in language bindings but
      passed down to XGBoost core.  Or some parameters are not used but slip through this
      verification. Please open an issue if you find above cases.
    
    


    2021-12-06 21:05:55,248 INFO Training Ended!
    2021-12-06 21:05:55,374 INFO mean_absolute_error: 19523.97050915026
    2021-12-06 21:05:55,375 INFO mean_squared_error: 891129659.8191046
    2021-12-06 21:05:55,375 INFO r2_score: 0.8838211523541055
    [32m[I 2021-12-06 21:05:55,446][0m Trial 20 finished with value: 891129659.8191046 and parameters: {'learning_rate': 0.14999053695281084, 'n_estimators': 20000, 'reg_lambda': 1.2704296846435246e-07, 'reg_alpha': 0.0008225883615747367, 'subsample': 0.3240920022885547, 'colsample_bytree': 0.9965396149951824, 'max_depth': 7, 'early_stopping_rounds': 106, 'tree_method': 'hist', 'booster': 'gbtree', 'gamma': 0.0002518451571799014, 'grow_policy': 'depthwise'}. Best is trial 0 with value: 607032267.1056623.[0m
    2021-12-06 21:05:55,585 INFO Training Started!


    [21:05:55] WARNING: /Users/travis/build/dmlc/xgboost/src/learner.cc:541: 
    Parameters: { early_stopping_rounds, eval_set } might not be used.
    
      This may not be accurate due to some parameters are only used in language bindings but
      passed down to XGBoost core.  Or some parameters are not used but slip through this
      verification. Please open an issue if you find above cases.
    
    


    2021-12-06 21:07:12,858 INFO Training Ended!
    2021-12-06 21:07:12,949 INFO mean_absolute_error: 15295.217291845034
    2021-12-06 21:07:12,950 INFO mean_squared_error: 629518554.1992229
    2021-12-06 21:07:12,950 INFO r2_score: 0.917928059746747
    [32m[I 2021-12-06 21:07:13,025][0m Trial 21 finished with value: 629518554.1992229 and parameters: {'learning_rate': 0.016262149275948868, 'n_estimators': 20000, 'reg_lambda': 3.0602182797930273, 'reg_alpha': 4.28112054954546e-05, 'subsample': 0.5976621107062762, 'colsample_bytree': 0.8189821798104691, 'max_depth': 5, 'early_stopping_rounds': 183, 'tree_method': 'hist', 'booster': 'gbtree', 'gamma': 0.09996502741004878, 'grow_policy': 'lossguide'}. Best is trial 0 with value: 607032267.1056623.[0m


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

    2021-12-06 21:07:13,069 INFO  model class:<class 'xgboost.sklearn.XGBRegressor'>
    /Users/pankajmathur/anaconda3/envs/nbdev_env/lib/python3.9/site-packages/sklearn/model_selection/_split.py:676: UserWarning: The least populated class in y has only 1 members, which is less than n_splits=5.
      warnings.warn(


    [21:07:13] WARNING: /Users/travis/build/dmlc/xgboost/src/learner.cc:541: 
    Parameters: { early_stopping_rounds } might not be used.
    
      This may not be accurate due to some parameters are only used in language bindings but
      passed down to XGBoost core.  Or some parameters are not used but slip through this
      verification. Please open an issue if you find above cases.
    
    


    2021-12-06 21:07:33,568 INFO Predicting Score!
    2021-12-06 21:07:33,577 INFO fold: 1 mean_absolute_error : 20715.247458261987
    2021-12-06 21:07:33,577 INFO fold: 1 mean_squared_error : 2957095332.886548
    2021-12-06 21:07:33,578 INFO fold: 1 r2_score : 0.4896348241260975
    2021-12-06 21:07:33,579 INFO Predicting Test Scores!


    [21:07:33] WARNING: /Users/travis/build/dmlc/xgboost/src/learner.cc:541: 
    Parameters: { early_stopping_rounds } might not be used.
    
      This may not be accurate due to some parameters are only used in language bindings but
      passed down to XGBoost core.  Or some parameters are not used but slip through this
      verification. Please open an issue if you find above cases.
    
    


    2021-12-06 21:07:53,660 INFO Predicting Score!
    2021-12-06 21:07:53,668 INFO fold: 2 mean_absolute_error : 16294.310145547945
    2021-12-06 21:07:53,669 INFO fold: 2 mean_squared_error : 830363243.5137496
    2021-12-06 21:07:53,669 INFO fold: 2 r2_score : 0.8520142405786058
    2021-12-06 21:07:53,670 INFO Predicting Test Scores!


    [21:07:53] WARNING: /Users/travis/build/dmlc/xgboost/src/learner.cc:541: 
    Parameters: { early_stopping_rounds } might not be used.
    
      This may not be accurate due to some parameters are only used in language bindings but
      passed down to XGBoost core.  Or some parameters are not used but slip through this
      verification. Please open an issue if you find above cases.
    
    


    2021-12-06 21:08:14,639 INFO Predicting Score!
    2021-12-06 21:08:14,648 INFO fold: 3 mean_absolute_error : 16816.43455693493
    2021-12-06 21:08:14,649 INFO fold: 3 mean_squared_error : 710149427.4676694
    2021-12-06 21:08:14,649 INFO fold: 3 r2_score : 0.9067089553667816
    2021-12-06 21:08:14,650 INFO Predicting Test Scores!


    [21:08:14] WARNING: /Users/travis/build/dmlc/xgboost/src/learner.cc:541: 
    Parameters: { early_stopping_rounds } might not be used.
    
      This may not be accurate due to some parameters are only used in language bindings but
      passed down to XGBoost core.  Or some parameters are not used but slip through this
      verification. Please open an issue if you find above cases.
    
    


    2021-12-06 21:08:33,527 INFO Predicting Score!
    2021-12-06 21:08:33,536 INFO fold: 4 mean_absolute_error : 15824.842572773972
    2021-12-06 21:08:33,536 INFO fold: 4 mean_squared_error : 523051432.0483315
    2021-12-06 21:08:33,537 INFO fold: 4 r2_score : 0.9062322810198855
    2021-12-06 21:08:33,537 INFO Predicting Test Scores!


    [21:08:33] WARNING: /Users/travis/build/dmlc/xgboost/src/learner.cc:541: 
    Parameters: { early_stopping_rounds } might not be used.
    
      This may not be accurate due to some parameters are only used in language bindings but
      passed down to XGBoost core.  Or some parameters are not used but slip through this
      verification. Please open an issue if you find above cases.
    
    


    2021-12-06 21:08:54,259 INFO Predicting Score!
    2021-12-06 21:08:54,266 INFO fold: 5 mean_absolute_error : 16750.133160316782
    2021-12-06 21:08:54,267 INFO fold: 5 mean_squared_error : 843596693.0681775
    2021-12-06 21:08:54,268 INFO fold: 5 r2_score : 0.876985526050572
    2021-12-06 21:08:54,268 INFO Predicting Test Scores!
    2021-12-06 21:08:54,297 INFO  Mean Metrics Results from all Folds are: {'mean_absolute_error': 17280.193578767125, 'mean_squared_error': 1172851225.7968953, 'r2_score': 0.8063151654283884}


    CPU times: user 11min 50s, sys: 12.4 s, total: 12min 2s
    Wall time: 1min 41s


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
