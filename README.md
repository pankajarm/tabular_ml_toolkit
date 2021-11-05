# Tabular ML Toolkit
> A superfast helper library to jumpstart your machine learning project based on tabular or structured data.


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
    idx_col="Id", target="SalePrice",
    model=scikit_model,
    random_state=42)

#scikit-pipeline
# tmlt.spl
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

    Fit Time: 1.0225746631622314
    X_valid MAE: 17634.989965753426


*You can also use MLPipeline with XGBoost model, Just make sure to install XGBooost first depending upon your OS.*

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
# Update pipeline with xgb model
tmlt.update_model(xgb_model)
# tmlt.spl
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

    Fit Time: 0.502791166305542
    X_valid MAE: 15851.009123501712


In background `prepare_data_for_training` method loads your input data into Pandas DataFrame, seprates X(features) and y(target), Then it preprocess all numerical and categorical type data found in these DataFrames. Then it bundle preprocessed data with your given model and return an MLPipeline object which contains dataframeloader, preprocessor and scikit-learn pipeline.

`create_train_valid` methods split X(features) into X_train, y_train, X_valid, y_valid DataFrames.

so you can call scikit-learn pipeline fit method on X_train and y_train and predict on X_valid or X_test.

Here is detail documentation and source code.

If you want to customize data and preprocessing steps you can do so by using `DataFrameLoader` and `PreProessor` classes. Please Check other Tutorials and detail documentations for these classes for more options. 
