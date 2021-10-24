# Tabular ML Toolkit
> A Helpful ML Toolkit to Jumpstart your Machine Learning Project based on Tabular or Structured data.


```
from tabular_ml_toolkit.MLPipeline import *
```

## Install

`pip install tabular_ml_toolkit`

## How to use

Start with your favorite model and then just simply create MLPipeline with one API.

*For example, Here we are using RandomForestRegressor from Scikit-Learn, on  [Melbourne Home Sale price data](https://www.kaggle.com/estrotococo/home-data-for-ml-course)*


*No need to install scikit-learn as it comes preinstall with Tabular_ML_Toolkit*

```
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# create scikit-learn ml model
scikit_model = RandomForestRegressor(n_estimators=100, random_state=42)
# createm ml pipeline for scikit-learn model
sci_ml_pl = MLPipeline().prepare_data_for_training(train_file_path= "input/home_data/train.csv",
                                              test_file_path= "input/home_data/test.csv",
                                              idx_col="Id", target="SalePrice",valid_size=0.2,
                                              model=scikit_model, random_state=42)
# Now fit and predict
sci_ml_pl.pipeline.fit(sci_ml_pl.dataframeloader.X_train, sci_ml_pl.dataframeloader.y_train)
preds = sci_ml_pl.pipeline.predict(sci_ml_pl.dataframeloader.X_valid)
print('X_valid MAE:', mean_absolute_error(sci_ml_pl.dataframeloader.y_valid, preds))
```

    X_valid MAE: 17678.294143835617


*You can also use MLPipeline with XGBoost model, Just make sure to install XGBooost first depending upon your OS.*

*After that all steps remains same. Here is example using XGBRegressor with [Melbourne Home Sale price data](https://www.kaggle.com/estrotococo/home-data-for-ml-course)*

```
from xgboost import XGBRegressor
# create xgb ml model
xgb_model = XGBRegressor(n_estimators=250,learning_rate=0.05, random_state=42)
# createm ml pipeline for xgb model
xgb_ml_pl = MLPipeline().prepare_data_for_training(train_file_path= "input/home_data/train.csv",
                                              test_file_path= "input/home_data/test.csv",
                                              idx_col="Id", target="SalePrice",valid_size=0.2,
                                              model=xgb_model, random_state=42)
# Now fit and predict
xgb_ml_pl.pipeline.fit(xgb_ml_pl.dataframeloader.X_train, xgb_ml_pl.dataframeloader.y_train)
preds = xgb_ml_pl.pipeline.predict(xgb_ml_pl.dataframeloader.X_valid)
print('X_valid MAE:', mean_absolute_error(xgb_ml_pl.dataframeloader.y_valid, preds))
```

    X_valid MAE: 16156.980896832192


In background `prepare_data_for_training` method loads your input data into Pandas DataFrame, seprates X(features) and y(target), split X(features) into X_train, y_train, X_valid, y_valid DataFrames. Then it preprocess all numerical and categorical type data found in these DataFrames. Then it bundle preprocessed data with your given model and return an MLPipeline object, so you can call MLPipeline to fit X_train and y_train and predict on X_valid or X_test.

Here is detail documentation and source code.

```
# show_doc(MLPipeline.prepare_data_for_training)
```


<h4 id="MLPipeline.prepare_data_for_training" class="doc_header"><code>MLPipeline.prepare_data_for_training</code><a href="https://github.com/psmathur/tabular_ml_toolkit/tree/master/tabular_ml_toolkit/MLPipeline.py#L59" class="source_link" style="float:right">[source]</a></h4>

> <code>MLPipeline.prepare_data_for_training</code>(**`train_file_path`**:`str`, **`test_file_path`**:`str`, **`idx_col`**:`str`, **`target`**:`str`, **`valid_size`**:`float`, **`model`**:`object`, **`random_state`**:`int`)




If you want to customize data and preprocessing steps you can do so by using `DataFrameLoader` and `PreProessor` classes. Check detail documentations for these classes for more options. 

```
# show_doc(MLPipeline)
```


<h2 id="MLPipeline" class="doc_header"><code>class</code> <code>MLPipeline</code><a href="https://github.com/psmathur/tabular_ml_toolkit/tree/master/tabular_ml_toolkit/MLPipeline.py#L21" class="source_link" style="float:right">[source]</a></h2>

> <code>MLPipeline</code>()

Represent MLPipeline class

Attributes:

pipeline: An MLPipeline instance 

dataframeloader: A DataFrameLoader instance 

preprocessor: A PreProcessor Instance 

model: The given Model

