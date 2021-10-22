# Tabular ML Toolkit
> A Helpful ML Toolkit to Jumpstart your Machine Learning Project based on Tabular or Structured data.


## Install

`pip install tabular_ml_toolkit`

## How to use

Just give input file path, index column, target column and validation set size and you will get X_train, X_valid, y_train and y_valid dataframes ready for modelling.

*For example below examples show how to load [Melbourne Home Sale price raw data](https://www.kaggle.com/estrotococo/home-data-for-ml-course)*

```
dfl = DataFrameLoader().from_csv(
    train_file_path="input/home_data/train.csv",
    test_file_path="input/home_data/test.csv",
    idx_col="Id",
    target="SalePrice",
    valid_size=0.2)
# show shape of dataframes
print("Home Data X_full shape:", dfl.X_full.shape)
print("Home Data X_test_full shape:", dfl.X_test_full.shape)
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    /var/folders/p3/zmg8jfwx0hb9gwzs0w69d7f0rgyjx2/T/ipykernel_34013/2444152123.py in <module>
    ----> 1 dfl = DataFrameLoader().from_csv(
          2     train_file_path="input/home_data/train.csv",
          3     test_file_path="input/home_data/test.csv",
          4     idx_col="Id",
          5     target="SalePrice",


    TypeError: from_csv() got an unexpected keyword argument 'valid_size'


Input raw data is already split into X (features) and y (target)

```
# show shape of X and y 
print("homedata X shape:", dfl.X.shape)
print("homedata y shape", dfl.y.shape)
```

Input raw data is also split into X_train, X_valid, y_train and y_valid based upon validation size provided

```
# show shape of X_train, X_valid, y_train and y_valid
print("homedata X_train shape:", dfl.X_train.shape)
print("homedata y_train shape", dfl.y_train.shape)
print("homedata X_valid shape:", dfl.X_valid.shape)
print("homedata y_valid shape", dfl.y_valid.shape)
```
