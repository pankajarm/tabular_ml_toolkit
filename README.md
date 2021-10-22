# Tabular ML Toolkit
> A Helpful ML Toolkit to Jumpstart your Machine Learning Project based on Tabular or Structured data.


## Install

`pip install tabular_ml_toolkit`

## How to use

Just give input file path, index column, target column and validation set size and you will get X_train, X_valid, y_train and y_valid dataframes ready for modelling.

*For example below examples show how to load [Melbourne Home Sale price raw data](https://www.kaggle.com/estrotococo/home-data-for-ml-course)*

```python
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

    Home Data X_full shape: (1460, 80)
    Home Data X_test_full shape: (1459, 79)


Input raw data is already split into X (features) and y (target)

```python
# show shape of X and y 
print("homedata X shape:", dfl.X.shape)
print("homedata y shape", dfl.y.shape)
```

    homedata X shape: (1460, 79)
    homedata y shape (1460,)


Input raw data is also split into X_train, X_valid, y_train and y_valid based upon validation size provided

```python
# show shape of X_train, X_valid, y_train and y_valid
print("homedata X_train shape:", dfl.X_train.shape)
print("homedata y_train shape", dfl.y_train.shape)
print("homedata X_valid shape:", dfl.X_valid.shape)
print("homedata y_valid shape", dfl.y_valid.shape)
```

    homedata X_train shape: (1168, 76)
    homedata y_train shape (1168,)
    homedata X_valid shape: (292, 76)
    homedata y_valid shape (292,)

