# Tabular ML Toolkit
> A Helpful ML Toolkit to Jumpstart your Machine Learning Project based on Tabular or Structured data.


## Install

`pip install tabular_ml_toolkit`

## How to use

Load raw data into Pandas dataframe

```python
dfl = DataFrameLoader()
dfl.from_csv(
    train_file_path='input/home_data/train.csv',
    test_file_path='input/home_data/train.csv',
    idx_col='Id')
# show shape of dataframes
print("X_full shape:", dfl.X_full.shape)
print("X_test_full shape:", dfl.X_test.shape)
```

    X_full shape: (1460, 80)
    X_test_full shape: (1460, 80)


Prepare X (features) dataframe and y(target) series

```python
# Prepare X and y for Melbourne Home Sale Price data
dfl.prepare_X_y(input_df=dfl.X_full, target='SalePrice')

# show shape of X and y 
print("homedata_X shape:", dfl.X.shape)
print("homedata_y shape", dfl.y.shape)
```

    homedata_X shape: (1460, 79)
    homedata_y shape (1460,)

