# AUTOGENERATED! DO NOT EDIT! File to edit: 00_dataframeloader.ipynb (unless otherwise specified).

__all__ = ['DataFrameLoader']

# Cell
# hide
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from .logger import *

# Cell
# hide

# make sure to pip install modin[ray]>=0.11.3
# #settings for modin
# import ray
# ray.init()
# import os
# os.environ["MODIN_ENGINE"] = "ray"
# import modin.pandas as pd

# Cell

class DataFrameLoader:
    """
    Represent DataFrameLoader class

    Attributes:
    X_test: test dataframe
    X: features dataframe
    y: target series
    """

    def __init__(self):

        self.numerics = ["int8", "int16", "int32", "int64", "float16", "float32", "float64"]
        self.shape_X_full = None
        self.X_full = None
        self.X_test = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_valid = None
        self.y_train = None
        self.y_valid = None
        self.use_num_cols = None
        self.use_cat_cols = None
        self.categorical_cols = None
        self.numerical_cols = None
        self.low_card_cat_cols = None
        self.high_card_cat_cols = None
        self.final_cols = None
        self.target = None

    def __str__(self):
        """Returns human readable string reprsentation"""
        return "DataFrameLoader object with attributes: X_full, X_test, X(features), y(target), X_train, X_valid, y_train and y_valid"

    def __repr__(self):
        return self.__str__()

    # utility method
    # Idea taken from https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65/comments
    # Author ArjenGroen https://www.kaggle.com/arjanso
    def reduce_num_dtype_mem_usage(self, df, verbose=True):
        start_mem = df.memory_usage().sum() / 1024 ** 2
        for col in df.columns:
            col_type = df[col].dtypes
            if col_type in self.numerics:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == "int":
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
        end_mem = df.memory_usage().sum() / 1024 ** 2
        if verbose:
            logger.info(
                "DataFrame Memory usage decreased to {:.2f} Mb ({:.1f}% reduction)".format(
                    end_mem, 100 * (start_mem - end_mem) / start_mem
                )
            )
        return df

    # CORE METHODS
    # load data from csv
    def read_csv(self,train_file_path:str,test_file_path:str, idx_col:str, nrows:int):
        # Read the csv files using pandas
        if train_file_path is not None:
            self.X_full = pd.read_csv(train_file_path, index_col=idx_col, nrows=nrows)
            self.shape_X_full = self.X_full.shape
            self.X_full = self.reduce_num_dtype_mem_usage(self.X_full, verbose=True)
        else:
            logger.warn(f"No valid train_file_path provided: {train_file_path}")

        if test_file_path is not None:
            self.X_test = pd.read_csv(test_file_path, index_col=idx_col, nrows=nrows)
            self.shape_X_test = self.X_test.shape
            self.X_test = self.reduce_num_dtype_mem_usage(self.X_test, verbose=True)

        else:
            logger.info(f"No test_file_path given, so training will continue without it!")
        return self


    # prepare X and y
    def prepare_X_y(self,input_df:object, target:str):
        # Remove rows with missing target
        self.X = input_df.dropna(axis=0, subset=[target])
        # separate target from predictors
        #TODO: change to to_numpy
        self.y = self.X[target].values
        self.target = target
        # drop target
        self.X = input_df.drop([target], axis=1)
        return self

    # select categorical columns
    def select_categorical_cols(self):
        # for low cardinality columns
        self.low_card_cat_cols = [cname for cname in self.X.columns if
                    self.X[cname].nunique() < 10 and
                    self.X[cname].dtype == "object"]
        # for high cardinality columns
        self.high_card_cat_cols = [cname for cname in self.X.columns if
                    self.X[cname].nunique() > 10 and
                    self.X[cname].dtype == "object"]
        # for all categorical columns
        self.categorical_cols = self.low_card_cat_cols + self.high_card_cat_cols

    # select numerical columns
    def select_numerical_cols(self):
        self.numerical_cols = [cname for cname in self.X.columns if
                self.X[cname].dtype in self.numerics]

    # prepare final columns by data type
    def prepare_final_cols(self, use_num_cols:bool, use_cat_cols:bool):
        self.use_num_cols = use_num_cols
        self.use_cat_cols = use_cat_cols

        if self.use_num_cols:
            self.select_categorical_cols()
        if self.use_cat_cols:
            self.select_numerical_cols()

        if (self.numerical_cols is not None) and (self.categorical_cols is not None):
            self.final_cols = self.numerical_cols + self.categorical_cols

        elif (self.numerical_cols is not None) and (self.categorical_cols is None):
            self.final_cols = self.numerical_cols

        elif (self.numerical_cols is None) and (self.categorical_cols is not None):
            self.final_cols = self.categorical_cols


    # prepare X_train, X_valid from selected columns
    def update_X_train_X_valid_X_test_with_final_cols(self, final_cols:object):
        self.X_train = self.X_train[final_cols]
        self.X_valid = self.X_valid[final_cols]
        if self.X_test is not None:
            self.X_test = self.X_test[final_cols]


    def update_X_y_with_final_cols(self,final_cols:object):
        self.X = self.X[final_cols]
        if self.X_test is not None:
            self.X_test = self.X_test[final_cols]

    # split X and y into X_train, y_train, X_valid & y_valid dataframes
    def create_train_valid(self, valid_size:float, X=None, y=None, random_state=42):

        if X and y:
            self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(
                X, y, train_size=(1-valid_size), test_size=valid_size, random_state=random_state)

        else:
            self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(
                self.X, self.y, train_size=(1-valid_size), test_size=valid_size, random_state=random_state)

        self.update_X_train_X_valid_X_test_with_final_cols(self.final_cols)

    # get train and valid dataframe
    def from_csv(self, train_file_path:str,
                 idx_col:str, target:str,
                 nrows:int=None,
                 test_file_path:str=None,
                 use_num_cols:bool=True,
                 use_cat_cols:bool=True,
                 random_state=42):

        # read csv and load dataframes using pandas
        self.read_csv(train_file_path,test_file_path, idx_col, nrows)
        if self.X_full is not None:
            self.prepare_X_y(self.X_full, target)
        # create final columns based upon type of columns
        self.prepare_final_cols(use_num_cols=use_num_cols, use_cat_cols=use_cat_cols)
        if self.final_cols is not None:
            self.update_X_y_with_final_cols(self.final_cols)

        # clean up unused dataframes
        unused_df_lst = [self.X_full]
        del unused_df_lst

        return self

        # get train and valid dataframe
    def regenerate_dfl(self, X:object, y:object, X_test:object, use_num_cols:bool=True,
                       use_cat_cols:bool=True, random_state=42):
        # assign X,y and X_test
        self.X = X
        self.y = y
        self.X_test = X_test

        # create final columns based upon dtype of columns
        self.prepare_final_cols(use_num_cols=use_num_cols, use_cat_cols=use_cat_cols)

        if self.final_cols is not None:
            self.update_X_y_with_final_cols(self.final_cols)

        return self