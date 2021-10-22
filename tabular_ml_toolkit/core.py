# AUTOGENERATED! DO NOT EDIT! File to edit: 00_dataframeloader.ipynb (unless otherwise specified).

__all__ = ['DataFrameLoader']

# Cell
# hide
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Cell

class DataFrameLoader:
    """
    Represent Data Frame Loader class

    Attributes:
    X_full: full dataframe load from raw input
    X_test_full: full test dataframe load from raw input
    X: features
    y: target
    """

    def __init__(self):
        self.X_full = None
        self.X_test_full = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_valid = None
        self.X_test = None
        self.y_train = None
        self.y_valid = None
        self.categorical_cols = None
        self.numerical_cols = None
        self.final_columns = None

    def __str__(self):
        """Returns human readable string reprsentation"""
        return "DataFrameLoader object with X_full, X_test and X(features) dataframe and y(target) series"

    def __repr__(self):
        return self.__str__()

#     def __lt__(self):
#         """returns: boolean"""
#         return True

    # load data from csv
    def read_csv(self,train_file_path:str,test_file_path:str, idx_col:str):
        # Read the csv files using pandas
        self.X_full = pd.read_csv(train_file_path, index_col=idx_col)
        self.X_test_full = pd.read_csv(test_file_path, index_col=idx_col)

    # prepare X and y
    def prepare_X_y(self,input_df:object, target:str):
        # Remove rows with missing target
        self.X = input_df.dropna(axis=0, subset=[target])
        # separate target from predictors
        self.y = self.X[target]
        # drop target
        self.X = input_df.drop([target], axis=1)

    # split X and y into X_train, y_train, X_valid & y_valid dataframes
    def prepare_train_valid(self,X:object,y:object, test_size:float, random_state=42):
        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(
            self.X, self.y, train_size=(1-test_size), test_size=test_size,random_state=random_state)

    # select categorical columns
    def select_categorical_cols(self):
        self.categorical_cols = [cname for cname in self.X_train.columns if
                    self.X_train[cname].nunique() < 10 and
                    self.X_train[cname].dtype == "object"]
        #TODO: seprate categorical columns into one hot eligible cols for low cardinality
        # and ordinal cols for high cardinatliy

    # select numerical columns
    def select_numerical_cols(self):
        self.numerical_cols = [cname for cname in self.X_train.columns if
                self.X_train[cname].dtype in ['int64', 'float64']]

    # prepare X_train, X_valid from selected columns
    def prepare_X_train_X_valid(self):
        self.select_categorical_cols()
        self.select_numerical_cols()
        self.final_columns = self.categorical_cols + self.numerical_cols
        self.X_train = self.X_train[self.final_columns].copy()
        self.X_valid = self.X_valid[self.final_columns].copy()
        self.X_test = self.X_test_full[self.final_columns].copy()

    # get train and valid dataframe
    def from_csv(self, train_file_path:str,test_file_path:str, idx_col:str, target:str, test_size:float, random_state=42):
        self.read_csv(train_file_path,test_file_path, idx_col)
        self.prepare_X_y(self.X_full, target)
        self.prepare_train_valid(self.X,self.y, test_size, random_state)
        self.prepare_X_train_X_valid()
        return self