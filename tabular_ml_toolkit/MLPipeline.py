# AUTOGENERATED! DO NOT EDIT! File to edit: 02_MLPipeline.ipynb (unless otherwise specified).

__all__ = ['MLPipeline']

# Cell
from .DataFrameLoader import *
from .PreProcessor import *

# Cell
# hide
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Cell

class MLPipeline:
    """
    Represent MLPipeline class

    Attributes:\n
    pipeline: An MLPipeline instance \n
    dataframeloader: A DataFrameLoader instance \n
    preprocessor: A PreProcessor Instance \n
    model: The given Model
    """

    def __init__(self):
        self.pipeline = None
        self.dataframeloader = None
        self.preprocessor = None
        self.model = None

    def __str__(self):
        """Returns human readable string reprsentation"""
        return "Training Pipeline object with attributes: pl"

    def __repr__(self):
        return self.__str__()

#     def __lt__(self):
#         """returns: boolean"""
#         return True

    # core methods
    # Bundle preprocessing and modeling code in a training pipeline
    def bundle_preproessor_model(self, preprocessor:object, model:object):
        self.pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', model)
                     ])
#     # return pipeline object
#     def create_pipeline(self, preprocessor:object, model:object):
#         self.bundle_preproessor_model(preprocessor, model)

    def prepare_data_for_training(self, train_file_path:str, test_file_path:str, idx_col:str, target:str, valid_size:float, model:object, random_state:int):
        self.model = model
        # call DataFrameLoader module
        self.dataframeloader = DataFrameLoader().from_csv(train_file_path,test_file_path,idx_col,target,valid_size)
        # call PreProcessor module
        self.preprocessor = PreProcessor().preprocess_data(
            numerical_cols=self.dataframeloader.numerical_cols,
            low_card_cat_cols=self.dataframeloader.low_card_cat_cols,
            high_card_cat_cols=self.dataframeloader.high_card_cat_cols
        )

        # call self module method
        self.bundle_preproessor_model(self.preprocessor.columns_transfomer, model)
        return self