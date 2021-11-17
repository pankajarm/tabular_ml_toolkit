#!/usr/bin/env python
# coding: utf-8

from tabular_ml_toolkit.mlpipeline import *
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

# Dataset file names and Paths
DIRECTORY_PATH = "https://raw.githubusercontent.com/psmathur/tabular_ml_toolkit/master/input/home_data/"
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
SAMPLE_SUB_FILE = "sample_submission.csv"

xgb_params = {
    'n_estimators':250,
    'learning_rate':0.05,
    'random_state':42,
}

# create xgb ml model
xgb_model = XGBRegressor(**xgb_params)

# createm ml pipeline for scikit-learn model
tmlt = MLPipeline().prepare_data_for_training(
    train_file_path= DIRECTORY_PATH+TRAIN_FILE,
    test_file_path= DIRECTORY_PATH+TEST_FILE,
    idx_col="Id", target="SalePrice",
    model=xgb_model,
    random_state=42)

study = tmlt.do_xgb_optuna_optimization(task="regression", xgb_eval_metric="rmse",
                                        kfold_metrics=mean_absolute_error, output_dir_path="output/")
print("Study Best Trial:", study.best_trial)