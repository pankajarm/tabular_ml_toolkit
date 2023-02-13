
from tabular_ml_toolkit.tmlt import *
from xgboost import XGBClassifier
import numpy as np
import gc
import math
import pandas as pd
import torch


from sklearn.metrics import roc_auc_score, accuracy_score, log_loss

# Dataset file names and Paths
DIRECTORY_PATH = "/home/pankaj/kaggle_datasets/tpc_jan_2022/"
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
SAMPLE_SUB_FILE = "sample_submission.csv"
OUTPUT_PATH = "kaggle_tps_jan_2022_output/"


# create tmlt
tmlt = TMLT().prepare_data(
    train_file_path= DIRECTORY_PATH + TRAIN_FILE,
    test_file_path= DIRECTORY_PATH + TEST_FILE,
    #make sure to use right index and target columns
    idx_col="row_id",
    target="num_sold",
    random_state=42,
    problem_type="regression"
)


print(type(tmlt.dfl.X))
print(tmlt.dfl.X.shape)
print(type(tmlt.dfl.y))
print(tmlt.dfl.y.shape)
print(type(tmlt.dfl.X_test))
print(tmlt.dfl.X_test.shape)

print(dict(pd.Series(tmlt.dfl.y).value_counts()))

###EDA


###Training
#PreProcess X, y and X_test DataFrames before Training
X = tmlt.dfl.X
y = tmlt.dfl.y
X_test = tmlt.dfl.X_test

X_np, y_np, X_test_np = tmlt.pp_fit_transform(X,y, X_test)
print(X_np.shape)
print(type(X_np))
print(y_np.shape)
print(type(y_np))
print(X_test_np.shape)
print(type(X_test_np))


gc.collect()


#### Tabnet Training

from pytorch_tabnet.tab_model import TabNetClassifier


tabnet_fit_params = {
    'max_epochs': 20,
    'patience': 3,
    'batch_size': 4096*1*tmlt.IDEAL_CPU_CORES,
    'virtual_batch_size' : 512*1*tmlt.IDEAL_CPU_CORES
}

#choose model
# tabnet_model_params = {
#     n_d=64, n_a=64, n_steps=5,
#     gamma=1.5, n_independent=2, n_shared=2,
#     cat_idxs=cat_idxs,
#     cat_dims=cat_dims,
#     cat_emb_dim=1,
#     lambda_sparse=1e-4, momentum=0.3, clip_value=2.,
#     optimizer_fn=torch.optim.Adam,
#     optimizer_params=dict(lr=2e-2),
#     scheduler_params = {"gamma": 0.95,
#                      "step_size": 20},
#     scheduler_fn=torch.optim.lr_scheduler.StepLR, epsilon=1e-15
# }
tabnet_model = TabNetClassifier(n_d=64, n_a=64, n_steps=5, gamma=1.5, n_independent=2, n_shared=2,
    lambda_sparse=1e-4, momentum=0.3, clip_value=2.,
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    scheduler_params = {"gamma": 0.95,
                     "step_size": 20},
    scheduler_fn=torch.optim.lr_scheduler.StepLR, epsilon=1e-15)

# k-fold training
tabnet_model_metrics_score, tabnet_model_test_preds = tmlt.do_kfold_training(X_np, y_np, X_test=X_test_np, n_splits=5,
 model=tabnet_model, kfold_metric=accuracy_score,
 eval_metric = 'accuracy',
 tabnet_fit_params=tabnet_fit_params)

gc.collect()
# predict on test dataset
if tabnet_model_test_preds is not None:
    print(tabnet_model_test_preds.shape)


# #### Create Kaggle Predictions
test_preds = tabnet_model_test_preds
print(type(test_preds))
print(f"{dict(pd.Series(test_preds).value_counts())}")

submission_file_name = 'tue_dec_27_1211_submission.csv'

sub = pd.read_csv(DIRECTORY_PATH + SAMPLE_SUB_FILE)
sub['Cover_Type'] = test_preds

sub.to_csv(OUTPUT_PATH+ submission_file_name, index=False)
print(f"{submission_file_name} saved!")




# LOG OF PREVIOUS RUNS
# 2021-12-27 13:43:47,734 INFO  Mean accuracy_score from all Folds are: {'accuracy_score': 0.9572222250003679}
# (1000000,)
# <class 'numpy.ndarray'>
# {2: 515302, 1: 383851, 3: 79160, 7: 11191, 6: 4021, 5: 3813, 4: 2662}