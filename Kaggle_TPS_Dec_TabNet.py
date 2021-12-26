
from tabular_ml_toolkit.tmlt import *
from xgboost import XGBClassifier
import numpy as np
import gc
import pandas as pd


from sklearn.metrics import roc_auc_score, accuracy_score, log_loss

# Dataset file names and Paths
DIRECTORY_PATH = "/home/pankaj/kaggle_datasets/tpc_dec_2021/"
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
SAMPLE_SUB_FILE = "sample_submission.csv"
OUTPUT_PATH = "kaggle_tps_dec_output/"


# create tmlt
tmlt = TMLT().prepare_data(
    train_file_path= DIRECTORY_PATH + TRAIN_FILE,
    test_file_path= DIRECTORY_PATH + TEST_FILE,
    #make sure to use right index and target columns
    idx_col="Id",
    target="Cover_Type",
    random_state=42,
    problem_type="multi_class_classification"
)


print(type(tmlt.dfl.X))
print(tmlt.dfl.X.shape)
print(type(tmlt.dfl.y))
print(tmlt.dfl.y.shape)
print(type(tmlt.dfl.X_test))
print(tmlt.dfl.X_test.shape)

print(dict(pd.Series(tmlt.dfl.y).value_counts()))


X_np, y_np, X_test_np = tmlt.pp_fit_transform(tmlt.dfl.X, tmlt.dfl.y, tmlt.dfl.X_test)
print(X_np.shape)
print(type(X_np))
print(y_np.shape)
print(type(y_np))
print(X_test_np.shape)
print(type(X_test_np))


gc.collect()


from pytorch_tabnet.tab_model import TabNetClassifier


tabnet_params = {
    'max_epochs': 30,
    'patience': 5,
    'batch_size': 4096*6*tmlt.IDEAL_CPU_CORES,
    'virtual_batch_size' : 512*6*tmlt.IDEAL_CPU_CORES
}

#choose model
tabnet_model = TabNetClassifier(optimizer_params=dict(lr=0.01), verbose=1)

# k-fold training
tabnet_model_metrics_score, tabnet_model_test_preds = tmlt.do_kfold_training(X_np, y_np, X_test=X_test_np, n_splits=5,
 model=tabnet_model,kfold_metric=accuracy_score,
 eval_metric = 'accuracy',
 tabnet_params=tabnet_params)

gc.collect()
# predict on test dataset
if tabnet_model_test_preds is not None:
    print(tabnet_model_test_preds.shape)


# #### Create Kaggle Predictions
test_preds = tabnet_model_test_preds
print(type(test_preds))

test_preds_round = np.around(test_preds).astype(int)
#test_preds_round[:1000]

print(f"{dict(pd.Series(test_preds_round).value_counts())}")

# target encoding changes 1 to 7 classes to 0 to 6
test_preds_round = test_preds_round + 1
print(type(test_preds_round))

print(f"{dict(pd.Series(test_preds_round).value_counts())}")

submission_file_name = 'tue_dec_21_2027_submission.csv'

sub = pd.read_csv(DIRECTORY_PATH + OUTPUT_PATH + SAMPLE_SUB_FILE)
sub['Cover_Type'] = test_preds_round

sub.to_csv(submission_file_name, index=False)
print(f"{submission_file_name} saved!")