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
    problem_type="multi_class_classification",
#     nrows=4000
)
# tmlt supports only below task type:
    # "binary_classification"
    # "multi_label_classification"
    # "multi_class_classification"
    # "regression"

print(type(tmlt.dfl.X))
print(tmlt.dfl.X.shape)
print(type(tmlt.dfl.y))
print(tmlt.dfl.y.shape)
print(type(tmlt.dfl.X_test))
print(tmlt.dfl.X_test.shape)

print(dict(pd.Series(tmlt.dfl.y).value_counts()))

gc.collect()

# ### PreProcess X, y and X_test
X_np, y_np, X_test_np = tmlt.pp_fit_transform(tmlt.dfl.X, tmlt.dfl.y, tmlt.dfl.X_test)
print(X_np.shape)
print(type(X_np))
print(y_np.shape)
print(type(y_np))
print(X_test_np.shape)
print(type(X_test_np))

# check class balance
print(dict(pd.Series(y_np).value_counts()))
gc.collect()


# ### For Meta Ensemble Models Training

# #### Base Model: TabNet

from pytorch_tabnet.tab_model import TabNetClassifier

# #### Now add back based models predictions to X and X_test
# OOF training and prediction on both train and test dataset by a given model
tabnet_params = {
    'max_epochs': 3,
    'patience': 1,
    'batch_size': 4096*6*tmlt.IDEAL_CPU_CORES,
    'virtual_batch_size' : 512*6*tmlt.IDEAL_CPU_CORES
}

#choose model
tabnet_oof_model = TabNetClassifier(optimizer_params=dict(lr=0.02), verbose=1)

#fit and predict
tabnet_oof_model_preds, tabnet_oof_model_test_preds = tmlt.do_oof_kfold_train_preds(n_splits=5, model=tabnet_oof_model,
X = X_np,
y = y_np,
X_test = X_test_np,
tabnet_params=tabnet_params
)
gc.collect()


if tabnet_oof_model_preds is not None:
    print(tabnet_oof_model_preds.shape)

if tabnet_oof_model_test_preds is not None:
    print(tabnet_oof_model_test_preds.shape)


# add based model oof predictions back to X and X_test before Meta model training
tmlt.dfl.X["tabnet_preds"] = tabnet_oof_model_preds
tmlt.dfl.X_test["tabnet_preds"] = tabnet_oof_model_test_preds

print(tmlt.dfl.X.shape)
print(tmlt.dfl.X_test.shape)


# #### now just update the tmlt with this new X and X_test
tmlt = tmlt.update_dfl(X=tmlt.dfl.X, y=tmlt.dfl.y, X_test=tmlt.dfl.X_test)


# #### For META Model Training

# ##### Now PreProcess X_train, X_valid
# 
# NOTE: Preprocessing gives back numpy arrays for pandas dataframe
X_np, y_np, X_test_np = tmlt.pp_fit_transform(tmlt.dfl.X, tmlt.dfl.y, tmlt.dfl.X_test)

print(X_np.shape)
print(type(X_np))
print(y_np.shape)
print(type(y_np))
print(X_test_np.shape)
print(type(X_test_np))

# ##### create train valid dataframes for training

X_train_np, X_valid_np,  y_train_np, y_valid_np =  tmlt.dfl.create_train_valid(X_np, y_np, valid_size=0.2)

print(X_train_np.shape)
print(type(X_train_np))
print(y_train_np.shape)
print(type(y_train_np))
print(X_valid_np.shape)
print(type(X_valid_np))
print(y_valid_np.shape)
print(type(y_valid_np))

gc.collect()
# xgb params taken after optuna optimization different notebook/script
xgb_params = {
    'use_label_encoder': False,
    'learning_rate': 0.22460180743878044,
    'n_estimators': 15,
    'reg_lambda': 3.144893773482e-05,
    'reg_alpha': 0.00023758525471934383,
    'subsample': 0.2640308356915845,
    'colsample_bytree': 0.7501402977241696,
    'max_depth': 7,
    'tree_method': 'gpu_hist',
    'gpu_id': 0,
    'predictor': 'gpu_predictor',
    'early_stopping_rounds': 384
}
xgb_model = XGBClassifier(**xgb_params)

# #### Let's Use K-Fold Training with best params
# k-fold training
xgb_model_metrics_score, xgb_model_test_preds = tmlt.do_kfold_training(X_np, y_np, X_test=X_test_np, n_splits=5,
model=xgb_model, kfold_metric=accuracy_score, eval_metric='mlogloss')

gc.collect()

# predict on test dataset
if xgb_model_test_preds is not None:
    print(xgb_model_test_preds.shape)

# #### Create Kaggle Predictions

test_preds = xgb_model_test_preds
print(type(test_preds))


print(f"{dict(pd.Series(test_preds).value_counts())}")

test_preds = np.around(test_preds).astype(int)

print(f"{dict(pd.Series(test_preds).value_counts())}")

# target encoding changes 1 to 7 classes to 0 to 6
test_preds = test_preds + 1

print(f"{dict(pd.Series(test_preds).value_counts())}")

submission_file_name = 'sat_dec_25_2036_submission.csv'

sub = pd.read_csv(DIRECTORY_PATH + SAMPLE_SUB_FILE)
sub['Cover_Type'] = test_preds

sub.to_csv(OUTPUT_PATH+submission_file_name, index=False)
print(f"{submission_file_name} saved!")