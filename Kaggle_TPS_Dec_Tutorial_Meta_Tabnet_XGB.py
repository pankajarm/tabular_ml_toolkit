from tabular_ml_toolkit.tmlt import *
from xgboost import XGBClassifier
import numpy as np
import gc
import pandas as pd
import torch


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

#tmlt has X, y and X_test dataframe loader
X = tmlt.dfl.X
y = tmlt.dfl.y
X_test = tmlt.dfl.X_test

print(type(X))
print(X.shape)
print(type(y))
print(y.shape)
print(type(X_test))
print(X_test.shape)

print(f"Original Target Class Count: {dict(pd.Series(tmlt.dfl.y).value_counts())}")

gc.collect()

# ### PreProcess X, y and X_test, returns numpy arrays
X_np, y_np, X_test_np = tmlt.pp_fit_transform(X, y, X_test)

print(X_np.shape)
print(type(X_np))
print(y_np.shape)
print(type(y_np))
print(X_test_np.shape)
print(type(X_test_np))

# check class balance
print(f"Encoded Target Class Count: {dict(pd.Series(tmlt.dfl.y).value_counts())}")
gc.collect()


# ### For Meta Ensemble Models Training

# #### Base Model: TabNet

from pytorch_tabnet.tab_model import TabNetClassifier

# for Categorical Columns
# from sklearn.preprocessing import LabelEncoder
# categorical_dims =  {}
# for col in X.columns[X.dtypes == object]:
#     print("Unique Categorical Count for:", col, " is:", X[col].nunique())
#     l_enc = LabelEncoder()
#     # train[col] = train[col].fillna("VV_likely")
#     temp = l_enc.fit_transform(X[col].values)
#     categorical_dims[col] = len(l_enc.classes_)

# print(f"categorical_dims values are: {categorical_dims}")
# target = tmlt.dfl.target
# categorical_columns = tmlt.dfl.categorical_cols

# unused_feat = []
# features = [ col for col in X.columns if col not in unused_feat+[target]] 
# cat_idxs = [ i for i, f in enumerate(features) if f in categorical_columns]
# cat_dims = [ categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns]
# print(f"cat_idxs values are: {cat_idxs}")
# print(f"cat_dims values are: {cat_dims}")

tabnet_fit_params = {
    'max_epochs': 2,
    'patience': 1,
    'batch_size': 4096*1*tmlt.IDEAL_CPU_CORES,
    'virtual_batch_size' : 512*1*tmlt.IDEAL_CPU_CORES
}

#choose model
tabnet_model = TabNetClassifier(n_d=64, n_a=64, n_steps=5, gamma=1.5, n_independent=2, n_shared=2,
    # cat_idxs=cat_idxs,
    # cat_dims=cat_dims,
    # cat_emb_dim=1,
    lambda_sparse=1e-4, momentum=0.3, clip_value=2.,
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    scheduler_params = {"gamma": 0.95,
                     "step_size": 20},
    scheduler_fn=torch.optim.lr_scheduler.StepLR, epsilon=1e-15)

#fit and predict
_, tabnet_oof_model_test_preds, tabnet_oof_model_preds = tmlt.do_kfold_train_preds(n_splits=5, model=tabnet_model, X = X_np, y = y_np, oof=True, X_test = X_test_np,
val_metric=accuracy_score, eval_metric='accuracy', tabnet_fit_params=tabnet_fit_params)

gc.collect()


if tabnet_oof_model_preds is not None:
    print(tabnet_oof_model_preds.shape)

if tabnet_oof_model_test_preds is not None:
    print(tabnet_oof_model_test_preds.shape)


# add based model oof predictions back to tmlt X and X_test before Meta model training
tmlt.dfl.X["tabnet_preds"] = tabnet_oof_model_preds
tmlt.dfl.X_test["tabnet_preds"] = tabnet_oof_model_test_preds

print(tmlt.dfl.X.shape)
print(tmlt.dfl.X_test.shape)


# #### now just update the tmlt with this new X and X_test
tmlt = tmlt.update_dfl(X=tmlt.dfl.X, y=tmlt.dfl.y, X_test=tmlt.dfl.X_test, problem_type = tmlt.problem_type)
print(type(tmlt.dfl.X))
print(tmlt.dfl.X.shape)
print(type(tmlt.dfl.y))
print(tmlt.dfl.y.shape)
print(type(tmlt.dfl.X_test))
print(tmlt.dfl.X_test.shape)

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
    'objective':'multi:softmax',
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
xgb_model_metrics_score, xgb_model_test_preds = tmlt.do_kfold_train_preds(X_np, y_np, X_test=X_test_np, n_splits=5,model=xgb_model, val_metric=accuracy_score, eval_metric='mlogloss')

gc.collect()

# predict on test dataset
if xgb_model_test_preds is not None:
    print(xgb_model_test_preds.shape)

# #### Create Kaggle Predictions
test_preds = xgb_model_test_preds
print(type(test_preds))
print(f"{dict(pd.Series(test_preds).value_counts())}")

submission_file_name = 'wed_dec_29_1152_submission.csv'

sub = pd.read_csv(DIRECTORY_PATH + SAMPLE_SUB_FILE)
sub['Cover_Type'] = test_preds

sub.to_csv(OUTPUT_PATH+ submission_file_name, index=False)
print(f"{submission_file_name} saved!")