
from tabular_ml_toolkit.tmlt import *
from xgboost import XGBClassifier
import numpy as np
import gc
import math
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
    problem_type="multi_class_classification"
)


print(type(tmlt.dfl.X))
print(tmlt.dfl.X.shape)
print(type(tmlt.dfl.y))
print(tmlt.dfl.y.shape)
print(type(tmlt.dfl.X_test))
print(tmlt.dfl.X_test.shape)

print(dict(pd.Series(tmlt.dfl.y).value_counts()))

#Do Pseudo Labelling


#Do Feature Engineering
# Feature Engineering Method
#These features are borrowed from https://www.kaggle.com/gulshanmishra/tps-dec-21-tensorflow-nn-feature-engineering 
# def feature_engineer(df):
#     # Distance features
#     # Euclidean distance to Hydrology
#     df["ecldn_dist_hydrlgy"] = (
#         df["Horizontal_Distance_To_Hydrology"]**2 + df["Vertical_Distance_To_Hydrology"]**2)**0.5
#     df["fire_road"] = np.abs(df["Horizontal_Distance_To_Fire_Points"]) + \
#         np.abs(df["Horizontal_Distance_To_Roadways"])

#     # Elevation features
#     df['highwater'] = (df.Vertical_Distance_To_Hydrology < 0).astype(int)

#     # Aspect features, hardest FE
#     df.loc[df["Aspect"] < 0, "Aspect"] += 360
#     df.loc[df["Aspect"] > 359, "Aspect"] -= 360
#     df['binned_aspect'] = [math.floor((v+60)/45.0) for v in df['Aspect']]
#     df['binned_aspect2'] = [math.floor((v+180)/30.0) for v in df['Aspect']]

#     # Soil and wilderness features
#     soil_features = [x for x in df.columns if x.startswith("Soil_Type")]
#     df["soil_type_count"] = df[soil_features].sum(axis=1)
#     wilderness_features = [
#         x for x in df.columns if x.startswith("Wilderness_Area")]
#     df["wilderness_area_count"] = df[wilderness_features].sum(axis=1)
#     df['soil_Type12_32'] = df['Soil_Type32'] + df['Soil_Type12']
#     df['soil_Type23_22_32_33'] = df['Soil_Type23'] + \
#         df['Soil_Type22'] + df['Soil_Type32'] + df['Soil_Type33']

#     # Hillshade features
#     features_Hillshade = ['Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']
#     df.loc[df["Hillshade_9am"] < 0, "Hillshade_9am"] = 0
#     df.loc[df["Hillshade_Noon"] < 0, "Hillshade_Noon"] = 0
#     df.loc[df["Hillshade_3pm"] < 0, "Hillshade_3pm"] = 0
#     df.loc[df["Hillshade_9am"] > 255, "Hillshade_9am"] = 255
#     df.loc[df["Hillshade_Noon"] > 255, "Hillshade_Noon"] = 255
#     df.loc[df["Hillshade_3pm"] > 255, "Hillshade_3pm"] = 255
#     df['Hillshade_Noon_is_bright'] = (df.Hillshade_Noon == 255).astype(int)
#     df['Hillshade_9am_is_zero'] = (df.Hillshade_9am == 0).astype(int)
#     df['hillshade_3pm_is_zero'] = (df.Hillshade_3pm == 0).astype(int)

#     df.drop(["Aspect", 'Horizontal_Distance_To_Hydrology'] +
#             features_Hillshade, axis=1, inplace=True)

#     return df

# # now update X and X_test with feature engineering method
# tmlt.dfl.X = feature_engineer(tmlt.dfl.X)
# tmlt.dfl.X_test = feature_engineer(tmlt.dfl.X_test)

# # remove unncesseary columns
# cols = ["Soil_Type7", "Soil_Type15"]
# tmlt.dfl.X .drop(cols, axis = 1, inplace= True)
# tmlt.dfl.X_test.drop(cols, axis = 1, inplace = True)

# #### now just update the tmlt with this new X and X_test
# tmlt = tmlt.update_dfl(X=tmlt.dfl.X, y=tmlt.dfl.y, X_test=tmlt.dfl.X_test, problem_type = tmlt.problem_type)
# print(type(tmlt.dfl.X))
# print(tmlt.dfl.X.shape)
# print(type(tmlt.dfl.y))
# print(tmlt.dfl.y.shape)
# print(type(tmlt.dfl.X_test))
# print(tmlt.dfl.X_test.shape)



### Now PreProcess X, y and X_test DataFrames before Training
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
from sklearn.preprocessing import LabelEncoder
categorical_dims =  {}
for col in X.columns[X.dtypes == object]:
    print(col, X[col].nunique())
    l_enc = LabelEncoder()
    # train[col] = train[col].fillna("VV_likely")
    temp = l_enc.fit_transform(X[col].values)
    categorical_dims[col] = len(l_enc.classes_)

print(f"categorical_dims values are: {categorical_dims}")
target = tmlt.dfl.target
categorical_columns = tmlt.dfl.categorical_cols

unused_feat = []

features = [ col for col in X.columns if col not in unused_feat+[target]] 

cat_idxs = [ i for i, f in enumerate(features) if f in categorical_columns]

cat_dims = [ categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns]

print(f"cat_idxs values are: {cat_idxs}")
print(f"cat_dims values are: {cat_dims}")

from pytorch_tabnet.tab_model import TabNetClassifier


tabnet_fit_params = {
    'max_epochs': 100,
    'patience': 10,
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
    cat_idxs=cat_idxs,
    cat_dims=cat_dims,
    cat_emb_dim=1,
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