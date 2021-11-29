# AUTOGENERATED! DO NOT EDIT! File to edit: 02_tmlt.ipynb (unless otherwise specified).

__all__ = ['TMLT']

# Cell
from .dataframeloader import *
from .preprocessor import *
from .logger import *
from .xgb_optuna_objective import *
from .utility import *

# Cell
# hide
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
# for Optuna
import optuna
#for XGB
import xgboost

# for finding n_jobs in all sklearn estimators
from sklearn.utils import all_estimators
import inspect

# Just to compare fit times
import time

# for os specific settings
import os
from shutil import rmtree

# Cell

class TMLT:
    """
    Represent Tabular ML Toolkit class

    Attributes:\n
    spl: A Scikit MLPipeline instance \n
    dfl: A DataFrameLoader instance \n
    pp: A PreProcessor instance \n
    model: The given Model
    """

    def __init__(self):
        self.dfl = None
        self.pp = None
        self.model = None
        self.spl = None
        self.transformer_type = None
        self.problem_type = None
        self.has_n_jobs = check_has_n_jobs()
        self.IDEAL_CPU_CORES = find_ideal_cpu_cores()


    def __str__(self):
        """Returns human readable string reprsentation"""
        attr_str = ("spl, dfl, pp, model")
        return ("Training Pipeline object with attributes:"+attr_str)

    def __repr__(self):
        return self.__str__()

    ## All Core Methods ##

    # Bundle preprocessing and modeling code in a training pipeline
    def create_final_sklearn_pipeline(self, transformer_type, model):
        self.spl = Pipeline(
            steps=[('preprocessor', transformer_type), ('model', model)],
            memory="pipeline_cache_dir")
        return self.spl

    # Main Method to create, load, preprocessed data based upon problem type
    def prepare_data_for_training(self, train_file_path:str,
                                  problem_type:str,
                                  idx_col:str, target:str,
                                  random_state:int,
                                  model:object,
                                  test_file_path:str=None,
                                  nrows=None):
        #set problem type
        self.problem_type = problem_type
        # check if given model supports n_jobs aka cpu core based Parallelism
        estimator_name = model.__class__.__name__
        # logger.info(estimator_name)
        # logger.info((self.has_n_jobs)
        if estimator_name in self.has_n_jobs :
            # In order to OS not to kill the job, leave one processor out
            model.n_jobs = self.IDEAL_CPU_CORES
            self.model = model
        else:
            print(f"{estimator_name} doesn't support parallelism yet! Training will continue on a single thread.")
            self.model = model

        # call DataFrameLoader module
        self.dfl = DataFrameLoader().from_csv(
            train_file_path=train_file_path,
            test_file_path=test_file_path,
            idx_col=idx_col,
            target=target,
            random_state=random_state,
            nrows=nrows)

        # call PreProcessor module
        self.pp = PreProcessor().preprocess_all_cols(dataframeloader=self.dfl, problem_type=self.problem_type)

        # call create final sklearn pipelien method
        self.spl = self.create_final_sklearn_pipeline(transformer_type=self.pp.transformer_type,
                                     model = model)
        # return tmlt
        return self

    # Force to update the dataframeloader in pipeline
    def update_dfl(self, X:object, y:object, X_test:object,
                   num_cols__imputer=SimpleImputer(strategy='median'),
                   num_cols__scaler=StandardScaler(),
                   cat_cols__imputer=SimpleImputer(strategy='constant'),
                   cat_cols__encoder=OneHotEncoder(handle_unknown='ignore')):

        # remove the old pipeline_cach directory
        if os.path.isdir("pipeline_cache_dir"):
            rmtree("pipeline_cache_dir")

        # regenerate dfl
        self.dfl = DataFrameLoader().regenerate_dfl(X,y,X_test)

        # change preprocessor
        self.pp = PreProcessor().preprocess_all_cols(self.dfl,
                                                     num_cols__imputer=num_cols__imputer,
                                                     num_cols__scaler=num_cols__scaler,
                                                     cat_cols__imputer=cat_cols__imputer,
                                                     cat_cols__encoder=cat_cols__encoder)
        # regenerate create final sklearn pipeline because of model update
        self.spl = self.create_final_sklearn_pipeline(transformer_type=self.pp.transformer_type,
                                     model = self.model)
        return self

    # Force to update the preprocessor in pipeline
    def update_preprocessor(self,
                            num_cols__imputer=SimpleImputer(strategy='median'),
                            num_cols__scaler=StandardScaler(),
                            cat_cols__imputer=SimpleImputer(strategy='constant'),
                            cat_cols__encoder=OneHotEncoder(handle_unknown='ignore')):

        # remove the old pipeline_cach directory
        if os.path.isdir("pipeline_cache_dir"):
            rmtree("pipeline_cache_dir")

        # change preprocessor
        self.pp = PreProcessor().preprocess_all_cols(self.dfl,
                                                     num_cols__imputer=num_cols__imputer,
                                                     num_cols__scaler=num_cols__scaler,
                                                     cat_cols__imputer=cat_cols__imputer,
                                                     cat_cols__encoder=cat_cols__encoder)
        # recall create final sklearn pipelien method
        self.spl = self.create_final_sklearn_pipeline(transformer_type=self.pp.transformer_type,
                                     model = self.model)
        return self


    # Force to update the model in pipeline
    def update_model(self, model:object):

        # remove the old pipeline_cach directory
        if os.path.isdir("pipeline_cache_dir"):
            rmtree("pipeline_cache_dir")

        #change model
        self.model = model

        # regenerate create final sklearn pipeline because of model update
        self.spl = self.create_final_sklearn_pipeline(transformer_type=self.pp.transformer_type, model = model)
        return self

    # cross validation
    def do_cross_validation(self, scoring:str, cv:int=5):
        """
        scoring take str which are predefined here from https://scikit-learn.org/stable/modules/model_evaluation.html
        cv takes int by default 5
        """
        scores = cross_val_score(
            estimator=self.spl,
            X=self.dfl.X,
            y=self.dfl.y,
            scoring=scoring,
            cv=cv)
        # Multiply by -1 since sklearn calculates *negative* scoring for some of the metrics
        if "neg_" in scoring:
            scores = -1 * scores
        return scores

    # GridSearch
    def do_grid_search(self, param_grid:object, cv:int,
                       scoring:str, n_jobs=None):

        if n_jobs is None:
            n_jobs = self.IDEAL_CPU_CORES

        # create GridSeachCV instance
        grid_search = GridSearchCV(estimator=self.spl,
                                   param_grid=param_grid,
                                   cv=cv,
                                   scoring=scoring,
                                   n_jobs=n_jobs)
        # now call fit
        grid_search.fit(self.dfl.X, self.dfl.y)
        return grid_search

    # do oof predictions using k-fold training
    # current supported model type is LinearSVM
    def do_oof_kfold_train_preds(self, n_splits:int, random_state=42):
        """
            This methods returns oof_preds and test_preds by doing kfold training on sklearn pipeline
            n_splits=5 by default, takes only int value
            random_sate=42, takes only int value

        """
        #fetch problem type params
        #_, val_preds_metrics, _, _ = fetch_xgb_params_for_problem_type(self.problem_type)

        #create stratified K Folds instance
        kfold = StratifiedKFold(n_splits=n_splits,
                             random_state=random_state,
                             shuffle=True)

        # for oof preds
        oof_preds = np.zeros(self.dfl.X.shape[0])

        # check whether test dataset exist before test preds
        oof_test_preds = None
        if self.dfl.X_test is not None:
            oof_test_preds = np.zeros(self.dfl.X_test.shape[0])

        # k-fold training and predictions for oof predictions
        oof_model_auc_mean = 0
        n=0
        for train_idx, valid_idx in kfold.split(self.dfl.X, self.dfl.y):
            # create X_train
            self.dfl.X_train = self.dfl.X.iloc[train_idx]
            # create X_valid
            self.dfl.X_valid = self.dfl.X.iloc[valid_idx]
            # create y_train
            self.dfl.y_train = self.dfl.y[train_idx]
            # create y_valid
            self.dfl.y_valid = self.dfl.y[valid_idx]

            #logger.info(f"self.dfl.X_train.columns.to_list: {self.dfl.X_train.columns.to_list()}")
            #logger.info(f"Training Started!")
            #simple pipeline fit
            self.spl.fit(self.dfl.X_train, self.dfl.y_train)

            #logger.info(f"Training Done!")
            # getting either hyperplane distance or probablities from predictions
            if "svm" in str(self.model.__class__):
                #logger.info(f"Predicting Valid Decision!")
                #BUG HERE FOR WARNING https://github.com/scikit-learn/scikit-learn/pull/21578
                oof_preds[valid_idx] = self.model.decision_function(self.dfl.X_valid)
            else:
                #logger.info(f"Predicting Valid Proba!")
                oof_preds[valid_idx] = self.model.predict_proba(self.dfl.X_valid)[:,1]

            # Getting linear model metric results for each fold
            oof_model_auc = roc_auc_score(self.dfl.y_valid, oof_preds[valid_idx])
            logger.info(f"fold: {n+1} OOF Model ROC AUC: {oof_model_auc}!")
            # for mean score
            oof_model_auc_mean += (oof_model_auc / kfold.n_splits)


            # for test preds
            # appending mean test data predictions
            # i.e. for each fold trained model get average test prediction and add them
            if self.dfl.X_test is not None:
                if "svm" in str(self.model.__class__):
                    oof_test_preds += self.model.decision_function(self.dfl.X_test) / kfold.n_splits
                else:
                    oof_test_preds += self.model.predict_proba(self.dfl.X_test)[:,1] / kfold.n_splits
            else:
                logger.warn(f"Trying to do OOF Test Predictions but No Test Dataset Provided!")

            # In order to better GC, del X_train, X_valid, y_train, y_valid df after each fold is done,
            # they will recreate again next time k-fold is called
            unused_df_lst = [self.dfl.X_train, self.dfl.X_valid, self.dfl.y_train, self.dfl.y_valid]
            del unused_df_lst

            # increment fold counter label
            n += 1

        #oof_model_auc_mean = (oof_model_auc / kfold.n_splits)
        logger.info(f"Mean OOF Model ROC AUC: {oof_model_auc_mean}!")

        return oof_preds, oof_test_preds


    # do k-fold training
    # test_preds_metric has to be a single sklearn metrics object type such as mean_absoulte_error, acccuracy
    def do_kfold_training(self, n_splits:int, test_preds_metric=None, random_state=42):

        """
            This methods returns kfold_metrics_results and test_preds by doing kfold training
            test_preds_metric=None by default, takes only single SKLearn Metrics for your test dataset
            n_splits=5 by default, takes only int value
            random_sate=42, takes only int value

        """

        logger.info(f" model class:{self.model.__class__}")

        # KNOWN BUG WHILE USING EVAL SET https://github.com/dmlc/xgboost/issues/2334
        #logger.info(f"X_train.columns.values.tolist: {X_train.columns.values.tolist()}")
        #logger.info(f"X_valid.columns.values.tolist: {X_valid.columns.values.tolist()}")
        #convert X, X_test to numpy
        X, y, X_test = self.dfl.X, self.dfl.y, self.dfl.X_test

        #should be better way without string matching
        if "xgb" in str(self.model.__class__):
            #fetch problem type params
            _, val_preds_metrics, eval_metric, _ = fetch_xgb_params_for_problem_type(self.problem_type)

        else:
            #fetch problem type params
            val_preds_metrics, _ = fetch_skl_params_for_problem_type(self.problem_type)

        #create stratified K Folds instance
        kfold = StratifiedKFold(n_splits=n_splits,
                             random_state=random_state,
                             shuffle=True)

        # check for test dataset before prediction
        test_preds = None
        if X_test is not None:
            test_preds = np.zeros(X_test.shape[0])

        # list contains metrics results for each fold
        kfold_metrics_results = []
        n=0
        for train_idx, valid_idx in kfold.split(X, y):
            # create X_train
            X_train = X.iloc[train_idx]
            # create X_valid
            X_valid = X.iloc[valid_idx]
            # create y_train
            y_train = y[train_idx]
            # create y_valid
            y_valid = y[valid_idx]

            #self.update_X_train_X_valid_X_test_with_final_cols(self.final_cols)
            # fit
            if "xgb" in str(self.model.__class__):
                #use xgb model fit
                # KNOWN BUG WHILE USING EVAL SET https://github.com/dmlc/xgboost/issues/2334
                #                 self.dfl.X_train = self.dfl.X_train.to_numpy()
                #                 self.dfl.X_valid = self.dfl.X_valid.to_numpy()
                #logger.info(f"X_train.columns.values.tolist: {X_train.columns.values.tolist()}")
                #logger.info(f"X_valid.columns.values.tolist: {X_valid.columns.values.tolist()}")
                self.spl.fit(X_train, y_train,
                             #there is known bug with xgboost and sklearn pipeline https://github.com/dmlc/xgboost/issues/2334
                             #model__eval_set=[(X_train, y_train), (X_valid, y_valid)],
                             model__eval_metric=eval_metric
                            )
            else:
                #simple fit for sklearn
                self.spl.fit(X_train, y_train)

            #simple pipeline fit
            #self.spl.fit(X_train, y_train)

            #TO-DO instead of single metrics use list of metrics and calculate mean using dict
            metric_result = {}

            # predictions
            for metric in val_preds_metrics:
                #TO-DO need to test and think about log_loss for SVM
                if ("log_loss" in str(metric.__name__)) or ("roc_auc_score" in str(metric.__name__)):
                    if "svm" in str(self.model.__class__):
                        #logger.info("Predicting Decision Function!")
                        preds_decs_func = self.spl.decision_function(X_valid) # CAN BE USE FOR OOF PREDS
                        metric_result[str(metric.__name__)] = metric(y_valid, preds_decs_func)
                    else:
                        #logger.info("Predicting Probablities!")
                        preds_probs = self.spl.predict_proba(X_valid)[:, 1] # CAN BE USE FOR OOF PREDS
                        metric_result[str(metric.__name__)] = metric(y_valid, preds_probs)

                else:
                    #logger.info("Predicting Score!")
                    preds = self.spl.predict(X_valid) # CAN BE USE FOR OOF PREDS
                    metric_result[str(metric.__name__)] = metric(y_valid, preds)

            #now show value of all the given metrics
            for metric_name, metric_value in metric_result.items():
                logger.info(f"fold: {n+1} {metric_name} : {metric_value}")

            #now append each kfold metric_result dict to list
            kfold_metrics_results.append(metric_result)

            # for test preds
            if X_test is not None and test_preds_metric is not None:
                if ("log_loss" in str(test_preds_metric.__name__)) or ("roc_auc_score" in str(test_preds_metric.__name__)):
                    logger.info("Predicting Test Preds Probablities!")
                    test_preds += self.spl.predict_proba(X_test)[:,1] / kfold.n_splits
                else:
                    test_preds += self.spl.predict(X_test) / kfold.n_splits
            elif self.dfl.X_test is None:
                logger.warn(f"Trying to do Test Predictions but No Test Dataset Provided!")

            # In order to better GC, del X_train, X_valid, y_train, y_valid df after each fold is done,
            # they will recreate again next time k-fold is called
            unused_df_lst = [X_train, X_valid, y_train, y_valid]
            del unused_df_lst

            # increment fold counter label
            n += 1

        #logger.info(f"kfold_metrics_results: {kfold_metrics_results} ")
        mean_metrics_results = kfold_dict_mean(kfold_metrics_results)
        logger.info(f" Mean Metrics Results from all Folds are: {mean_metrics_results}")

        return mean_metrics_results, test_preds

    # Do optuna bases study optimization for hyperparmaeter search
    def do_xgb_optuna_optimization(self, optuna_db_path:str, use_gpu=False, opt_trials=100,
                                   opt_timeout=360):
        """
            This methods returns and do optuna bases study optimization for hyperparmaeter search
            optuna_db_path is output directory you want to use for storing sql db used for optuna
            use_gpu=False by default, make it True if running on gpu machine
            opt_trials=100 by default, change it based upon need
            opt_timeout=360 by default, timeout value in seconds

        """

        # get params based on problem type
        xgb_model, val_preds_metrics, eval_metric, direction = fetch_xgb_params_for_problem_type(self.problem_type)

        # Load the dataset in advance for reusing it each trial execution.
        objective = XGB_Optuna_Objective(dfl=self.dfl, tmlt=self,
                                     val_preds_metrics=val_preds_metrics,
                                     xgb_model=xgb_model,
                                     xgb_eval_metric=eval_metric,
                                     use_gpu=use_gpu)
        # create sql db in optuna db path
        db_path = os.path.join(optuna_db_path, "params.db")

        # now create study
        logger.info(f"Optimization Direction is: {direction}")
        study = optuna.create_study(
            direction=direction,
            study_name="tmlt_autoxgb",
            storage=f"sqlite:///{db_path}",
            load_if_exists=True,
        )
        study.optimize(objective, n_trials=opt_trials, timeout=opt_timeout)
        return study


    # helper methods for users before updating preprocessor in pipeline
    def get_preprocessor_best_params_from_grid_search(self, grid_search_object:object):
        pp_best_params = {}
        for k in grid_search_object.best_params_:
            #print(k)
            if 'preprocessor' in k:
                key = k.split('__')[1] + "__" + k.split('__')[2]
                pp_best_params[key] = grid_search_object.best_params_[k]
        return pp_best_params

    # helper methods for users before updating model in pipeline
    def get_model_best_params_from_grid_search(self, grid_search_object:object):
        model_best_params = {}
        for k in grid_search_object.best_params_:
            #print(k)
            if 'model' in k:
                key = k.split('__')[1]
                model_best_params[key] = grid_search_object.best_params_[k]
        return model_best_params