# AUTOGENERATED! DO NOT EDIT! File to edit: 02_tmlt.ipynb (unless otherwise specified).

__all__ = ['TMLT']

# Cell
from .dataframeloader import *
from .preprocessor import *
from .logger import *
from .optuna_objective import *

# Cell
# hide
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, roc_auc_score,accuracy_score
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
# for tune based GridSearch
from tune_sklearn import TuneGridSearchCV
# for Optuna
import optuna
#for XGB
import xgboost


# for displaying diagram of pipelines
from sklearn import set_config
set_config(display="diagram")

# for finding n_jobs in all sklearn estimators
from sklearn.utils import all_estimators
import inspect

# Just to compare fit times
import time

# for os specific settings
import os

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
        self.has_n_jobs = self.create_has_n_jobs()
        self.IDEAL_CPU_CORES = self.find_ideal_cpu_cores()


    def __str__(self):
        """Returns human readable string reprsentation"""
        attr_str = ("spl, dfl, pp, model")
        return ("Training Pipeline object with attributes:"+attr_str)

    def __repr__(self):
        return self.__str__()

    #helper method to find ideal cpu cores
    def find_ideal_cpu_cores(self):
        if os.cpu_count() > 2:
            ideal_cpu_cores = os.cpu_count()-1
            logger.info(f"{os.cpu_count()} cores found, parallel processing is enabled!")
        else:
            ideal_cpu_cores = None
            logger.info(f"{os.cpu_count()} cores found, parallel processing NOT enabled!")
        return ideal_cpu_cores

    #Helper method to find all sklearn estimators with support for parallelism aka n_jobs
    def create_has_n_jobs(self):
        self.has_n_jobs = ['XGBRegressor', 'XGBClassifier']
        for est in all_estimators():
            s = inspect.signature(est[1])
            if 'n_jobs' in s.parameters:
                self.has_n_jobs.append(est[0])
        return self.has_n_jobs

    # core methods

    # Bundle preprocessing and modeling code in a training pipeline
    def create_final_sklearn_pipeline(self, transformer_type, model):
        self.spl = Pipeline(
            steps=[('preprocessor', transformer_type),
                   ('model', model)])
        return self.spl

    # Core methods for Simple Training
    def prepare_data_for_training(self, train_file_path:str,
                                  idx_col:str, target:str,
                                  random_state:int,
                                  model:object,
                                  test_file_path:str=None,
                                 problem_type="regression"):
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
            random_state=random_state)

        # call PreProcessor module
        #TODO: For problem type classification, encode target using PreProcessor
        self.pp = PreProcessor().preprocess_all_cols(dataframeloader=self.dfl, problem_type=self.problem_type)

        # call create final sklearn pipelien method
        self.spl = self.create_final_sklearn_pipeline(transformer_type=self.pp.transformer_type,
                                     model = model)
        # return MLPipeline
        return self

    # Force to update the preprocessor in pipeline
    def update_preprocessor(self,
                            num_cols__imputer=SimpleImputer(strategy='median'),
                            num_cols__scaler=StandardScaler(),
                            cat_cols__imputer=SimpleImputer(strategy='constant'),
                            cat_cols__encoder=OneHotEncoder(handle_unknown='ignore')):
        # change preprocessor
        self.pp = PreProcessor().preprocess_all_cols(self.dfl,
                                                     num_cols__imputer=num_cols__imputer,
                                                     num_cols__scaler=num_cols__scaler,
                                                     cat_cols__imputer=cat_cols__imputer,
                                                     cat_cols__encoder=cat_cols__encoder)
        # recall create final sklearn pipelien method
        self.spl = self.create_final_sklearn_pipeline(transformer_type=self.pp.transformer_type,
                                     model = self.model)


    # Force to update the model in pipeline
    def update_model(self, model:object):
        #change model
        self.model = model
        # recall create final sklearn pipelien method
        self.spl = self.create_final_sklearn_pipeline(transformer_type=self.pp.transformer_type,
                                     model = self.model)

    # HELPER METHODS
    # cross validation
    def do_cross_validation(self, cv:int, scoring:str):
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

    # Core methods for GridSearch
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

    # Core methods for Tune SK-Learn GridSearch
    def do_tune_grid_search(self,
                            param_grid:object,
                            scoring:str=None,
                            mode:str='max',
                            cv:int=5,
                            early_stopping=True,
                            time_budget_s:int=None,
                            name:str=None,
                            use_gpu:bool=False,
                            stopper:object=None,
                            max_iters:int=10,
                            n_jobs=None):

        if n_jobs is None:
            n_jobs = self.IDEAL_CPU_CORES

        # create GridSeachCV instance
        tune_search = TuneGridSearchCV(
            estimator=self.spl,
            param_grid=param_grid,
            scoring=scoring,
            mode=mode,
            cv=cv,
            time_budget_s=time_budget_s,
            name=name,
            use_gpu=use_gpu,
            early_stopping=early_stopping,
            stopper=stopper,
            max_iters=max_iters,
            n_jobs=n_jobs,
            pipeline_auto_early_stop=True)

        # now call fit
        tune_search.fit(self.dfl.X, self.dfl.y)
        return tune_search


    # do k-fold training
    # metrics has to be sklearn metrics object type such as mean_absoulte_error, acccuracy
    def do_kfold_training(self, n_splits:int, metrics:object, random_state=42):

        #create stratified K Folds instance
        kfold = StratifiedKFold(n_splits=n_splits,
                             random_state=random_state,
                             shuffle=True)

        # check for test dataset before prediction
        test_preds = None
        if self.dfl.X_test is not None:
            test_preds = np.zeros(self.dfl.X_test.shape[0])
        # list contains metrics score for each fold
        metrics_score = []
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

            # fit
            #TODO use early_stopping_rounds = True for XGBoost based Sklearn Pipeline
            self.spl.fit(self.dfl.X_train, self.dfl.y_train)

            #TODO CHANGE HERE FOR multi metrics calculation, i.e. metrics provided in list
            #evaluate metrics based upon input
            if "proba" in metrics.__globals__:
                # predictions on valid dataset
                metrics_score.append(metrics(self.dfl.y_valid,
                                               self.spl.predict_proba(self.dfl.X_valid)[:,1]))
                if self.dfl.X_test is not None:
                    # prediction probabs on test dataset
                    test_preds += self.spl.predict_proba(self.dfl.X_test)[:,1] / kfold.n_splits
            else:
                metrics_score.append(metrics(self.dfl.y_valid,
                                               self.spl.predict(self.dfl.X_valid)))
                if self.dfl.X_test is not None:
                    # predictions on test dataset
                    test_preds += self.spl.predict(self.dfl.X_test) / kfold.n_splits

            logger.info(f"fold: {n+1} , {str(metrics.__name__)}: {metrics_score[n]}")
            # In order to better GC, del X_train, X_valid, y_train, y_valid df after each fold is done,
            # they will recreate again next time k-fold is called
            unused_df_lst = [self.dfl.X_train, self.dfl.X_valid, self.dfl.y_train, self.dfl.y_valid]
            del unused_df_lst
            # increment fold counter label
            n += 1

        mean_metrics_score = np.mean(metrics_score)
        logger.info(f" mean metrics score: {mean_metrics_score}")

        return metrics_score, test_preds

    # do optuna bases study optimization for hyperparmaeter search

    def do_xgb_optuna_optimization(self, metrics, output_dir_path:str, use_gpu=False, opt_trials=100,
                                   opt_timeout=360):
        """
            This methods returns and do optuna bases study optimization for hyperparmaeter search
            xgb_eval_metric string reprsenting "mae", "rmse", "logloss"
            kfold_metrics need to be sklearn metrics object type some of them are:
                from sklearn.metrics import mean_absolute_error, roc_auc_score,accuracy_score
            kfold_splits should be int, default is 5
            output_dir_path is output directory you want to use for storing sql db used for optuna
            use_gpu=False by default, make it True if running on gpu machine
            opt_trials=100 by default, change it based upon need
            opt_timeout=360 by default, timeout value in seconds

        """

        # get xgb p
        xgb_model, use_predict_proba, eval_metric, direction = self.fetch_xgb_model_params()

        # Load the dataset in advance for reusing it each trial execution.
        objective = Optuna_Objective(dfl=self.dfl, tmlt=self,
                                     metrics=metrics,
                                     xgb_model=xgb_model,
                                     xgb_eval_metric=eval_metric,
                                     use_predict_proba=use_predict_proba,
                                     use_gpu=use_gpu)
        # create sql db in output directory path
        db_path = os.path.join(output_dir_path, "params.db")

        # now create study
        logger.info(f"direction is: {direction}")
        study = optuna.create_study(
            direction=direction,
            study_name="tmlt_autoxgb",
            storage=f"sqlite:///{db_path}",
            load_if_exists=True,
        )
        study.optimize(objective, n_trials=opt_trials, timeout=opt_timeout)
        return study


    # Taken from AutoXGB Library, Thanks to https://github.com/abhishekkrthakur
    def fetch_xgb_model_params(self):
        if self.problem_type == "classification":
            xgb_model = xgboost.XGBClassifier
            use_predict_proba = True
            direction = "minimize"
            eval_metric = "logloss"
        elif self.problem_type == "multi_class_classification":
            xgb_model = xgboost.XGBClassifier
            use_predict_proba = True
            direction = "minimize"
            eval_metric = "mlogloss"
        elif self.problem_type == "regression":
            xgb_model = xgboost.XGBRegressor
            use_predict_proba = False
            direction = "minimize"
            eval_metric = "rmse"
        else:
            raise NotImplementedError

        return xgb_model, use_predict_proba, eval_metric, direction

    # helper method for updating preprocessor in pipeline
    # to create params value dict from grid_search object
    def get_preprocessor_best_params_from_grid_search(self, grid_search_object:object):
        pp_best_params = {}
        for k in grid_search_object.best_params_:
            #print(k)
            if 'preprocessor' in k:
                key = k.split('__')[1] + "__" + k.split('__')[2]
                pp_best_params[key] = grid_search_object.best_params_[k]
        return pp_best_params

    # helper method for update_model
    def get_model_best_params_from_grid_search(self, grid_search_object:object):
        model_best_params = {}
        for k in grid_search_object.best_params_:
            #print(k)
            if 'model' in k:
                key = k.split('__')[1]
                model_best_params[key] = grid_search_object.best_params_[k]
        return model_best_params