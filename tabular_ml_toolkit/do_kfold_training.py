    # do k-fold training
    def do_kfold_training(self, X:object, y:object, n_splits:int, model:object, metric:object,
                          X_test:object=None, tabnet_params:dict=None, random_state=42):
        
        """
            This methods returns kfold_metrics_results and test_preds by doing kfold training
            n_splits=5 by default, takes only int value
            random_sate=42, takes only int value

        """ 
        
        #logger.info(f" model class:{model.__class__}")
        #should be better way without string matching
        if "tabnet" in str(model.__class__):
            #fetch problem type params
            _, val_preds_metrics, eval_metric, _ = fetch_tabnet_params_for_problem_type(self.problem_type)        
        
        elif "xgb" in str(model.__class__):
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
            if isinstance(X, np.ndarray):
                # create NUMPY based X_train, X_valid, y_train, y_valid
                X_train , X_valid, y_train, y_valid = X[train_idx], X[valid_idx], y[train_idx], y[valid_idx]
            else:
                # create PANDAS based X_train, X_valid, y_train, y_valid
                X_train , X_valid, y_train, y_valid = X.iloc[train_idx], X.iloc[valid_idx], y[train_idx], y[valid_idx]
            
            #TRAINING
            logger.info(f"Training Started!")
            #should be better way without string matching
            if "tabnet" in str(model.__class__):
                #change for tabnet
                model.fit(X_train, y_train,
                          eval_set=[(X_train, y_train), (X_valid, y_valid)],
                          eval_metric=[eval_metric],
                          **tabnet_params)        

            elif "xgb" in str(model.__class__):
                #change for xgb
                model.fit(X_train, y_train,
                          eval_set=[(X_train, y_train), (X_valid, y_valid)],
                          eval_metric=eval_metric,
                         verbose=False)

            else:
                #standard sklearn fit
                model.fit(X_train, y_train)
            
            logger.info(f"Training Finished!")
            
            #TO-DO: Merge OOF_KFold and KFold methods at this level
            metric_result = {}
            preds = None
            
            #VAL PREDICTIONS
            if "svm" in str(model.__class__):
                logger.info("Predicting Val Decision Function!")
                preds = model.decision_function(X_valid)
            else:
                if needs_predict_proba(metric):
                    logger.info("Predicting Val Probablities!")
                    preds = model.predict_proba(X_valid)[:, 1]
                else:
                    logger.info("Predicting Val Score!")
                    preds = model.predict(X_valid)
            
            #METRIC CALCULATION
            metric_result[str(metric.__name__)] = metric(y_valid, preds)

            #now show value of all the given metrics
            for metric_name, metric_value in metric_result.items():
                logger.info(f"fold: {n+1} {metric_name} : {metric_value}")
            
            #now append each kfold metric_result dict to list
            kfold_metrics_results.append(metric_result)
            
            ##TEST PREDICTIONS
            if X_test is not None:
                #                 if "classification" in self.problem_type:
                #                     logger.info("Predicting Test Probablities!")
                #                     test_preds += model.predict_proba(X_test)[:,1] / kfold.n_splits
                #                 else:
                logger.info("Predicting Test Scores!")
                test_preds += model.predict(X_test) / kfold.n_splits
            else:
                logger.warn(f"Trying to do Test Predictions but No Test Dataset Provided!")

            # In order to better GC, del X_train, X_valid, y_train, y_valid df after each fold is done,
            # they will recreate again next time k-fold is called
            unused_df_lst = [X_train, X_valid, y_train, y_valid]
            del unused_df_lst
            gc.collect()
            
            # increment fold counter label
            n += 1
        
        #logger.info(f"kfold_metrics_results: {kfold_metrics_results} ")
        mean_metrics_results = kfold_dict_mean(kfold_metrics_results)
        logger.info(f" Mean Metrics Results from all Folds are: {mean_metrics_results}")
        gc.collect()
        return mean_metrics_results, test_preds