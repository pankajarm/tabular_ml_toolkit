    # do oof k-fold train and predictions
    def do_oof_kfold_train_preds(self, X:object, y:object, n_splits:int, model:object, X_test:object=None,
                                 tabnet_params:dict=None, random_state=42):
        """
            This methods returns oof_preds and test_preds by doing kfold training on sklearn pipeline
            n_splits=5 by default, takes only int value
            random_sate=42, takes only int value

        """
        #TODO: Find better way without string matching
        if "tabnet" in str(model.__class__):
            #fetch problem type params
            _,_, eval_metric,_,oof_val_metric = fetch_tabnet_params_for_problem_type(self.problem_type)        
        
        elif "xgb" in str(model.__class__):
            #fetch problem type params
            _,_, eval_metric,_,oof_val_metric = fetch_xgb_params_for_problem_type(self.problem_type)
            
        else:
            #fetch problem type params
            _,_,oof_val_metric  = fetch_skl_params_for_problem_type(self.problem_type)
        
        #create stratified K Folds instance
        kfold = StratifiedKFold(n_splits=n_splits,
                             random_state=random_state,
                             shuffle=True)

        # for oof preds
        oof_preds = np.zeros(X.shape[0])

        # check whether test dataset exist before test preds
        oof_test_preds = None
        if X_test is not None:
            oof_test_preds = np.zeros(X_test.shape[0])

        # k-fold training and predictions for oof predictions
        oof_model_metrics_mean = 0
        n=0
        for train_idx, valid_idx in kfold.split(X, y):
            if isinstance(X, np.ndarray):
                # create NUMPY based X_train, X_valid, y_train, y_valid
                # fix the stratification problem
                #train_idx,valid_idx = clip_splits(train_idx,valid_idx,self.dfl.X_full)
                X_train , X_valid, y_train, y_valid = X[train_idx], X[valid_idx], y[train_idx], y[valid_idx]
            else:
                # create PANDAS based X_train, X_valid, y_train, y_valid
                # fix the stratification problem
                #train_idx,valid_idx = clip_splits(train_idx,valid_idx,self.dfl.X_full)
                X_train , X_valid, y_train, y_valid = X.iloc[train_idx], X.iloc[valid_idx], y[train_idx], y[valid_idx]
                
            #TRAINING
            logger.info(f"Training Started!")
                        #should be better way without string matching
            if "tabnet" in str(model.__class__):
                #change for tabnet
                if  tabnet_params:
                    model.fit(X_train, y_train,
                              eval_set=[(X_valid, y_valid)],
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

            #VAL PREDICTIONS
            # getting either hyperplane distance or probablities from predictions
            if "svm" in str(model.__class__):
                #logger.info(f"Predicting Valid Decision!")
                oof_preds[valid_idx] = model.decision_function(X_valid)
            else:
                #logger.info(f"Predicting Valid Proba!")
                if "classification" in self.problem_type:
                    oof_preds[valid_idx] = model.predict_proba(X_valid)[:,1]
                    preds = model.predict_proba(X_valid)
                else:
                    oof_preds[valid_idx] = model.predict(X_valid)
                    preds = model.predict(X_valid)

            #METRICS
            # Get metric results for each fold
            oof_metric_results = oof_val_metric(y_valid, preds)
            logger.info(f"fold: {n+1} OOF {str(oof_val_metric.__name__)}: {oof_metric_results}!")
            
            # for mean score
            oof_model_results_mean += (oof_metric_results / kfold.n_splits)


            #TEST PREDICTIONS
            # appending mean test data predictions
            if X_test is not None:
                if "svm" in str(model.__class__):
                    oof_test_preds += model.decision_function(X_test) / kfold.n_splits
                else:
                    if "classification" in self.problem_type:
                        oof_test_preds += model.predict_proba(X_test)[:,1] / kfold.n_splits
                    else:
                        oof_test_preds += model.predict(X_test) / kfold.n_splits
            else:
                logger.warn(f"Trying to do OOF Test Predictions but No Test Dataset Provided!")

            # In order to better GC, del X_train, X_valid, y_train, y_valid df after each fold is done,
            # they will recreate again next time k-fold is called
            unused_df_lst = [X_train, X_valid, y_train, y_valid]
            del unused_df_lst
            gc.collect()

            # increment fold counter label
            n += 1

        #oof_model_auc_mean = (oof_model_auc / kfold.n_splits)
        logger.info(f"Mean OOF {str(oof_val_metric.__name__)}: {oof_model_results_mean}!")
        gc.collect()

        return oof_preds, oof_test_preds
    
