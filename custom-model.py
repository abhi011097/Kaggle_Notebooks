# %% [code]
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold ,KFold
import numpy as np
import pandas as pd


def getFoldData(fold_type,data,targetColumn,random_number,nsplits):
    
    loc_data=data.copy()
    
    if fold_type==1:
        kf = KFold(n_splits=nsplits, shuffle=True, random_state=random_number) 
        for fold, (train_index , valid_index) in enumerate(kf.split(X=loc_data)):
            loc_data.loc[valid_index, 'kfold'] = fold
    
    elif fold_type==2 :
        kf = StratifiedKFold(n_splits=nsplits, shuffle=True, random_state=random_number) 
        for fold, (train_index , valid_index) in enumerate(kf.split(X=loc_data, y=loc_data[targetColumn])):
            loc_data.loc[valid_index, 'kfold'] = fold
            
    elif fold_type==3 :        
        loc_data=data.copy()
        num_bins = int(np.floor(1 + np.log2(len(loc_data))))
        
        # bin targets
        loc_data.loc[:, "bins"] = pd.cut(loc_data[targetColumn], bins=num_bins, labels=False)
    
        kf = StratifiedKFold(n_splits=nsplits, shuffle=True, random_state=random_number)
        for fold, (train_index , valid_index) in enumerate(kf.split(X=loc_data, y=loc_data.bins.values)):
            loc_data.loc[valid_index, 'kfold'] = fold
    
        loc_data.drop(columns=['bins'],inplace=True)
    
    return loc_data



def execute_fold( model, model_type ,kfold, curr_data_kfold, test_data, target_column , probability=False,multi_class=False,class_metric_method='binary'):
    
    data_kfold    = curr_data_kfold.copy() # training data with kfold and target column
    test_preds=None
    
    
    print("*"*25,'Fold ', kfold,"*"*25)
    print()
    
    # Preparing Train Data from data_kfold
    x_kf_train = data_kfold[ data_kfold['kfold'] != kfold].drop(columns=[target_column,'kfold'])
    y_kf_train = data_kfold[ data_kfold['kfold'] != kfold][target_column]
    
    # Preparing Validation Data from data_kfold
    x_kf_valid = data_kfold[ data_kfold['kfold'] == kfold].drop(columns=[target_column,'kfold'])
    y_kf_valid = data_kfold[ data_kfold['kfold'] == kfold][target_column]
    
    # Training Model
    model.fit(x_kf_train,y_kf_train)
    
    # Regression Model Generating Validation Predictions for Current Fold 
    if model_type== 'R':
        valid_preds  = model.predict(x_kf_valid)
        
        # Generating Test Data Predictions if test_data is not none 
        if test_data is not None:
            test_preds = model.predict(test_data)
        
        # Metrics for regression Model
        rmse         = np.sqrt(metrics.mean_squared_error(y_kf_valid,valid_preds))       
        r2_score     = metrics.r2_score(y_kf_valid, valid_preds)
        rmsle        = np.sqrt(metrics.mean_squared_log_error(y_kf_valid, valid_preds))
        
        # for returning 
        fold_performance ={'rmse':rmse, 'r2_score':r2_score , 'rmsle':rmsle }
        
        # Printing Stuff
        print('Rmse - >',round(rmse,3),' | R2_Score - >',round(r2_score,3),' | Rmsle - >',round(rmsle,3))
        print()
        
        del rmse, r2_score , rmsle
        
        
    # Classification Model Generating Validation Predictions for Current Fold     
    if model_type== 'C':
        
        valid_preds_1 = model.predict(x_kf_valid)
        valid_preds_2 = model.predict_proba(x_kf_valid)[:,1]
        
        accuracy   = metrics.accuracy_score(y_kf_valid,valid_preds_1) 
        precision  = metrics.precision_score(y_kf_valid,valid_preds_1, average=class_metric_method)   
        recall     = metrics.recall_score(y_kf_valid, valid_preds_1, average=class_metric_method)
        f1         = metrics.f1_score(y_kf_valid, valid_preds_1, average=class_metric_method)
        
        if multi_class:
            roc_auc    = np.nan
        else :
            roc_auc    = metrics.roc_auc_score(y_kf_valid, valid_preds_2)
            
        
        if probability and (not multi_class):
            valid_preds = valid_preds_2
            # Generating Test Data Predictions if test_data is not none 
            if test_data is not None:
                test_preds = model.predict_proba(test_data)[:,1]
            
        else:
            valid_preds = valid_preds_1
            # Generating Test Data Predictions if test_data is not none 
            if test_data is not None:
                test_preds = model.predict(test_data)
        
        
        # for returning 
        fold_performance ={'accuracy':accuracy, 'precision':precision , 'recall':recall ,'f1':f1,
                           'roc_auc':roc_auc }
        
        # Printing Stuff
        print('Accuracy - >',round(accuracy,3),' | Precision - >',round(precision,3),' | Recall - >',
          round(recall,3),' | F1_Score - >',round(f1,3),' | Roc_Auc_Score - >',round(roc_auc,3))
        print()
        print("Confusion Matrix \n\n" , metrics.confusion_matrix(y_kf_valid, valid_preds_1))
        print("\nClassifiaction Metrics \n\n" , metrics.classification_report(y_kf_valid, valid_preds_1))
        
        del  accuracy, precision, recall, f1, roc_auc 
    
    # Feature_Importance
    
    try :
        fe_imp= model.feature_importances_
    except :
        fe_imp = np.empty((x_kf_train.shape[1],))
        fe_imp[:] = np.nan
        
    # x_kf_valid.index -> return  index of current fold 
    # valid_preds      -> return predictions done on current fold
    # test_preds      -> return predictions for test set if not empty
    # fold_performance -> return metric dictionary -> fold_performance
    # fe_imp -> return feature importance as per this fold
    
    return x_kf_valid.index, valid_preds, test_preds, fold_performance, fe_imp


def execute_models(df_train_valid,trainval_col , test_col, target_column,models_list,df_test=None, repeat_kfold=1,nsplits=5, kfold_type=2,
                   model_type='C',probability_enabled=False,multi_class_enabled=False,
                   multi_class_metric_mtd='macro',random_number=22
                  ):

    
    loc_df = df_train_valid.copy()
    final_training_preds = loc_df.loc[:,[target_column]]
    final_test_preds = pd.DataFrame()
    feature_imp = pd.DataFrame(index=test_col)
    foldwise_metric = pd.DataFrame()
    
    for i in range(repeat_kfold):
        print('*'*25)
        rand_state = random_number+(i)**2
       
        loc_data_kfold = custom_model.getFoldData(kfold_type,loc_df,target_column,rand_state,nsplits)
        
        for name, model in models_list.items():
            
            try:
                loc_model = model.set_params(random_state=rand_state)
            except:
                loc_model = model
                
            
            all_valid_scores_rmse= []
            
            
            loc_df_train_valid = loc_data_kfold[trainval_col+['kfold']].copy()
            
            if df_test is not None:
                loc_df_test = df_test[test_col].copy()
            else :
                loc_df_test = None
            
            
            
            print()
            print('*'+("-*"*15),'Current Model -> ',name,'*'+("-*"*15))
            print()
            
            for fld in range(nsplits):
                print('-'*100,'\n')
                print("Cycle Number : ",i+1," | Model Fold : ",fld +(nsplits * i),'  | Random State Number : ',rand_state,' | Current Model : ',name)
                print()
                index_x_kf_valid, valid_preds, test_preds ,metrics,fe_imp = custom_model.execute_fold(model=loc_model,
                                                                                model_type=model_type, 
                                                                                kfold=fld,
                                                                                curr_data_kfold=loc_df_train_valid , 
                                                                                test_data=loc_df_test , 
                                                                                target_column=target_column,
                                                                                probability=probability_enabled,
                                                                                multi_class=multi_class_enabled,
                                                                                class_metric_method=multi_class_metric_mtd)
                
                
                metrics['model']=name
                metrics['fold']=fld +(nsplits * i)
                final_training_preds.loc[index_x_kf_valid,name+'_'+str(i)]=valid_preds
                final_test_preds[name+'_'+str(fld +(nsplits * i))]=test_preds
                feature_imp[name+'_'+str(fld +(nsplits * i))]=fe_imp
                foldwise_metric = foldwise_metric.append(metrics, ignore_index = True)

                

    
    
    print('*'*25)
    
    return   final_training_preds,final_test_preds,feature_imp,foldwise_metric



# %% [code]
