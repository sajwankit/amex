import xgboost as xgb

class XGBModel:
    # XGB MODEL PARAMETERS
    xgb_parms = { 
        'max_depth':4, 
        'learning_rate':0.05, 
        'subsample':0.8,
        'colsample_bytree':0.6, 
        'eval_metric':'logloss',
        'objective':'binary:logistic',
        'tree_method':'gpu_hist',
        'predictor':'gpu_predictor',
        'random_state':SEED
    }
    