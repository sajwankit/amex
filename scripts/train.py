import os, sys
import sys.path.append(".")

import logging
import logging.config
from utils.logging import LOGGING_CONFIG
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)
log = logging.getLogger('always_log')

from sklearn.model_selection import KFold

if __name__=="__main__":
    importances = []
    oof = []
    train = train.to_pandas() # free GPU memory
    TRAIN_SUBSAMPLE = 1.0
    gc.collect()

    skf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    for fold,(train_idx, valid_idx) in enumerate(skf.split(
                train, train.target )):
        
        # TRAIN WITH SUBSAMPLE OF TRAIN FOLD DATA
        if TRAIN_SUBSAMPLE<1.0:
            np.random.seed(SEED)
            train_idx = np.random.choice(train_idx, 
                        int(len(train_idx)*TRAIN_SUBSAMPLE), replace=False)
            np.random.seed(None)
        
        log.info('#'*25)
        log.info('### Fold',fold+1)
        log.info('### Train size',len(train_idx),'Valid size',len(valid_idx))
        log.info(f'### Training with {int(TRAIN_SUBSAMPLE*100)}% fold data...')
        log.info('#'*25)
        
        # TRAIN, VALID, TEST FOR FOLD K
        Xy_train = IterLoadForDMatrix(train.loc[train_idx], FEATURES, 'target')
        X_valid = train.loc[valid_idx, FEATURES]
        y_valid = train.loc[valid_idx, 'target']
        
        dtrain = xgb.DeviceQuantileDMatrix(Xy_train, max_bin=256)
        dvalid = xgb.DMatrix(data=X_valid, label=y_valid)
        
        # TRAIN MODEL FOLD K
        model = xgb.train(xgb_parms, 
                    dtrain=dtrain,
                    evals=[(dtrain,'train'),(dvalid,'valid')],
                    num_boost_round=9999,
                    early_stopping_rounds=100,
                    verbose_eval=100) 
        model.save_model(f'XGB_v{VER}_fold{fold}.xgb')
        
        # GET FEATURE IMPORTANCE FOR FOLD K
        dd = model.get_score(importance_type='weight')
        df = pd.DataFrame({'feature':dd.keys(),f'importance_{fold}':dd.values()})
        importances.append(df)
                
        # INFER OOF FOLD K
        oof_preds = model.predict(dvalid)
        acc = amex_metric_mod(y_valid.values, oof_preds)
        log.info('Kaggle Metric =',acc,'\n')
        
        # SAVE OOF
        df = train.loc[valid_idx, ['customer_ID','target'] ].copy()
        df['oof_pred'] = oof_preds
        oof.append( df )
        
        del dtrain, Xy_train, dd, df
        del X_valid, y_valid, dvalid, model
        _ = gc.collect()
        
    log.info('#'*25)
    oof = pd.concat(oof,axis=0,ignore_index=True).set_index('customer_ID')
    acc = amex_metric_mod(oof.target.values, oof.oof_pred.values)
    log.info('OVERALL CV Kaggle Metric =',acc)