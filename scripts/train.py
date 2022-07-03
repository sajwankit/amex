import os, sys
sys.path.append("../..")
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

# To debug in vscode and use env vars uncomment following.
from dotenv import load_dotenv
os.environ.clear()
load_dotenv() # load current project dir
load_dotenv(f"""{os.environ["PROJECT_DIR"]}/config/amex.env""")

import logging
import logging.config
from utils.logging import LOGGING_CONFIG
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

import gc, argparse, joblib
from sklearn.model_selection import KFold
import xgboost
import pandas as pd

from data.get_data import read_file, add_targets
from dataloader import IterLoadForDMatrix
from data.pipelines import AmexPreProcessPipeline
from utils.metrics import amex_metric_mod
from utils.callbacks import XGBLogging

def get_parser():
    parser = argparse.ArgumentParser(
        description="Amex training script",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # number of folds for training
    parser.add_argument(
        "--folds",
        type=int,
        required=False,
        help="number of folds",
        default=5 if "FOLDS" not in os.environ else os.environ["FOLDS"]
    )
    # common seed
    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        default=2022 if "SEED" not in os.environ else os.environ["SEED"]
    )
    # train subsample ratio
    parser.add_argument(
        "--subsample-ratio",
        type=float,
        required=False,
        default=1 if "TRAIN_SUBSAMPLE" not in os.environ else os.environ["TRAIN_SUBSAMPLE"]
    )
    # batch_size
    parser.add_argument(
        "--batch-size",
        type=int,
        required=False,
        default=256*1024 if "BATCH_SIZE" not in os.environ else os.environ["BATCH_SIZE"]
    )
    # preprocess pipeline name
    parser.add_argument(
        "--preprocess-pipe-name",
        type=str,
        required=False,
        default='AmexPreProcessPipeline'
    )
    # model type
    parser.add_argument(
        "--model",
        type=str,
        required=False,
        default='xgb' if "MODEL_TYPE" not in os.environ else os.environ["MODEL_TYPE"]
    )
    # model save version
    parser.add_argument(
        "--version",
        type=str,
        required=False,
        default='1'
    )
    return parser

class AmexTrainer:

    def __init__(
        self,
        model_type
    ):
        self.pipeline = AmexPreProcessPipeline()
        self.pipeline_path = f"""{os.environ["PIPELINE_DIR"]}{args.version}_{args.preprocess_pipe_name}.joblib"""
        self.model_type = model_type
        self.feature_importances = []
        self.oofs = []

    def get_tranformed_data(self, df, train=True):
        transformed_df = self.pipeline.fit_transform(df)
        if train:
            joblib.dump(self.pipeline, self.pipeline_path)
        return transformed_df
    
    def train_and_predict(self, train_df, train_idx, valid_idx, model_save_path, predict=True):
        if args.model == "xgb":
            features = train_df.columns[1:-1]
            Xy_train = IterLoadForDMatrix(
                train_df.loc[train_idx],
                features=features,
                target='target',
                batch_size=args.batch_size
            )
            # X_train = train_df.loc[train_idx, features]
            # y_train = train_df.loc[train_idx, 'target']
            X_valid = train_df.loc[valid_idx, features]
            y_valid = train_df.loc[valid_idx, 'target']
            dtrain=xgboost.DeviceQuantileDMatrix(Xy_train, max_bin=256)
            # dtrain = xgboost.DMatrix(data=X_train, label=y_train)
            dvalid=xgboost.DMatrix(data=X_valid, label=y_valid)

            xgboost_parms = { 
            'max_depth':4, 
            'learning_rate':0.05, 
            'subsample':0.8,
            'colsample_bytree':0.6, 
            'eval_metric':'logloss',
            'objective':'binary:logistic',
            'tree_method':'gpu_hist',
            'predictor':'gpu_predictor',
            'random_state': args.seed
            }
            progress = dict()
            model = xgboost.train(
                xgboost_parms, 
                dtrain=dtrain,
                evals=[(dtrain,"train"),(dvalid,"valid")],
                num_boost_round=9999,
                early_stopping_rounds=100,
                verbose_eval=100,
                evals_result=progress,
                callbacks=[XGBLogging()]
            ) 
            model.save_model(model_save_path)
            logger.info(f"xgboost model training done, model saved at: {model_save_path}")

            if predict:
                oof_preds = model.predict(dvalid)
                acc = amex_metric_mod(y_valid.values, oof_preds)
                logger.info(f"Kaggle Metric = {acc}")
                # SAVE OOF
                df = train_df.loc[valid_idx, ['customer_ID','target'] ].copy()
                df['oof_pred'] = oof_preds
                self.oofs.append(df)
                
                del dtrain, Xy_train, df
                del X_valid, y_valid, dvalid
                _ = gc.collect()
            
            return model

    def fold_train(self, train_df):
        train_df = self.get_tranformed_data(df=train_df, train=True)
        train_df = add_targets(train_df) # add targets after feature engineering
        train_df = train_df.to_pandas() # free GPU memory
        skf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        for fold,(train_idx, valid_idx) in enumerate(skf.split(train_df, train_df.target)):
            # TRAIN WITH SUBSAMPLE OF TRAIN FOLD DATA
            if args.subsample_ratio < 1.0:
                np.random.seed(args.seed)
                train_idx = np.random.choice(
                    train_idx,
                    int(len(train_idx)*args.subsample_ratio),
                    replace=False
                )
                np.random.seed(None)
            
            logger.info("#"*25)
            logger.info(f"### Fold {fold+1}")
            logger.info(f"### Train size: {len(train_idx)}, Valid size: {len(valid_idx)}")
            logger.info(f"""### Training with {int(args.subsample_ratio)*100}% fold data...""")
            logger.info("#"*25)

            # train model and get back trained model
            model = self.train_and_predict(
                train_df,
                train_idx,
                valid_idx,
                model_save_path=f"""{os.environ["MODEL_DIR"]}xgboost_v{args.version}_fold{fold}.xgboost"""
            )

            # GET FEATURE IMPORTANCE FOR FOLD K
            dd = model.get_score(importance_type='weight')
            fold_feature_imp_df = pd.DataFrame({'feature':dd.keys(),f'importance_{fold}':dd.values()})
            self.feature_importances.append(fold_feature_imp_df)
            
if __name__=="__main__":
    # define args
    parser = get_parser()
    args = parser.parse_args()

    # get train data
    train_path = f"""{os.environ["DATA_BASE_DIR"]}train.parquet"""
    train_df = read_file(path=train_path)

    trainer = AmexTrainer(model_type="xgb")
    trainer.fold_train(train_df)

    logger.info('#'*25)
    oof = pd.concat(trainer.oofs,axis=0,ignore_index=True).set_index('customer_ID')
    acc = amex_metric_mod(oof.target.values, oof.oof_pred.values)
    logger.info(f"OVERALL CV Kaggle Metric = {acc}")
