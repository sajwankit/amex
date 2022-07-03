import os, sys
sys.path.append("../..")
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

# To debug in vscode and use env vars uncomment following.
# # os.environ.clear()
# from dotenv import load_dotenv
# load_dotenv() # load current project dir
# load_dotenv(f"""{os.environ["PROJECT_DIR"]}/config/amex.env""")
# os.environ["LOG_FILENAME"]="infer.log"
# os.environ["LOG_LEVEL"]="INFO"

import logging
import logging.config
from utils.logging import LOGGING_CONFIG
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

import gc, argparse, joblib, time
import xgboost
import cudf
import numpy as np

from data.get_data import read_file, add_targets
from data.pipelines import AmexPreProcessPipeline
from utils.metrics import amex_metric_mod

def get_parser():
    parser = argparse.ArgumentParser(
        description="Amex infer script",
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
    # test parts
    parser.add_argument(
        "--test-parts",
        type=int,
        required=False,
        default=4
    )
    # write submission file
    parser.add_argument(
        "--write-submission",
        type=int,
        required=False,
        default=1
    )
    return parser

def get_rows(customers, test, NUM_PARTS = 4, verbose = ''):
    """CALCULATE SIZE OF EACH SEPARATE TEST PART."""
    chunk = len(customers)//NUM_PARTS
    if verbose != '':
        logger.info(f'We will process {verbose} data as {NUM_PARTS} separate parts.')
        logger.info(f'There will be {chunk} customers in each part (except the last part).')
        logger.info('Below are number of rows in each part:')

    rows = []
    for k in range(NUM_PARTS):
        if k==NUM_PARTS-1: cc = customers[k*chunk:]
        else: cc = customers[k*chunk:(k+1)*chunk]
        s = test.loc[test.customer_ID.isin(cc)].shape[0]
        rows.append(s)
    if verbose != '': print( rows )
    return rows, chunk

def write_submission(test_preds, customers):
    # WRITE SUBMISSION FILE
    test_preds = np.concatenate(test_preds)
    test = cudf.DataFrame(index=customers, data={'prediction':test_preds})
    sub = cudf.read_csv(f"""{os.environ["DATA_BASE_DIR"]}sample_submission.csv""")[['customer_ID']]
    sub['customer_ID_hash'] = sub['customer_ID'].str[-16:].str.hex_to_int().astype('int64')
    sub = sub.set_index('customer_ID_hash')
    sub = sub.merge(test[['prediction']], left_index=True, right_index=True, how='left')
    sub = sub.reset_index(drop=True)

    # DISPLAY PREDICTIONS
    sub.to_csv(f"submission_{args.model}_v{args.version}.csv", index=False)
    logger.info(f"Submission file shape is: {sub.shape}")
    sub.head()

if __name__ == "__main__":
    job_start_time = time.time()

    parser = get_parser()
    args = parser.parse_args()

    # get test data
    test_path = f"""{os.environ["DATA_BASE_DIR"]}test.parquet"""
    test_df = read_file(path = test_path, usecols = ['customer_ID','S_2'])
    customers = test_df[['customer_ID']].drop_duplicates().sort_index().values.flatten()
    rows, num_cust = get_rows(customers, test_df[['customer_ID']], NUM_PARTS = args.test_parts, verbose = 'test')

    saved_pipeline_path = f"""{os.environ["PIPELINE_DIR"]}{args.version}_{args.preprocess_pipe_name}.joblib"""
    pipeline = joblib.load(saved_pipeline_path)

    skip_rows = 0
    skip_cust = 0
    test_preds = []
    for k in range(args.test_parts):
        # READ PART OF TEST DATA
        logger.info(f'Reading test data...')
        test = read_file(path=test_path)
        test = test.iloc[skip_rows:skip_rows+rows[k]]
        skip_rows += rows[k]
        logger.info(f"Test part {k+1} has shape: {test.shape}" )
        
        # PROCESS AND FEATURE ENGINEER PART OF TEST DATA
        test = pipeline.transform(test)
        if k==args.test_parts-1: test = test.loc[customers[skip_cust:]]
        else: test = test.loc[customers[skip_cust:skip_cust+num_cust]]
        skip_cust += num_cust
        
        if "xgb" in args.model:
            # TEST DATA FOR XGB
            X_test = test[test.columns[1:-1]] # TODO
            dtest = xgboost.DMatrix(data=X_test)
            test = test[['P_2_mean']] # reduce memory
            del X_test
            gc.collect()

            # INFER XGB MODELS ON TEST DATA
            model = xgboost.Booster()
            model.load_model(f"""{os.environ["MODEL_DIR"]}{args.model}_v{args.version}_fold0.{args.model}""")
            preds = model.predict(dtest)
            for fold in range(1,args.folds):
                model.load_model(f"""{os.environ["MODEL_DIR"]}{args.model}_v{args.version}_fold{fold}.{args.model}""")
                preds += model.predict(dtest)
            preds /= args.folds
            test_preds.append(preds)
            logger.info(f"Test preds generated for part: {k+1}")

            # CLEAN MEMORY
            del dtest, model
            _ = gc.collect()

            if args.write_submission == 1:
                # write submission file
                write_submission(test_preds, customers)

    logger.info(f"Infer job completed. Time Taken: {(time.time() - job_start_time)/60} minutes.")

