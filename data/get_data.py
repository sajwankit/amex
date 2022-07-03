import os
# import custom logger class.
import logging
logger = logging.getLogger(__name__)

import cudf

def read_file(path = '', usecols = None):
    # LOAD DATAFRAME
    if usecols is not None: df = cudf.read_parquet(path, columns=usecols)
    else: df = cudf.read_parquet(path)
    # REDUCE DTYPE FOR CUSTOMER AND DATE
    df['customer_ID'] = df['customer_ID'].str[-16:].str.hex_to_int().astype('int64')
    df.S_2 = cudf.to_datetime( df.S_2 )
    # SORT BY CUSTOMER AND DATE (so agg('last') works correctly)
    #df = df.sort_values(['customer_ID','S_2'])
    #df = df.reset_index(drop=True)
    # FILL NAN
    df = df.fillna(int(os.environ["FILL_NAN_VALUE"])) 
    logger.info(f"Train data shape: {df.shape}")
    return df

def add_targets(train):
    # ADD TARGETS
    targets = cudf.read_csv(f"{os.environ['DATA_BASE_DIR']}train_labels.csv")
    targets['customer_ID'] = targets['customer_ID'].str[-16:].str.hex_to_int().astype('int64')
    targets = targets.set_index('customer_ID')
    train = train.merge(targets, left_index=True, right_index=True, how='left')
    train.target = train.target.astype('int8')
    del targets

    # NEEDED TO MAKE CV DETERMINISTIC (cudf merge above randomly shuffles rows)
    train = train.sort_index().reset_index()

    return train