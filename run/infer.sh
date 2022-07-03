export LOG_FILENAME=infer_4July2022.log
export LOG_LEVEL=INFO
export $(grep -v '^#' $HOME/amex/scripts/.env | xargs)
export $(grep -v '^#' $HOME/amex/config/amex.env | xargs)
python $HOME/amex/scripts/infer.py \
--folds 3 \
--seed 2022 \
--model xgboost \
--preprocess-pipe-name AmexPreProcessPipeline \
--version 1 \
--write-submission 0