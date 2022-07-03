import logging
logger = logging.getLogger(__name__)

import xgboost

class XGBLogging(xgboost.callback.TrainingCallback):
    """log train logs to file"""

    def __init__(self, epoch_log_interval=100):
        self.epoch_log_interval = epoch_log_interval

    def after_iteration(self, model, epoch, evals_log):
        if epoch % self.epoch_log_interval == 0:
            for data, metric in evals_log.items():
                metrics = list(metric.keys())
                metrics_str = ""
                for m_key in metrics:
                    metrics_str = metrics_str + f"{m_key}: {metric[m_key][-1]}"
                logger.info(f"Epoch: {epoch}, {data}: {metrics_str}")
        # False to indicate training should not stop.
        return False