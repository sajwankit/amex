import logging
logger = logging.getLogger(__name__)

from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin
import cudf

cat_features = ["B_30","B_38","D_114","D_116","D_117","D_120","D_126","D_63","D_64","D_66","D_68"]

class CuDFTransforms(TransformerMixin):

    def __init__(self, cat_features):
        self.cat_features = cat_features

    def fit(self, df):
        return self

    def transform(self, df):
        cat_features = self.cat_features
        all_cols = [c for c in list(df.columns) if c not in ['customer_ID','S_2']]
        num_features = [col for col in all_cols if col not in cat_features]

        test_num_agg = df.groupby("customer_ID")[num_features].agg(['mean', 'std', 'min', 'max', 'last'])
        test_num_agg.columns = ['_'.join(x) for x in test_num_agg.columns]

        test_cat_agg = df.groupby("customer_ID")[cat_features].agg(['count', 'last', 'nunique'])
        test_cat_agg.columns = ['_'.join(x) for x in test_cat_agg.columns]

        df = cudf.concat([test_num_agg, test_cat_agg], axis=1)
        del test_num_agg, test_cat_agg
        logger.info(f"shape after feature engineering: {df.shape}" )
        
        return df

class AmexPreProcessPipeline(Pipeline):

    def __init__(self):
        self.cat_features = cat_features
        steps = [
            ('cudf_transforms', CuDFTransforms(cat_features)),
            ('passthrough', None)
        ]
        super().__init__(steps)
