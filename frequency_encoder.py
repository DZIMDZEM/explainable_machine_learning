from sklearn.base import BaseEstimator, TransformerMixin


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(
            self,
    ):
        self.encoders = {}
        self.columns = []

    def fit(self, X, y=None):
        self.columns = X.columns

        for col in self.columns:
            count = X[col].value_counts()
            self.encoders[col] = count / count.sum()

        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            X[col] = X[col].map(self.encoders[col]).fillna(0)

        return X

    def get_feature_names_out(self, *args, **kwargs):
        return self.columns