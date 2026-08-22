import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class PercentileWinsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names, cuts_by_variable=None, default_limits=(0.01, 0.99)):
        self.feature_names = feature_names
        self.cuts_by_variable = cuts_by_variable
        self.default_limits = default_limits

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X, columns=self.feature_names).copy()

        self.lower_bounds_ = {}
        self.upper_bounds_ = {}

        for col in self.feature_names:
            lower_q, upper_q = self.default_limits

            if self.cuts_by_variable is not None and col in self.cuts_by_variable:
                lower_q, upper_q = self.cuts_by_variable[col]

            self.lower_bounds_[col] = X_df[col].quantile(lower_q)
            self.upper_bounds_[col] = X_df[col].quantile(upper_q)

        return self

    def transform(self, X):
        X_df = pd.DataFrame(X, columns=self.feature_names).copy()

        for col in self.feature_names:
            X_df[col] = X_df[col].clip(
                lower=self.lower_bounds_[col],
                upper=self.upper_bounds_[col]
            )

        return X_df