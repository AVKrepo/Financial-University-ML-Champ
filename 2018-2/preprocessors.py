import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder


class NanHandler(BaseEstimator, ClassifierMixin):
    """
    Preprocessor, which goal is to smarty handle columns with NaN.
    """
    def __init__(self, features='all', is_nan_column=True, substitution_mode='zero'):
        self.features = features
        self.feature_idxs = None
        self.substitution_mode = substitution_mode
        self.is_nan_column = is_nan_column
        self.substitution_values = []

    def fit(self, X, y=None):
        if self.features == 'all':
            self.feature_idxs = range(X.shape[1])
        else:
            self.feature_idxs = self.features

        for feature_idx in self.feature_idxs:
            column = np.array(X[:, feature_idx], dtype=np.object)
            mask = pd.isnull(column)
            mask_valid = (1 - mask).astype(bool)

            if self.substitution_mode == 'zero':
                value = 0
            elif self.substitution_mode == 'mean':
                # print(column[mask_valid])
                value = np.mean(column[mask_valid])
            elif self.substitution_mode == 'most_frequent':
                (values, counts) = np.unique(column[mask_valid], return_counts=True)
                index = np.argmax(counts)
                value = values[index]
            elif self.substitution_mode == 'unique':
                value = -1
            else:
                raise Exception("Unknown mode in NanHandler")

            self.substitution_values.append((feature_idx, value))
        return self

    def transform(self, X):
        X_new = X.copy()
        for feature_idx, value in self.substitution_values:
            column = np.array(X[:, feature_idx])
            mask = pd.isnull(column)

            print("NanHandler({}):: idx={}, value={}, column=".format(self.substitution_mode, feature_idx, value))
            # print(column)

            if self.is_nan_column:
                X_new = np.append(X_new, np.array([mask]).transpose(), axis=1)

            column[mask] = value

            X_new[:, feature_idx] = column
        return X_new


########################################################################################################################


class ColumnDropper(BaseEstimator, ClassifierMixin):
    """
    Preprocessor, which goal is to drop certain columns.
    """
    def __init__(self, features='all'):
        self.features = features
        self.feature_idxs = None

    def fit(self, X, y=None):
        if self.features == 'all':
            self.feature_idxs = range(X.shape[1])
        else:
            self.feature_idxs = self.features
        return self

    def transform(self, X):
        X_new = np.delete(X, self.feature_idxs, 1)
        return X_new


########################################################################################################################


class CategoricalFrequencyEncoder(BaseEstimator, ClassifierMixin):
    """
    Preprocessor, which substitute categorical features by their frequency (appending new columns).
    """
    def __init__(self, features='all', add_columns=True):
        self.features = features
        self.feature_idxs = None
        self.add_columns = add_columns
        self.encoders = []

    def fit(self, X, y=None):
        if self.features == 'all':
            self.feature_idxs = range(X.shape[1])
        else:
            self.feature_idxs = self.features

        for feature_idx in self.feature_idxs:
            print("CategoricalFrequencyEncoder:: idx={} ".format(feature_idx), end='')
            column = X[:, feature_idx]
            frequency_dict = {}
            for label in set(column):
                frequency = np.sum(column == label)
                frequency_dict[label] = frequency
            print("len={}".format(len(frequency_dict)))
            self.encoders.append((feature_idx, frequency_dict))
        return self

    def transform(self, X):
        X_new = X.copy()
        for feature_idx, frequency_dict in self.encoders:
            column = np.array(X_new[:, feature_idx])
            for label in set(column):
                location = np.argwhere(column == label)
                if label in frequency_dict:
                    frequency = frequency_dict[label]
                else:
                    frequency = 0
                column[location] = frequency
            # print(column)
            X_new = np.append(X_new, np.array([column]).transpose(), axis=1)
        return X_new


########################################################################################################################


class SubstitutiveCategoricalEncoder(BaseEstimator, ClassifierMixin):
    """
    Preprocessor, which substitute categorical features in input data by LabelEncoder for certain columns.
    """
    def __init__(self, features='all', add_initial_column=False):
        self.features = features
        self.add_initial_column = add_initial_column
        self.feature_idxs = None
        self.encoders = []

    class MyLabelEncoder(BaseEstimator):
        """
        This class is like sklearn's LabelEncoder, but it can substitute new categories (during transform) by new value.
        """
        def __init__(self):
            self.labels_to_numeric = {}

        def fit(self, X):
            for label in set(X):
                self.labels_to_numeric[label] = len(self.labels_to_numeric)
            return self

        def transform(self, X):
            X_new = X.copy()
            for label in set(X):
                location = np.argwhere(X == label)
                if label in self.labels_to_numeric:
                    X_new[location] = self.labels_to_numeric[label]
                else:
                    X_new[location] = len(self.labels_to_numeric) + 1
            return X_new

    def fit(self, X, y=None):
        if self.features == 'all':
            self.feature_idxs = range(X.shape[1])
        else:
            self.feature_idxs = self.features

        for feature_idx in self.feature_idxs:
            encoder = self.MyLabelEncoder()
            encoder.fit(X[:, feature_idx])
            self.encoders.append((feature_idx, encoder))
        return self

    def transform(self, X):
        X_new = X.copy()
        for feature_idx, encoder in self.encoders:
            if self.add_initial_column:
                print("SubstitutiveCategoricalEncoder:: idx={}, column=".format(feature_idx))
                column = np.array(X_new[:, feature_idx])
                # print(column)
                X_new = np.append(X_new, np.array([column]).transpose(), axis=1)
            X_new[:, feature_idx] = encoder.transform(X_new[:, feature_idx])
        return X_new


########################################################################################################################


# class SubstitutiveStandartScaler(BaseEstimator, ClassifierMixin):
#     """
#     Preprocessor, which applies sklearn's StandardScaler to certain columns.
#     """
#     def __init__(self, features='all'):
#         self.features = features
#         self.feature_idxs = None
#         self.scaler = StandardScaler()
#         # self.params = []  # tuple (feature_idx, mean, std)
#
#     def fit(self, X, y=None):
#         if self.features == 'all':
#             self.feature_idxs = range(X.shape[1])
#         else:
#             self.feature_idxs = self.features
#         submatrix = np.array(X[:, self.feature_idxs], dtype=np.float64)
#         submatrix[np.isnan(submatrix)] = 0
#         self.scaler.fit(submatrix)
#         # for feature_idx in self.feature_idxs:
#         #     column = np.array(X[:, feature_idx], dtype=np.float64)
#         #     column[np.isnan(column)] = 0
#         #     mean = np.mean(column)
#         #     std = np.std(column)
#         #     self.params.append((feature_idx, mean, std))
#         return self
#
#     def transform(self, X):
#         X_new = X.copy()
#         submatrix = np.array(X[:, self.feature_idxs], dtype=np.float64)
#         submatrix[np.isnan(submatrix)] = 0
#         submatrix = self.scaler.transform(submatrix)
#         X_new[:, self.feature_idxs] = submatrix
#         # for feature_idx, mean, std in self.params:
#         #     column = np.array(X[:, feature_idx], dtype=np.float64)
#         #     column[np.isnan(column)] = 0
#         #     column = (column - mean) / std
#         #     X_new[:, feature_idx] = column
#         return X_new


########################################################################################################################


class MeanTargetEncoder(BaseEstimator, ClassifierMixin):
    """
    Preprocessor, which substitute categorical values by mean-target.
    """
    def __init__(self, features='all', new_column=True):
        self.features = features
        self.new_column = new_column
        self.feature_idxs = None
        self.values = []  # list of pairs (feature_idx, {category: mean_target})

    def fit(self, X, y):
        if self.features == 'all':
            self.feature_idxs = range(X.shape[1])
        else:
            self.feature_idxs = self.features

        for feature_idx in self.feature_idxs:
            targets = {}
            targets["unknown"] = np.mean(y)
            column = X[:, feature_idx]
            for category in set(column):
                location = np.argwhere(column == category)
                mean_target = np.mean(y[location])
                targets[category] = mean_target
            self.values.append((feature_idx, targets))
        return self

    def transform(self, X):
        X_new = X.copy()
        for feature_idx, targets in self.values:
            column = np.array(X[:, feature_idx])
            for category in set(column):
                location = np.argwhere(column == category)
                if category in targets:
                    column[location] = targets[category]
                    # X_new[location, feature_idx] = targets[category]
                else:
                    column[location] = targets["unknown"]
                    # X_new[location, feature_idx] = targets["unknown"]

            if self.new_column:
                X_new = np.append(X_new, np.array([column]).transpose(), axis=1)
            else:
                X_new[:, feature_idx] = column
        return X_new


class GlobalPreprocessor(BaseEstimator, ClassifierMixin):
    """
    One Preprocessor for heterogeneous features, which sequentially applies different preprocessors.
    """
    def __init__(self):
        self.preprocessors = [
            NanHandler(features=[7, 8, 14], substitution_mode='zero'),
            NanHandler(features=[5, 6, 11, 12], substitution_mode='mean'),
            NanHandler(features=[13, 15, 16, 17], substitution_mode='unique'),
            NanHandler(features=[19, 23, 26], substitution_mode='most_frequent'),
            CategoricalFrequencyEncoder(features=[0, 3, 4, 5, 6, 13, 15, 16, 17, 18, 19, 20, 21, 26, 27]),
            SubstitutiveCategoricalEncoder(features=[0, 4, 19, 20, 21, 27]),
            SubstitutiveCategoricalEncoder(features=[13, 15, 16], add_initial_column=True),
            # SubstitutiveStandartScaler(features=[]),
            # ColumnDropper(features=[2, 14]),  # DO NOT USE!!!
            OneHotEncoder(categorical_features=[0, 13, 15, 16, 19, 27], sparse=False),
        ]

    def fit(self, X, y):
        X_new = X.copy()
        for preprocessor in self.preprocessors:
            preprocessor.fit(X_new, y)
            X_new = preprocessor.transform(X_new)
        return self

    def transform(self, X):
        X_new = X.copy()
        for preprocessor in self.preprocessors:
            X_new = preprocessor.transform(X_new)
            # print(pd.DataFrame(X_new).head())

        # if X_new.shape[1] != 58:
        #     print("ASSERTION: X_new.shape[1] = {} != 58".format(X_new.shape[1]))
        return X_new
