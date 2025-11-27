import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from numpy import pi


class HydroEstimator(BaseEstimator, RegressorMixin):

    def __init__(self,
                 w_10=0, w_B0=0, w_C1=0, w_C2=0, w_D1=0, w_D2=0, w_E1=0, w_E2=0, w_F1=0, w_F2=0, w_F3=0, w_G0=0, w_H0=0, w_I1=0, w_I2=0, w_I3=0, w_J1=0, w_J2=0, w_K1=0, w_K2=0, w_L0=0,
                 l_10=0, l_B0=0, l_C1=0, l_C2=0, l_D1=0, l_D2=0, l_E1=0, l_E2=0, l_F1=0, l_F2=0, l_F3=0, l_G0=0, l_H0=0, l_I1=0, l_I2=0, l_I3=0, l_J1=0, l_J2=0, l_K1=0, l_K2=0, l_L0=0,
                 test_size=365, model=None):
        self.regions = ["10", "B0",
                        "C1", "C2",
                        "D1", "D2",
                        "E2", "E2",
                        "F1", "F2", "F3",
                        "G0", "H0",
                        "I1", "I2", "I3",
                        "J1", "J2",
                        "K1", "K2",
                        "L0"]
        self.w_10 = w_10
        self.w_B0 = w_B0
        self.w_C1 = w_C1
        self.w_C2 = w_C2
        self.w_D1 = w_D1
        self.w_D2 = w_D2
        self.w_E1 = w_E1
        self.w_E2 = w_E2
        self.w_F1 = w_F1
        self.w_F2 = w_F2
        self.w_F3 = w_F3
        self.w_G0 = w_G0
        self.w_H0 = w_H0
        self.w_I1 = w_I1
        self.w_I2 = w_I2
        self.w_I2 = w_I2
        self.w_I3 = w_I3
        self.w_J1 = w_J1
        self.w_J2 = w_J2
        self.w_K1 = w_K1
        self.w_K2 = w_K2
        self.w_L0 = w_L0

        self.l_10 = l_10
        self.l_B0 = l_B0
        self.l_C1 = l_C1
        self.l_C2 = l_C2
        self.l_D1 = l_D1
        self.l_D2 = l_D2
        self.l_E1 = l_E1
        self.l_E2 = l_E2
        self.l_F1 = l_F1
        self.l_F2 = l_F2
        self.l_F3 = l_F3
        self.l_G0 = l_G0
        self.l_H0 = l_H0
        self.l_I1 = l_I1
        self.l_I2 = l_I2
        self.l_I2 = l_I2
        self.l_I3 = l_I3
        self.l_J1 = l_J1
        self.l_J2 = l_J2
        self.l_K1 = l_K1
        self.l_K2 = l_K2
        self.l_L0 = l_L0

        self.test_size = test_size
        self.model = model if model is not None else LinearRegression()
        self.scaler = None
        self.T = 365
        self.feature_columns_ = None

    def prepare_features(self, X):
        df = X.copy()

        df['cos'] = np.cos(2 * pi * df.index.dayofyear / self.T)
        df['sin'] = np.sin(2 * pi * df.index.dayofyear / self.T)

        for region in self.regions:
            df[f"{region}_tp_acc"] = df[f'FR{region}_TP'].rolling(
                window=self.__getattribute__(f'w_{region}')).sum().fillna(0).shift(self.__getattribute__(f'l_{region}'))

        return df

    def get_columns_to_normalize(self, df):
        all_columns = df.columns.tolist()
        exclude_columns = ['cos', 'sin']
        return [col for col in all_columns if col not in exclude_columns]

    def fit(self, X, y=None):
        X_prepared = self.prepare_features(X)
        self.feature_columns_ = X_prepared.columns.tolist()

        if self.test_size > 0 and self.test_size < len(X_prepared):
            X_train, X_test, y_train, y_test = train_test_split(
                X_prepared, y, test_size=self.test_size, shuffle=False
            )
        else:
            X_train, y_train = X_prepared, y

        columns_to_normalize = self.get_columns_to_normalize(X_train)

        self.scaler = StandardScaler()

        if columns_to_normalize:
            X_train_normalized = X_train.copy()
            X_train_normalized[columns_to_normalize] = self.scaler.fit_transform(
                X_train[columns_to_normalize]
            )
        else:
            X_train_normalized = X_train

        if X_train_normalized.isna().any().any():
            X_train_normalized = X_train_normalized.fillna(0)

        self.model.fit(X_train_normalized, y_train)

        return self

    def predict(self, X):

        X_prepared = self.prepare_features(X)

        if self.scaler is not None:
            columns_to_normalize = self.get_columns_to_normalize(X_prepared)

            if columns_to_normalize:
                X_normalized = X_prepared.copy()
                X_normalized[columns_to_normalize] = self.scaler.transform(
                    X_prepared[columns_to_normalize]
                )
            else:
                X_normalized = X_prepared
        else:
            X_normalized = X_prepared

        if X_normalized.isna().any().any():
            X_normalized = X_normalized.fillna(0)

        return self.model.predict(X_normalized)

    def get_feature_names(self):
        """Retourne les noms des features utilisées"""
        return self.feature_columns_ if self.feature_columns_ is not None else []
