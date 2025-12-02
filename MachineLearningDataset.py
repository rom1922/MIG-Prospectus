from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from numpy import pi


class MLHydroDataset():
    def __init__(self, data_directory="data", correlation_file="correlation_features_hyperparameters.csv"):
        self.data_directory = data_directory
        cf = pd.read_csv(f"{data_directory}/CF_1d.csv", index_col="Date",
                         parse_dates=["Date"])
        ta = pd.read_csv(f"{data_directory}/TA_1d.csv", index_col="Date",
                         parse_dates=["Date"])
        tp = pd.read_csv(f"{data_directory}/TP_1d.csv", index_col="Date",
                         parse_dates=["Date"])

        cf = cf[["FR"]]
        ta = ta.loc[:, ta.columns.str.startswith("FR")].add_suffix("_TA")
        tp = tp.loc[:, tp.columns.str.startswith("FR")].add_suffix("_TP")

        ta = ta.mean(axis=1).rename("TA_mean")
        X = pd.concat([ta, tp,
                      cf["FR"].rename("CF")], axis=1)
        self.y = X[["CF"]]
        X = X.drop(columns=["CF"])
        X.index = X.index.astype("datetime64[s]")
        self.X = X
        self.T = 365

        self.regions = ["FR10", "FRB0",
                        "FRC1", "FRC2",
                        "FRD1", "FRD2",
                        "FRE2", "FRE2",
                        "FRF1", "FRF2", "FRF3",
                        "FRG0", "FRH0",
                        "FRI1", "FRI2", "FRI3",
                        "FRJ1", "FRJ2",
                        "FRK1", "FRK2",
                        "FRL0"]

        self.correlation_df = pd.read_csv(
            correlation_file, sep=',', index_col='Region')
        self.scaler = StandardScaler()

    def prepare_features(self):
        self.X['cos'] = np.cos(2 * pi * self.X.index.dayofyear / self.T)
        self.X['sin'] = np.sin(2 * pi * self.X.index.dayofyear / self.T)

        for region in self.regions:
            self.X[f"{region}_TP_acc"] = self.X[f'{region}_TP'].rolling(
                window=self.correlation_df.at[region, 'Optimal Window']).sum().fillna(0)

    def save_features(self):
        df_acc_TP_all_regions = self.X[[
            f"{region}_TP_acc" for region in self.regions]]
        df_acc_TP_not_all_regions = self.X[[
            f"{region}_TP_acc" for region in self.regions if self.correlation_df.at[region, "Max Correlation"] >= 0.5]]
        df_acc_TP_all_regions.to_csv(
            f'{self.data_directory}/TP_ACC_ALL_W_OPTI.csv')
        df_acc_TP_not_all_regions.to_csv(
            f'{self.data_directory}/TP_ACC_NOTALL_W_OPTI.csv')
        self.X[["TA_mean"]].to_csv(f'{self.data_directory}/TA_MEAN.csv')
        self.X[["cos", "sin"]].to_csv(f'{self.data_directory}/COS_SIN.csv')

    def get_columns_to_normalize(self):
        all_columns = self.X.columns.tolist()
        exclude_columns = ['cos', 'sin']
        return [col for col in all_columns if col not in exclude_columns]

    def fit_transform(self, X2):
        columns_to_normalize = self.get_columns_to_normalize()
        X_std = X2.copy()

        X_std[columns_to_normalize] = self.scaler.fit_transform(
            X2[columns_to_normalize]
        )

        return X_std

    def transform(self, X2):
        columns_to_normalize = self.get_columns_to_normalize()
        X_std = X2.copy()
        X_std[columns_to_normalize] = self.scaler.transform(
            X2[columns_to_normalize]
        )
        return X_std
