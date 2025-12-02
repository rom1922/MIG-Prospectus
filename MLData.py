from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from numpy import pi

if __name__ == '__main__':

    data_directory = "data"
    correlation_file = "correlation_features_hyperparameters.csv"

    cf = pd.read_csv(f"{data_directory}/CF_1d.csv", index_col="Date",
                     parse_dates=["Date"])
    ta = pd.read_csv(f"{data_directory}/TA_1d.csv", index_col="Date",
                     parse_dates=["Date"])
    tp = pd.read_csv(f"{data_directory}/TP_1d.csv", index_col="Date",
                     parse_dates=["Date"])

    cf = cf[["FR"]]
    ta = ta.loc[:, ta.columns.str.startswith("FR")].add_suffix("_TA")
    ta.to_csv(f"{data_directory}/TA_FR.csv")
    tp = tp.loc[:, tp.columns.str.startswith("FR")].add_suffix("_TP")
    tp.to_csv(f"{data_directory}/TP_FR.csv")

    X = pd.concat([ta.mean(axis=1).rename("TA_mean"), tp,
                   cf["FR"].rename("CF")], axis=1)
    y = X[["CF"]]
    X = X.drop(columns=["CF"])
    X.index = X.index.astype("datetime64[s]")
    T = 365

    regions = ["FR10", "FRB0",
               "FRC1", "FRC2",
               "FRD1", "FRD2",
               "FRE2", "FRE2",
               "FRF1", "FRF2", "FRF3",
               "FRG0", "FRH0",
               "FRI1", "FRI2", "FRI3",
               "FRJ1", "FRJ2",
               "FRK1", "FRK2",
               "FRL0"]

    correlation_df = pd.read_csv(
        correlation_file, sep=',', index_col='Region')
    scaler = StandardScaler()

    X['cos'] = np.cos(2 * pi * X.index.dayofyear / T)
    X['sin'] = np.sin(2 * pi * X.index.dayofyear / T)

    for region in regions:
        X[f"{region}_TP_acc"] = X[f'{region}_TP'].rolling(
            window=correlation_df.at[region, 'Optimal Window']).sum().fillna(0)

    df_acc_TP_all_regions = X[[
        f"{region}_TP_acc" for region in regions]]
    df_acc_TP_not_all_regions = X[[
        f"{region}_TP_acc" for region in regions if correlation_df.at[region, "Max Correlation"] >= 0.5]]
    df_TA_mean_not_all_regions = ta[[
        f"{region}_TA" for region in regions if correlation_df.at[region, "Max Correlation"] >= 0.5]].mean(axis=1).rename("TA_mean")
    df_TA_mean_not_all_regions.to_csv(f"{data_directory}/TA_MEAN_NOTALL.csv")
    df_acc_TP_all_regions.to_csv(
        f'{data_directory}/TP_ACC_ALL_W_OPTI.csv')
    df_acc_TP_not_all_regions.to_csv(
        f'{data_directory}/TP_ACC_NOTALL_W_OPTI.csv')
    X[["TA_mean"]].to_csv(f'{data_directory}/TA_MEAN.csv')
    X[["cos", "sin"]].to_csv(f'{data_directory}/COS_SIN.csv')
