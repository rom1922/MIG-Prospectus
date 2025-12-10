---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  name: python3
  display_name: Python 3 (ipykernel)
  language: python
---

```{code-cell} ipython3
from sklearn.metrics import r2_score, mean_squared_error

def display_result(y_true, y_pred):
    dmap = {
        12: 'DJF', 1: 'DJF', 2: 'DJF',
        3: 'MAM', 4: 'MAM', 5: 'MAM',
        6: 'JJA', 7: 'JJA', 8: 'JJA',
        9: 'SON', 10: 'SON', 11: 'SON'
    }
    cmap = {"DJF": "tab:blue", "MAM": "tab:green",
            "JJA": "tab:red", "SON": "tab:orange"}
    seasons = y_true.index.month.map(dmap)
    colors = seasons.map(cmap)

    fig = plt.figure(figsize=(16, 4), constrained_layout=True)
    gs = fig.add_gridspec(ncols=2, nrows=1, width_ratios=[2, 1])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # Plot 1
    ax1.set_title("Capacity factor predictions")
    ax1.plot(y_true.index, y_true, color="tab:blue", label="Actual")
    ax1.plot(y_true.index, y_pred, color="tab:red", label="Predicted")
    ax1.set_xlim(y_true.index[0], y_true.index[-1])
    ax1.legend(loc="lower right", title="Capacity Factor")
    ax1.grid()
    # Plot 2 : 
    ax2.set_title("Actual vs Predicted")
    ax2.set_xlabel("Actual")
    ax2.set_ylabel("Predicted")
    ax2.scatter(y_true, y_pred, c=colors, s=10)

    # Diagonale
    left, right = ax2.get_xlim()
    bottom, top = ax2.get_ylim()
    lb = min(left, bottom) - 0.01
    ub = max(right, top) + 0.01
    ax2.set_xlim(lb, ub)
    ax2.set_ylim(lb, ub)
    ax2.axline((lb, lb), (ub, ub), color="tab:red")

    # Légende
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', label=s, 
                   markerfacecolor=cmap[s], markersize=6)
        for s in ["DJF", "MAM", "JJA", "SON"]
    ]
    ax2.legend(handles=handles, title="Season")
    plt.grid()
    plt.show()
```

```{code-cell} ipython3
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

#données 

cf = pd.read_csv("data/CF_1d.csv", index_col="Date", parse_dates=True)
tp = pd.read_csv("data/TP_1d.csv", index_col="Date", parse_dates=True)
ta = pd.read_csv("data/TA_1d.csv", index_col = "Date", parse_dates = True)

cf_fr = cf.FR
tp_fr = tp[tp.columns[tp.columns.str.startswith("FR")]]
ta_fr = ta[ta.columns[ta.columns.str.startswith("FR")]]

cf_fr = cf_fr.loc[cf_fr.index < "2023"]
tp_fr = tp_fr.loc[tp_fr.index < "2023"]
ta_fr = ta_fr.loc[ta_fr.index < "2023"]
cf_mean = cf_fr.rename("CF")

#normalisation
scaler = StandardScaler()
tp_norm = pd.DataFrame(scaler.fit_transform(tp_fr), 
                       index=tp_fr.index, 
                       columns=tp_fr.columns)

ta_norm = pd.DataFrame(scaler.fit_transform(ta_fr), index = ta_fr.index, columns = ta_fr.columns)
max_window = 180

tp_norm = tp_norm.iloc[max_window:,:]
cf = cf_mean.iloc[max_window:]

def window_optimization_res(max_window):
    
    windows = range(1, max_window + 1)
    results_matrix = pd.DataFrame(index=tp_norm.columns, columns=windows)

    for w in windows:
        tp_rolled = tp_fr.shift(1).rolling(window=w).sum()
        max_corrs_for_window = pd.Series([-1.0]*len(tp_norm.columns), index=tp_norm.columns)
        
        current_corrs = tp_rolled.corrwith(cf)
        max_corrs_for_window = np.maximum(max_corrs_for_window, current_corrs)
        
        results_matrix[w] = max_corrs_for_window

    results_matrix = results_matrix.astype(float)

    best_windows = results_matrix.idxmax(axis=1)
    best_values = results_matrix.max(axis=1)
    
    optimal_df = pd.DataFrame({
        "Optimal_Window": best_windows,
        "Max_Correlation": best_values
    })
    
    return optimal_df
res_tp = window_optimization_res(max_window)
res_tp
```

```{code-cell} ipython3
#données 
cf = pd.read_csv("data/CF_1d.csv", index_col="Date", parse_dates=True)
tp = pd.read_csv("data/TP_1d.csv", index_col="Date", parse_dates=True)
ta = pd.read_csv("data/TA_1d.csv", index_col="Date", parse_dates=True)

cf_fr = cf.FR
tp_fr = tp[tp.columns[tp.columns.str.startswith("FR")]].copy()
ta_fr = ta[ta.columns[ta.columns.str.startswith("FR")]].mean(axis=1).to_frame(name = "FR")


for name, column in tp_fr.items():
    if res_tp.loc[name, "Max_Correlation"] < 0.5:
        tp_fr.drop(name, axis = 1)
    else :
        tp_fr[name] = tp_fr[name].rolling(window = res_tp.loc[name, "Optimal_Window"], min_periods = 1).sum()

ta_fr["FR_rolling"] = ta_fr.shift(1).rolling(window = 14, min_periods = 1).mean()
ta_fr["FR_lag_1"] = ta_fr[["FR"]].shift(1)
ta_fr["FR_lag_3"] = ta_fr[["FR"]].shift(3)
ta_fr["FR_lag_5"] = ta_fr[["FR"]].shift(5)
ta_fr["FR_lag_7"] = ta_fr[["FR"]].shift(7)
ta_fr["FR_lag_14"] = ta_fr[["FR"]].shift(14)
ta_fr["FR_lag_30"] = ta_fr[["FR"]].shift(30)


# Features & Target

lag = 30
X = pd.merge(ta_fr, tp_fr, left_index = True, right_index = True, suffixes = ("_TA", "_TP")).iloc[lag:]
y = cf_fr.iloc[lag:]
```

```{code-cell} ipython3
Xt_train, Xt_test, yt_train, yt_test = train_test_split(X, y, test_size=365, shuffle=False)

from sklearn.decomposition import PCA

# Normalisation
scaler = StandardScaler().set_output(transform="pandas")
Xt_train_std = scaler.fit_transform(Xt_train)
Xt_test_std = scaler.transform(Xt_test)

yt_train = yt_train.squeeze()
yt_test = yt_test.squeeze()

for df in [Xt_train_std, Xt_test_std]:
    df["cos"] = np.cos(df.index.dayofyear * 2 * np.pi /365)
    df["sin"] = np.sin(df.index.dayofyear * 2 * np.pi / 365)

# 2. PCA
pca = PCA(n_components=10)
X_train_pca = pca.fit_transform(Xt_train)
X_test_pca = pca.transform(Xt_test)

Xt = pd.concat([pd.DataFrame(X_train_pca), pd.DataFrame(X_test_pca)])

from sklearn.ensemble import GradientBoostingRegressor

Xt_train_pca, Xt_test_pca, y_train, y_test = train_test_split(Xt, y, test_size=365, shuffle=False)
```

```{code-cell} ipython3
# Modèle Gradient Boosting
gbr = GradientBoostingRegressor(
    n_estimators=100,  
    learning_rate=0.1, 
    max_depth=3,
    min_samples_leaf = 1,
    subsample = 1.0,
    random_state = 42,
)


#Entraînement
gbr.fit(X_train_pca, y_train.values.ravel())
  
# Prédictions
y_pred = gbr.predict(X_test_pca)-0.052
y_pred = pd.Series(y_pred, index=y_test.index)

# Métriques

r2 = r2_score(y_test,y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
nrmse = rmse / np.mean(y_test)
mae = mean_absolute_error(y_test, y_pred)
nmae = mae / np.mean(y_test)
r = np.corrcoef(y_test, y_pred)[0,1]

print(f"R2: {r2:.06f}")
print(f"MSE: {mse:.06f}")
print(f"RMSE: {rmse:.06f}")
print(f"NRMSE: {nrmse:06f}")
print(f"MAE: {mae:0.6f}")
print(f"NMAE: {nmae:06f}")
print(f"r: {r:0.6f}")
print()

display_result(y_test, y_pred)
```

```{code-cell} ipython3
from sklearn.model_selection import TimeSeriesSplit

def time_split_cv(X, y, n_splits=8):

    tscv = TimeSeriesSplit(
    n_splits=n_splits)
    splits=[]
    for train_idx, test_idx in tscv.split(X):
        splits.append((train_idx, test_idx))
        
    return splits

liste_biais = []
for i, (train_idx, test_idx) in enumerate(time_split_cv(Xt, y, n_splits=8)):
    # Découpage des données
    biais=0
    X_train, X_test = Xt.iloc[train_idx], Xt.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Entraînement
    gbr.fit(X_train, y_train.values.ravel())

    # Prédictions
    y_pred = gbr.predict(X_test)
    y_pred = pd.Series(y_pred, index=y_test.index)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"Split {i+1}: Train={len(train_idx)}, Test={len(test_idx)}")
    
    # Métriques
    
    
    print(f"R2: {r2:.06f}")
    print(f"MSE: {mse:.06f}")
    print(f"MAE: {mae:.06f}")
    print()
    for i in range(len(y_pred)):
        biais+=(y_test[i]-y_pred[i])
    biais=biais/len(y_test)
    liste_biais.append(biais)
    # Affichage personnalisé 
    display_result(y_test, y_pred)
print(f"biais : {np.array(liste_biais.pop()).mean()}")
```

```{code-cell} ipython3

```

## Validation sur 2022

```{code-cell} ipython3
X_2022 = X.loc[X.index < "2023"]
y_2022 = y.loc[y.index < "2023"]

Xt_train, Xt_test, yt_train, yt_test = train_test_split(X_2022, y_2022, test_size=365, shuffle=False)

from sklearn.decomposition import PCA

# Normalisation
scaler = StandardScaler().set_output(transform="pandas")
Xt_train_std = scaler.fit_transform(Xt_train)
Xt_test_std = scaler.transform(Xt_test)

yt_train = yt_train.squeeze()
yt_test = yt_test.squeeze()

for df in [Xt_train_std, Xt_test_std]:
    df["cos"] = np.cos(df.index.dayofyear * 2 * np.pi /365)
    df["sin"] = np.sin(df.index.dayofyear * 2 * np.pi / 365)

# 2. PCA
pca = PCA(n_components=10)
X_train_pca = pca.fit_transform(Xt_train)
X_test_pca = pca.transform(Xt_test)

X_2022 = pd.concat([pd.DataFrame(X_train_pca), pd.DataFrame(X_test_pca)])

from sklearn.ensemble import GradientBoostingRegressor

Xt_train_pca, Xt_test_pca, y_train, y_test = train_test_split(X_2022, y_2022, test_size=365, shuffle=False)




# Modèle Gradient Boosting
gbr = GradientBoostingRegressor(
    n_estimators=100,  
    learning_rate=0.1, 
    max_depth=3,
    min_samples_leaf = 1,
    subsample = 1.0,
    random_state = 42,
)


#Entraînement
gbr.fit(X_train_pca, y_train.values.ravel())
  
# Prédictions
y_pred = gbr.predict(X_test_pca) - 0.071
y_pred = pd.Series(y_pred, index=y_test.index)

# Métriques

r2 = r2_score(y_test,y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
nrmse = rmse / np.mean(y_test)
mae = mean_absolute_error(y_test, y_pred)
nmae = mae / np.mean(y_test)
r = np.corrcoef(y_test, y_pred)[0,1]

print(f"R2: {r2:.06f}")
print(f"MSE: {mse:.06f}")
print(f"RMSE: {rmse:.06f}")
print(f"NRMSE: {nrmse:06f}")
print(f"MAE: {mae:0.6f}")
print(f"NMAE: {nmae:06f}")
print(f"r: {r:0.6f}")
print()

display_result(y_test, y_pred)
```

```{code-cell} ipython3
from sklearn.model_selection import TimeSeriesSplit

def time_split_cv(X, y, n_splits=8):

    tscv = TimeSeriesSplit(
    n_splits=n_splits)
    splits=[]
    for train_idx, test_idx in tscv.split(X):
        splits.append((train_idx, test_idx))
        
    return splits

liste_biais = []
for i, (train_idx, test_idx) in enumerate(time_split_cv(X_2022, y_2022, n_splits=7)):
    # Découpage des données
    biais=0
    X_train, X_test = X_2022.iloc[train_idx], X_2022.iloc[test_idx]
    y_train, y_test = y_2022.iloc[train_idx], y_2022.iloc[test_idx]
    
    # Entraînement
    gbr.fit(X_train, y_train.values.ravel())

    # Prédictions
    y_pred = gbr.predict(X_test)
    y_pred = pd.Series(y_pred, index=y_test.index)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"Split {i+1}: Train={len(train_idx)}, Test={len(test_idx)}")
    
    # Métriques
    
    
    print(f"R2: {r2:.06f}")
    print(f"MSE: {mse:.06f}")
    print(f"MAE: {mae:.06f}")
    print()
    for i in range(len(y_pred)):
        biais+=(y_test[i]-y_pred[i])
    biais=biais/len(y_test)
    liste_biais.append(biais)
    # Affichage personnalisé 
    display_result(y_test, y_pred)
print(f"biais : {np.array(liste_biais.pop()).mean()}")
```

## Random Forest

```{code-cell} ipython3
#Random Forest

from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA


Xt_train, Xt_test, yt_train, yt_test = train_test_split(X, y, test_size=365, shuffle=False)

# Normalisation
scaler = StandardScaler().set_output(transform="pandas")
Xt_train_std = scaler.fit_transform(Xt_train)
Xt_test_std = scaler.transform(Xt_test)

yt_train = yt_train.squeeze()
yt_test = yt_test.squeeze()

for df in [Xt_train_std, Xt_test_std]:
    df["cos"] = np.cos(df.index.dayofyear * 2 * np.pi /365)
    df["sin"] = np.sin(df.index.dayofyear * 2 * np.pi / 365)

# 2. PCA
pca = PCA(n_components=10)
X_train_pca = pca.fit_transform(Xt_train)
X_test_pca = pca.transform(Xt_test)

Xt = pd.concat([pd.DataFrame(X_train_pca), pd.DataFrame(X_test_pca)])


X_train_pca, X_test_pca, y_train, y_test = train_test_split(X, y, test_size=365, shuffle=False)



# Modèle Random Forest
rf = RandomForestRegressor(
    n_estimators=1000, 
    max_depth=50,
    min_samples_leaf = 1,
    random_state = 42
)



#Entraînement
rf.fit(X_train_pca, y_train.values.ravel())



# Prédictions
y_pred = rf.predict(X_test_pca) - 0.051
y_pred = pd.Series(y_pred, index=y_test.index)

# Métriques

r2 = r2_score(y_test,y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
nrmse = rmse / np.mean(y_test)
mae = mean_absolute_error(y_test, y_pred)
nmae = mae / np.mean(y_test)
r = np.corrcoef(y_test, y_pred)[0,1]

print(f"R2: {r2:.06f}")
print(f"MSE: {mse:.06f}")
print(f"RMSE: {rmse:.06f}")
print(f"NRMSE: {nrmse:06f}")
print(f"MAE: {mae:0.6f}")
print(f"NMAE: {nmae:06f}")
print(f"r: {r:0.6f}")
print()

display_result(y_test, y_pred)
```

```{code-cell} ipython3
from sklearn.model_selection import TimeSeriesSplit

def time_split_cv(X, y, n_splits=8):

    tscv = TimeSeriesSplit(
    n_splits=n_splits)
    splits=[]
    for train_idx, test_idx in tscv.split(X):
        splits.append((train_idx, test_idx))
        
    return splits

liste_biais = []
for i, (train_idx, test_idx) in enumerate(time_split_cv(Xt, y, n_splits=8)):
    # Découpage des données
    biais=0
    X_train, X_test = Xt.iloc[train_idx], Xt.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Entraînement
    rf.fit(X_train, y_train.values.ravel())

    # Prédictions
    y_pred = rf.predict(X_test)
    y_pred = pd.Series(y_pred, index=y_test.index)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"Split {i+1}: Train={len(train_idx)}, Test={len(test_idx)}")
    
    # Métriques
    
    
    print(f"R2: {r2:.06f}")
    print(f"MSE: {mse:.06f}")
    print(f"MAE: {mae:.06f}")
    print()
    for i in range(len(y_pred)):
        biais+=(y_test[i]-y_pred[i])
    biais=biais/len(y_test)
    liste_biais.append(biais)
    # Affichage personnalisé 
    display_result(y_test, y_pred)
print(f"biais : {np.array(liste_biais.pop()).mean()}")
```

## Validation sur 2022

```{code-cell} ipython3
#Random Forest

from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA

X_2022 = X.loc[X.index < "2023"]
y_2022 = y.loc[y.index < "2023"]

Xt_train, Xt_test, yt_train, yt_test = train_test_split(X_2022, y_2022, test_size=365, shuffle=False)

# Normalisation
scaler = StandardScaler().set_output(transform="pandas")
Xt_train_std = scaler.fit_transform(Xt_train)
Xt_test_std = scaler.transform(Xt_test)

yt_train = yt_train.squeeze()
yt_test = yt_test.squeeze()

for df in [Xt_train_std, Xt_test_std]:
    df["cos"] = np.cos(df.index.dayofyear * 2 * np.pi /365)
    df["sin"] = np.sin(df.index.dayofyear * 2 * np.pi / 365)

# 2. PCA
pca = PCA(n_components=10)
X_train_pca = pca.fit_transform(Xt_train)
X_test_pca = pca.transform(Xt_test)

Xt = pd.concat([pd.DataFrame(X_train_pca), pd.DataFrame(X_test_pca)])


X_train_pca, X_test_pca, y_train, y_test = train_test_split(Xt, y_2022, test_size=365, shuffle=False)



# Modèle Random Forest
rf = RandomForestRegressor(
    n_estimators=1000, 
    max_depth=50,
    min_samples_leaf = 1,
)



#Entraînement
rf.fit(X_train_pca, y_train.values.ravel())



# Prédictions
y_pred = rf.predict(X_test_pca) - 0.072
y_pred = pd.Series(y_pred, index=y_test.index)

# Métriques

r2 = r2_score(y_test,y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
nrmse = rmse / np.mean(y_test)
mae = mean_absolute_error(y_test, y_pred)
nmae = mae / np.mean(y_test)
r = np.corrcoef(y_test, y_pred)[0,1]

print(f"R2: {r2:.06f}")
print(f"MSE: {mse:.06f}")
print(f"RMSE: {rmse:.06f}")
print(f"NRMSE: {nrmse:06f}")
print(f"MAE: {mae:0.6f}")
print(f"NMAE: {nmae:06f}")
print(f"r: {r:0.6f}")
print()

display_result(y_test, y_pred)
```

```{code-cell} ipython3
from sklearn.model_selection import TimeSeriesSplit

def time_split_cv(X, y, n_splits):

    tscv = TimeSeriesSplit(
    n_splits=n_splits)
    splits=[]
    for train_idx, test_idx in tscv.split(X):
        splits.append((train_idx, test_idx))
        
    return splits

liste_biais = []
for i, (train_idx, test_idx) in enumerate(time_split_cv(Xt, y, n_splits=7)):
    # Découpage des données
    biais=0
    X_train, X_test = Xt.iloc[train_idx], Xt.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Entraînement
    rf.fit(X_train, y_train.values.ravel())

    # Prédictions
    y_pred = rf.predict(X_test)
    y_pred = pd.Series(y_pred, index=y_test.index)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"Split {i+1}: Train={len(train_idx)}, Test={len(test_idx)}")
    
    # Métriques
    
    
    print(f"R2: {r2:.06f}")
    print(f"MSE: {mse:.06f}")
    print(f"MAE: {mae:.06f}")
    print()
    for i in range(len(y_pred)):
        biais+=(y_test[i]-y_pred[i])
    biais=biais/len(y_test)
    liste_biais.append(biais)
    # Affichage personnalisé 
    display_result(y_test, y_pred)
print(f"biais : {np.array(liste_biais.pop()).mean()}")
```

```{code-cell} ipython3

```
