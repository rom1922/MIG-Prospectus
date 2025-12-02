# Corrélation cumul des régions par rapport au CF


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ipywidgets import interact, IntSlider
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import StandardScaler


sns.set_style("whitegrid")


```


```python
#chargement des données
cf = pd.read_csv("data/CF_1d.csv", index_col="Date", parse_dates=True)
tp = pd.read_csv("data/TP_1d.csv", index_col="Date", parse_dates=True)


cf_fr = cf.FR
tp_fr = tp[tp.columns[tp.columns.str.startswith("FR")]]
ta_fr = ta[ta.columns[ta.columns.str.startswith("FR")]]

cf_fr = cf_fr.loc[cf_fr.index < "2023"]
tp_fr = tp_fr.loc[tp_fr.index < "2023"]
cf_mean = cf_fr.rename("CF")

#normalisation
scaler = StandardScaler()
tp_norm = pd.DataFrame(scaler.fit_transform(tp_fr), 
                       index=tp_fr.index, 
                       columns=tp_fr.columns)

ta_norm = pd.DataFrame(scaler.fit_transform(ta_fr), index = ta_fr.index, columns = ta_fr.columns)

```


```python
def make_continuous_threshold_cmap(threshold):
    """
    Création d'une colormap continue :
    - Bleu foncé -> bleu clair -> blanc autour du threshold -> orange -> rouge
    - Beaucoup plus de nuances.
    """
    
    colors = [
        (0.0, (0, 0.15, 0.8)),    # bleu foncé
        (0.40, (0.3, 0.5, 1.0)),  # bleu clair
        (0.50, (1, 1, 1)),        # neutre autour du seuil
        (0.65, (1.0, 0.7, 0.2)),  # orange
        (1.0, (1.0, 0, 0))        # rouge
    ]
    
    cmap = LinearSegmentedColormap.from_list("smooth_blue_red", colors)
    return cmap

```


## Correlation des précipitations par régions

```python
from ipywidgets import interactive, IntSlider
import IPython.display as display

# 1. La définition de ta fonction (avec le return ajouté)
def window_optimization_heatmap(min_window=1, max_window=180, step=2, search_lag_max=10, threshold=0.15):
    
    windows = range(min_window, max_window + 1, step)
    
    # Initialisation dataframe résultat
    results_matrix = pd.DataFrame(index=tp_norm.columns, columns=windows)
    
    for w in windows:
        tp_rolled = tp_norm.rolling(window=w).sum()
        max_corrs_for_window = pd.Series([-1.0]*len(tp_norm.columns), index=tp_norm.columns)
        
        for lag in range(search_lag_max + 1):
            tp_shifted = tp_rolled.shift(lag)
            current_corrs = tp_shifted.corrwith(cf_mean)
            max_corrs_for_window = np.maximum(max_corrs_for_window, current_corrs)
        
        results_matrix[w] = max_corrs_for_window

    results_matrix = results_matrix.astype(float)

    # --- Affichage Heatmap ---
    # Je suppose que make_continuous_threshold_cmap est défini ailleurs dans ton code
    cmap = make_continuous_threshold_cmap(threshold) 
    
    plt.figure(figsize=(16, 10))
    sns.heatmap(
        results_matrix,
        cmap=cmap,
        vmin=-1, vmax=1,
        center=threshold,
        linewidths=.0,
        cbar_kws={'label': 'Max Correlation (optimized over lags)'}
    )

    plt.title(
        f"Sensibilité de la Corrélation à la taille du Cumul\n"
        f"Lag optimisé [0-{search_lag_max}] | Threshold={threshold}"
    )
    plt.xlabel("Taille de la fenêtre (Jours)")
    plt.ylabel("Régions")
    plt.show()

    # --- Identification des Optimas ---
    best_windows = results_matrix.idxmax(axis=1)
    best_values = results_matrix.max(axis=1)
    
    optimal_df = pd.DataFrame({
        "Optimal Window": best_windows,
        "Max Correlation": best_values
    }).sort_values(by="Max Correlation", ascending=False)
    
    
    print("\n--- Top Fenêtres Optimales ---")
    print(optimal_df[optimal_df["Max Correlation"] > threshold])

   
    return optimal_df


w = interactive(
    window_optimization_heatmap,
    min_window=IntSlider(value=1, min=1, max=30, step=1, description="Min Wind"),
    max_window=IntSlider(value=180, min=30, max=180, step=5, description="Max Wind"),
    step=IntSlider(value=5, min=1, max=10, step=1, description="Step"),
    search_lag_max=IntSlider(value=0, min=0, max=30, description="Lag Search"),
    threshold=(0.0, 1.0, 0.05)
)


display.display(w)
```


```python
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
```

```python
res_tp
```

```python

nom_fichier = "Resultats_Optimisation_Pluie.csv"
res_tp.query("Max_Correlation > 0.5").to_csv(nom_fichier)
```

## Correlation des températures par régions

```python
def window_and_lag_optimization(min_window=1, max_window=30, step=1, search_lag_max=180, threshold=0.25):

    windows = range(min_window, max_window + 1, step)

    results_matrix = pd.DataFrame(index=ta_norm.columns, columns=windows, dtype=float)
    lags_matrix = pd.DataFrame(index=ta_norm.columns, columns=windows, dtype=float)
    

    for w in windows:
        
        ta_rolled = ta_norm.rolling(window=w).mean()
  
        best_corr_w = pd.Series([0.0]*len(ta_norm.columns), index=ta_norm.columns)
        best_lag_w = pd.Series([0]*len(ta_norm.columns), index=ta_norm.columns)
        
        for lag in range(search_lag_max + 1):
            ta_shifted = ta_rolled.shift(lag)
            current_corrs = ta_shifted.corrwith(cf_mean)
 
            is_stronger = current_corrs.abs() > best_corr_w.abs()
            
        
            best_corr_w[is_stronger] = current_corrs[is_stronger]
            best_lag_w[is_stronger] = lag
     
        results_matrix[w] = best_corr_w
        lags_matrix[w] = best_lag_w

    plt.figure(figsize=(16, 10))
    sns.heatmap(
        results_matrix,
        cmap=cmap,
        vmin=-1, vmax=1,
        center=0,
        linewidths=.0,
        cbar_kws={'label': 'Corrélation Max (Optimisée sur Lag)'}
    )

    plt.title(
        f"Optimisation Température : Window vs Région\n"
        f"Lag optimisé ({0}-{search_lag_max}j) | Threshold={threshold}"
    )
    plt.xlabel("Taille de la fenêtre (Jours)")
    plt.ylabel("Régions")
    plt.show()

    best_windows = results_matrix.abs().idxmax(axis=1)

    best_corrs = []
    best_lags = []
    
    for region, window in best_windows.items():

        corr_val = results_matrix.loc[region, window]
        best_corrs.append(corr_val)
        lag_val = lags_matrix.loc[region, window]
        best_lags.append(int(lag_val))


    optimal_df = pd.DataFrame({
        "Optimal Window": best_windows,
        "Optimal Lag": best_lags,  
        "Max Correlation": best_corrs,
        "Magnitude": [abs(c) for c in best_corrs] # Aide pour le tri
    }, index=best_windows.index)

    return optimal_df

# Interface interactive

interact(
    window_and_lag_optimization,
    min_window=IntSlider(value=1, min=1, max=30, step=1, description="Min Wind"),
    max_window=IntSlider(value=60, min=30, max=180, step=5, description="Max Wind"),
    step=IntSlider(value=2, min=1, max=10, step=1, description="Step"),
    search_lag_max=IntSlider(value=5, min=0, max=180, description="Lag Search"),
    threshold=(0.0, 0.8, 0.05)
)
```

```python
def window_and_lag_optimization_res(min_window=1, max_window=180, step=1, search_lag_max=180, threshold=0.25):

    windows = range(min_window, max_window + 1, step)

    results_matrix = pd.DataFrame(index=ta_norm.columns, columns=windows, dtype=float)
    lags_matrix = pd.DataFrame(index=.columns, columns=windows, dtype=float)
    for lag in range(search_lag_max +1):
        lags_matrix

    for w in windows:
        
        ta_rolled = ta_norm.rolling(window=w).mean()
  
        best_corr_w = pd.Series([0.0]*len(ta_norm.columns), index=ta_norm.columns)
        best_lag_w = pd.Series([0]*len(ta_norm.columns), index=ta_norm.columns)
        
        for lag in range(search_lag_max + 1):
            ta_shifted = ta_rolled.shift(lag)
            current_corrs = ta_shifted.corrwith(cf_mean)
 
            is_stronger = current_corrs.abs() > best_corr_w.abs()
            
        
            best_corr_w[is_stronger] = current_corrs[is_stronger]
            best_lag_w[is_stronger] = lag
     
        results_matrix[w] = best_corr_w
        lags_matrix[w] = best_lag_w

    best_windows = results_matrix.abs().idxmax(axis=1)

    best_corrs = []
    best_lags = []
    
    for region, window in best_windows.items():

        corr_val = results_matrix.loc[region, window]
        best_corrs.append(corr_val)
        lag_val = lags_matrix.loc[region, window]
        best_lags.append(int(lag_val))


    optimal_df = pd.DataFrame({
        "Optimal Window": best_windows,
        "Optimal Lag": best_lags,  
        "Max Correlation":[abs(c) for c in best_corrs]
    }, index=best_windows.index)

    return optimal_df

res = window_and_lag_optimization_res()
```

```python
res_ta = res
```

## Implémentation des nouvelles donnnées

```python
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

    plt.show()
```

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

#données 

cf = pd.read_csv("data/CF_1d.csv", index_col = "Date", parse_dates = True)
ta = pd.read_csv("data/TA_1d.csv", index_col = "Date", parse_dates = True)
tp = pd.read_csv("data/TP_1d.csv", index_col = "Date", parse_dates = True)

cf_FR = cf[["FR"]]
ta_FR = ta[ta.columns[ta.columns.str.startswith("FR")]]
tp_FR = tp[tp.columns[tp.columns.str.startswith("FR")]]

for name, column in tp_FR.items():
    if res_tp.loc[name, "Max Correlation"] < 0.45:
        tp_FR.drop(name, axis = 1)
    else :
        tp_FR[name] = tp_FR[name].rolling(window = res_tp.loc[name, "Optimal Window"], min_periods = 1).sum()

for name, colums in ta_FR.items():
    ta_FR[name] = ta_FR[name].rolling(window = res_ta.loc[name, "Optimal Window"], min_periods = 1).mean().shift(res_ta.loc[name, "Optimal Lag"]).fillna(0)


# Features & Target
X = pd.merge(ta_FR, tp_FR, left_index = True, right_index = True, suffixes = ("_TA", "_TP"))
y = cf_FR


# Séparation des données d'entrainement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=365, shuffle=False)

# Normalisation
scaler = StandardScaler().set_output(transform="pandas")
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

#X_train_std["cos"] = np.cos(X_train_std.index.dayofyear * 2 * np.pi/365)
#X_train_std["sin"] = np.sin(X_train_std.index.dayofyear * 2 * np.pi/365)

#X_test_std["cos"] = np.cos(X_test_std.index.dayofyear * 2 * np.pi/365)
#X_test_std["sin"] = np.sin(X_test_std.index.dayofyear * 2 * np.pi/365)

results2 = {"Actual": y_test}

from sklearn.ensemble import GradientBoostingRegressor

# Modèle Gradient Boosting
gbr = GradientBoostingRegressor(
    n_estimators=1000,   # nombre d’étapes de boosting
    learning_rate=0.1, # vitesse d'apprentissage
    max_depth=3,        # profondeur des arbres faibles
    random_state=75
)

# Entraînement
gbr.fit(X_train_std, y_train)

# Prédictions
y_pred = gbr.predict(X_test_std)
y_pred = pd.Series(y_pred, index=y_test.index)

# Ajout dans results
results2["GBR"] = y_pred

# Métriques
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"R2: {r2:.06f}")
print(f"MSE: {mse:.06f}")
print()

display_result(y_test, y_pred)
```

### Cross-validation - optimisation des paramètres

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor

cf = pd.read_csv("data/CF_1d.csv", index_col="Date", parse_dates=True)
ta = pd.read_csv("data/TA_1d.csv", index_col="Date", parse_dates=True)
tp = pd.read_csv("data/TP_1d.csv", index_col="Date", parse_dates=True)

cf_FR = cf[["FR"]]
ta_FR = ta[ta.columns[ta.columns.str.startswith("FR")]]
tp_FR = tp[tp.columns[tp.columns.str.startswith("FR")]]


for name, column in tp_FR.items():

    win_size = res_tp.loc[name, "Optimal Window"]
    tp_FR[name] = tp_FR[name].rolling(window=int(win_size), min_periods=1).sum()


ta_mean = ta_FR.mean(axis=1).rename("TA_Mean")

X = pd.merge(ta_mean, tp_FR, left_index=True, right_index=True)
y = cf_FR

# Alignement
common_index = X.index.intersection(y.index)
X = X.loc[common_index]
y = y.loc[common_index]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=365, shuffle=False)

scaler = StandardScaler().set_output(transform="pandas")
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)


for df in [X_train_std, X_test_std]:
    df["cos"] = np.cos(df.index.dayofyear * 2 * np.pi/365)
    df["sin"] = np.sin(df.index.dayofyear * 2 * np.pi/365)

#    df["Interaction_Temp_Season"] = df["TA_Mean"] * df["cos"]

#GridSearch

param_grid = {
    'n_estimators': [10000],         
    'learning_rate': [0.01, 0.05, 0.1],   
    'max_depth': [3],           
    'min_samples_leaf': [5],    
    'subsample': [0.8]              
}


# TimeSeriesSplit pour la validation croisée 
tscv = TimeSeriesSplit(n_splits=3)

gbr_search = GridSearchCV(
    estimator=GradientBoostingRegressor(random_state=42, loss='squared_error'),
    param_grid=param_grid,
    cv=tscv,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)

gbr_search.fit(X_train_std, y_train.values.ravel())

print(f"Meilleurs paramètres : {gbr_search.best_params_}")
print(f"Meilleur Score CV : {gbr_search.best_score_:.4f}")


best_model = gbr_search.best_estimator_
y_pred = best_model.predict(X_test_std)
y_pred_series = pd.Series(y_pred, index=y_test.index)

# Résultats
print(f"R2 Final (Test): {r2_score(y_test, y_pred_series):.4f}")
print(f"MSE Final (Test): {mean_squared_error(y_test, y_pred_series):.4f}")

# Visualisation
display_result(y_test, y_pred)
```

```python

```
