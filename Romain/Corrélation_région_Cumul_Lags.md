# Corrélation cumul et lag des régions par rapport au CF


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

cf_mean = cf_fr.rename("CF")

#normalisation
scaler = StandardScaler()
tp_norm = pd.DataFrame(scaler.fit_transform(tp_fr), 
                       index=tp_fr.index, 
                       columns=tp_fr.columns)



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


```python
#Fonction interactive (analysis corrélation × lag × cumul)

def cross_corr_heatmap(max_lag, window, threshold=0.15):
    lags = range(0, max_lag + 1)
    corr_matrix = pd.DataFrame(index=tp_norm.columns, columns=lags)


    for region in tp_norm.columns:
        tp_wind = tp_norm[region].rolling(window=window).sum()

        for lag in lags:
            tp_lagged = tp_wind.shift(lag)
            df_tmp = pd.concat([tp_lagged, cf_mean], axis=1).dropna()
            corr_matrix.loc[region, lag] = df_tmp.iloc[:, 0].corr(df_tmp["CF"])

    corr_matrix = corr_matrix.astype(float)

    cmap = make_continuous_threshold_cmap(threshold)

    plt.figure(figsize=(14, 10))
    sns.heatmap(
        corr_matrix,
        cmap=cmap,
        vmin=-1, vmax=1,
        center=threshold,
        linewidths=.4
    )

    plt.title(
        f"Corrélation TP cumulées (fenêtre={window} jours) × CF\n"
        f"threshold={threshold:.2f})"
    )
    plt.xlabel("Lag (jours)")
    plt.ylabel("Régions")
    plt.show()

    # régions parasites = aucune corrélation au-dessus du threshold
    parasites = corr_matrix.abs().max(axis=1)
    parasites = parasites[parasites < threshold].index.tolist()

    print(f"\nRégions parasites :{parasites}")
    print()

```


```python
#interface intéractive
interact(
    cross_corr_heatmap,
    max_lag=IntSlider(value=10, min=1, max=90, step=1,
                      description="Lag max", continuous_update=False),
    window=IntSlider(value=4, min=1, max=40, step=1,
                     description="Cumul (jours)", continuous_update=False)
)
```


    interactive(children=(IntSlider(value=10, continuous_update=False, description='Lag max', max=90, min=1), IntS…





    <function __main__.cross_corr_heatmap(max_lag, window, threshold=0.15)>




```python

```


```python

```
