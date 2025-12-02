---
jupytext:
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

+++ {"id": "at5zIeuVk6FI", "editable": true, "slideshow": {"slide_type": ""}}

# Machine Learning

Ce notebook propose une première exploration des méthodes de modélisation appliquées à des données temporelles, afin d'illustrer comment le machine learning peut être utilisé pour estimer un facteur de charge à partir de séries chronologiques climatiques.

+++ {"id": "X10t3fiNYb8P"}

## Généralités

Les grandes étapes de la réalisation d'un modèle de machine learning :
1. Préparation et Exploration
  *  Nettoyage et préparation des données
  *  Exploration et analyse des données (EDA)
  *  Feature engineering
2. Modélisation
  *  Découpage du jeu de données
  *  Choix du modèle
  *  Entraînement et optimisation
3. Évaluation et Interprétation
  *  Évaluation du modèle
  *  Interprétation et validation métier

Ici nous omettons les étapes de collecte des données (étape 0) et de mise en production du modèle (étape 4).

+++ {"id": "GFRj88TEn_uK"}

**Contexte :**

Nous disposons de données climatiques régionales de température et de précipitation pour la France continentale (21 régions NUTS2) de 2015 à 2023. Pour chaque année, nous disposons du **facteur de charge national** (NUTS0) des centrales hydroélectriques au fil de l’eau.

**Objectifs :** explorer les données, construire des variables explicatives simples et tester plusieurs modèles de régression

+++ {"id": "wBQaWwFoqrl_"}

## 1. Préparation et Exploration

+++ {"id": "XtGpEyd1V55f"}

### Nettoyage et préparation des données

Télécharger les données nécessaires pour l'analyse exploratoire. Les données sont décompressées dans le répertoire `data`:
- `CF_1d.csv` : facteur de charge des centrales hydroélectriques au fil de l'eau au pas journalier de chaque pays européen,
- `TA_1d.csv` : température moyenne de l'air au pas journalier de chaque région administrative de chaque pays européen,
- `TP_1d.csv` : cumul des précipitations au pas journalier de chaque région administrative de chaque pays européen.

```{code-cell} ipython3
:id: QCGkj3VQVj1Z

!curl -sSL -q -o - "https://cloud.minesparis.psl.eu/index.php/s/MGp21fRa8LEzO3f/download?path=%2F&files=mig25_data.tgz" | tar -xzv
```

```{code-cell} ipython3
:id: F_ebaVmFHwMu

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

sns.set_style('whitegrid')
```

+++ {"id": "skjJZLvIaS1L"}

1. Charger les données dans des dataframes nommés `cf`, `ta` et `tp` :

```{code-cell} ipython3
:id: Ym2zAg01-ANR

cf = pd.read_csv("data/CF_1d.csv", index_col = "Date", parse_dates=True)#facteur de charge - échelle nationale
#Les données ci-dessous sont à léchelle régional de 1 à 21
ta = pd.read_csv("data/TA_1d.csv", index_col = "Date", parse_dates=True)#température en Kelvin
tp = pd.read_csv("data/TP_1d.csv", index_col = "Date", parse_dates=True)#précipitation en mètre
cf.tail(5)
```

+++ {"id": "NxvjYl3baki6", "editable": true, "slideshow": {"slide_type": ""}}

2. Extraire les données en rapport avec la France continentale (21 régions) pour chaque variable :

```{code-cell} ipython3
---
id: SheFL8ZR-Ca-
editable: true
slideshow:
  slide_type: ''
---
# on ne garde que ce qui contient des fr
cf_fr = cf[["FR"]]
ta_fr = ta[[name for name in ta.columns if "FR" in name]]
tp_fr = tp[[name for name in tp.columns if "FR" in name]]
ta_fr.shape
# ou cols = ta.columns[ta.columns.str.startswith("FR")]
```

+++ {"id": "F0OYV_TTEJfp"}

### Exploration et analyse des données

3. Afficher des informations de base sur les dataframes :

```{code-cell} ipython3
:id: 8w6C4seh98Hx

#cf_fr.info()
#ta_fr.info()
#tp_fr.info()
```

+++ {"id": "jM9FVVZnrMWD"}

4. a) Visualiser les données disponibles pour une région :

```{code-cell} ipython3
:id: ghLbqgJLA9vB

#rajouter le code jusqu'à 4.c
```

+++ {"id": "ipN9MFk2aim4"}

4. b) Comment pourriez-vous organiser ces données pour comparer les profils journaliers d'une année à l'autre ? Visualiser ces derniers sous forme de courbes et d'une heatmap.

```{code-cell} ipython3
:id: W9CxhcY-92qM

# votre code ici
```

+++ {"id": "wIfT_MEOdFSo"}

4. c) Comment pourriez-vous résumer statistiquement ces profils sur l’ensemble des années pour chaque jour (quantiles, moyenne, etc) ?

```{code-cell} ipython3
:id: deV9HuXe9tMN

# votre code ici
```

+++ {"id": "loLbQfTKrhK7"}

### Feature engineering

L'étape préliminaire dans le processus de développement d'un modèle de machine learning est de construire ses variables de décision pour qualifier ses observations. C'est une étape clé de l'ingénierie des données. Vous verrez que de mauvaises données (brutes, reconstruites ou composées) ne conduisent à aucun bon résultat.

5. Construire un nouveau dataframe `data` de 3 colonnes : les températures moyennes, le cumul moyen des précipitations et le facteur de charge :

```{code-cell} ipython3
:id: kGucUwQ0-T4S

data = pd.concat([cf_fr,ta_fr.mean(axis=1),tp_fr.mean(axis=1)], axis = 1)
data = data.rename(columns = {"FR":"CF",0:"TA",1:"TP"})
data.head(5)
```

+++ {"id": "OA-PX64ILCx-"}

## Modélisation

Avant d'attaquer réellement la modélisation, il nous reste une dernière étape de traitement de données. Il nous faut désormais séparer nos données en plusieurs jeux de données :

- un jeu d'entraînement,
- un jeu de validation,
- un jeu de test.

Selon les modèles d'apprentissage que nous sélectionnerons, nous aurons besoin de standardiser/normaliser nos valeurs.

+++ {"id": "ln4MBjn5oDu9"}

### Découpage du jeu de données

6. a) Séparer les variables de décision et la cible en 2 variables `X` et `y`.  
   b) Créer 2 jeux de données pour l'entrainement et le test  à l'aide de de la fonction [`train_test_split`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html).  
   c) Standardiser les variables de décision avec [`StandardScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)

```{code-cell} ipython3
---
id: nH8CcojVtXuK
editable: true
slideshow:
  slide_type: ''
---
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Features & Target
X = data.drop(columns="CF")
y = data["CF"]

# Séparation des données d'entrainement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=365, shuffle=False)

# Normalisation
scaler = StandardScaler().set_output(transform="pandas")
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

results = {"Actual": y_test}
```

+++ {"id": "vtcXr00KpRsI", "editable": true, "slideshow": {"slide_type": ""}}

### Choix du modèle

Nous en avons fini avec les données, tout est prêt pour modéliser notre problème. Nous allons commencer avec des modèles simples de régression. Pour nos premiers pas, nous utiliserons les modèles suivants :

* Régression linéaire : [`LinearRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
* Régression linéaire avec pénalité L1 [`Lasso`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)
* Régression linéaire avec pénalité L2 [`Ridge`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
* Arbre de décision : [`DecisionTreeRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)

Pour pousser plus loin, nous verrons également les modèles suivants :
* Forêt aléatoire : [`RandomForestRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
* Boosting de gradient : [`GradientBoostingRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)

*Les étapes d'évaluation et d'interprétation de la 3ème partie se feront en même temps que la modélisation et l'entrainement.*

```{code-cell} ipython3
:id: Ya_VGZXI2fmU

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

+++ {"id": "IGiMnJThjKpt"}

#### 1. Régression linéaire

$$
\hat{\beta} = \arg\min_{\beta}
\left(
\sum_{i=1}^{n} (y_i - \mathbf{x}_i^\top \beta)^2
\right)
$$

Nous allons commencer par un modèle de régression linéaire `LinearRegression`.

```{code-cell} ipython3
:id: iQdJy4dqLTpP

from sklearn.linear_model import LinearRegression

lr = LinearRegression()  # modèle de régression linéaire
lr.fit(X_train, y_train)  # apprentissage supervisé

y_pred = lr.predict(X_test)  # prédiction
y_pred = pd.Series(y_pred, index=y_test.index)
results["LinReg"] = y_pred

# Métriques
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

# Paramètres
w_ta, w_tp = lr.coef_
bias = lr.intercept_

print(f"R2: {r2:.06f}")
print(f"MSE: {mse:.06f}")
print(f"Weight[ta]: {w_ta:.6f}")
print(f"Weight[tp]: {w_tp:.6f}")
print(f"Bias: {bias:.6f}")
print()

display_result(y_test, y_pred) #De bonne tendance mais des extrêmes très éloignésd
```

+++ {"id": "wtpw1KZwhWoA"}

#### 2. Régression Lasso (L1)

$$
\hat{\beta} = \arg\min_{\beta}
\left(
\sum_{i=1}^{n} (y_i - \mathbf{x}_i^\top \beta)^2
\;+\;
\lambda \sum_{j=1}^{p} |\beta_j|
\right)
$$

Pour changer de modèle, c'est aussi simple que de changer son nom : de `LinearRegression` à `Lasso`.

```{code-cell} ipython3
:id: CV8RyFHuYy-e

from sklearn.linear_model import Lasso

lasso = Lasso()  # modèle de régression linéaire avec pénalité L1
lasso.fit(X_train, y_train)  # apprentissage supervisé

y_pred = lasso.predict(X_test)  # prédiction
y_pred = pd.Series(y_pred, index=y_test.index)
results["Lasso"] = y_pred

# Métriques
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

# Paramètres
w_ta, w_tp = lasso.coef_
bias = lasso.intercept_

print(f"R2: {r2:.06f}")
print(f"MSE: {mse:.06f}")
print(f"Weight[ta]: {w_ta:.6f}")
print(f"Weight[tp]: {w_tp:.6f}")
print(f"Bias: {bias:.6f}")
print()

display_result(y_test, y_pred) #on a sûrement mit un lanmbda trop grando
```

+++ {"id": "DdSnwXmUJMr8"}

7. Observez les prédictions réalisez ? Pourquoi un tel comportement et d'où provient ce résultat.

+++ {"id": "awB5EcxriOAE"}

#### Régression Ridge (L2)

$$
\hat{\beta} = \arg\min_{\beta}
\left(
\sum_{i=1}^{n} (y_i - \mathbf{x}_i^\top \beta)^2
\;+\;
\lambda \sum_{j=1}^{p} \beta_j^2
\right)
$$

Vous l'aurez compris pour faire un modèle `Ridge`, il suffit d'instancier le modèle du même nom. Ici nous allons observer 2 comportement différents selon les données passées à l'entrainement : données brutes ou données standardisées.

+++ {"id": "hJn1D-_XjAts"}

**A. Sur données brutes :**

```{code-cell} ipython3
:id: N8W9JcRT756X

from sklearn.linear_model import Ridge
ridge = Ridge()  # modèle de régression linéaire avec pénalité L1
ridge.fit(X_train, y_train)  # apprentissage supervisé

y_pred = ridge.predict(X_test)  # prédiction
y_pred = pd.Series(y_pred, index=y_test.index)
results["Ridge"] = y_pred

# Métriques
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

# Paramètres
w_ta, w_tp = ridge.coef_
bias = ridge.intercept_

print(f"R2: {r2:.06f}")
print(f"MSE: {mse:.06f}")
print(f"Weight[ta]: {w_ta:.6f}")
print(f"Weight[tp]: {w_tp:.6f}")
print(f"Bias: {bias:.6f}")
print()

display_result(y_test, y_pred)
```

+++ {"id": "JKtmLIdfix8S"}

**B. Sur données standardisées :**

```{code-cell} ipython3
:id: lwBxWl4O78ev

from sklearn.linear_model import Ridge
ridgestd = Ridge()  # modèle de régression linéaire avec pénalité L1
ridgestd.fit(X_train_std, y_train)  # apprentissage supervisé

y_pred = ridgestd.predict(X_test_std)  # prédiction
y_pred = pd.Series(y_pred, index=y_test.index)
results["Ridge"] = y_pred

# Métriques
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

# Paramètres
w_ta, w_tp = ridgestd.coef_
bias = ridgestd.intercept_

print(f"R2: {r2:.06f}")
print(f"MSE: {mse:.06f}")
print(f"Weight[ta]: {w_ta:.6f}")
print(f"Weight[tp]: {w_tp:.6f}")
print(f"Bias: {bias:.6f}")
print()


display_result(y_test, y_pred) #Quasi le même que le premier modèle
```

+++ {"id": "tGIXhOgiJBpz"}

8. Observez les prédictions réalisées. Que remarquez vous ?

+++ {"id": "N8gjR0PwA7iS"}

#### Arbre de décision

Même si nous changeons de type de modèle, la méthodologie reste la même. Par contre, il est évident que les paramètres du modèle ne seront plus les mêmes (poids et biais pour la régression linéaire vs variables, seuils et valeurs de prédiction pour l'arbre de décision)

```{code-cell} ipython3
:id: R9kS4Bw8ZCN6

from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
dt.fit(X_train, y_train)  # apprentissage supervisé

y_pred = dt.predict(X_test)  # prédiction
y_pred = pd.Series(y_pred, index=y_test.index)
results["DecisionTree"] = y_pred

display_result(y_test, y_pred)#bon les arbres.... peut pertinant
```

+++ {"id": "nEadHXHvZYbw"}

Nous pouvons visualiser sous forme de table les différents paramètres du modèle :

```{code-cell} ipython3
:id: CB6cWH9nBxwk

dmap = dict(enumerate(X.columns)) | {-2: "Leave"}
params = {"Feature": dt.tree_.feature,
          "Threshold": dt.tree_.threshold,
          "Value": dt.tree_.value.squeeze()}

params = pd.DataFrame(params).replace({"Feature": dmap})
```

+++ {"id": "ny_G5Nr9Zxr0"}

Il est même possible de visualiser facilement l'arbre de décision :

```{code-cell} ipython3
from sklearn.tree import plot_tree

fig, ax = plt.subplots(figsize=(18, 9))
plot_tree(dt, feature_names=X.columns, filled=True, fontsize=10, max_depth=3, ax=ax)
plt.show()
```

+++ {"id": "dAvKuso1LnSg"}

#### Recherche par grille

Jusqu'à présent, nous avons utilisé nos 4 modèles sans configurer quoi ce soit :

```
lr = LinearRegression()
lasso = Lasso()
ridge = Ridge()
dt = DecisionTreeRegressor()
```

Cela manque de souplesse n'est ce pas ? Comment régler correctement le coefficient de pénalité dans mes régressions ou définir la profondeur optimale de mon arbre de décision ? Il s'agit donc pour nous de configurer les meilleurs hyperparamètres du modèle afin de contrôler son apprentissage.

Pour cela nous utiliserons une recherche par grille : [`GridSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)

> **Note** : il ne faut pas confondre les paramètres d'une fonction (ou pluôt ses arguments) qui sont les hyperparamètres du modèle avec les paramètres du modèle qui sont les variables internes permettant de sortir une prédiction après apprentissage.

```{code-cell} ipython3
:id: ji1BzCNoIXdb

from sklearn.model_selection import GridSearchCV

params = {
    "max_depth": np.arange(1, 10),
}
reg = DecisionTreeRegressor(random_state=2024)  # modèle d'arbre de décision
cv = GridSearchCV(reg, param_grid=params)  # recherche par grille
cv.fit(X_train, y_train)  # apprentissage supervisé

y_pred = cv.predict(X_test)  # prédiction
y_pred = pd.Series(y_pred, index=y_test.index)
results["DTCV"] = y_pred

# Métriques
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"R2: {r2:.06f}")
print(f"MSE: {mse:.06f}")
print()

display_result(y_test, y_pred)
```

```{code-cell} ipython3
:id: 5BepN2qqLvkk

pd.DataFrame(cv.cv_results_)
```

```{code-cell} ipython3
:id: NExTlFMUi7Y1

from sklearn.tree import plot_tree

fig, ax = plt.subplots(figsize=(18, 9))
plot_tree(cv.best_estimator_, feature_names=X.columns, filled=True, fontsize=10, max_depth=3, ax=ax)
plt.show()
```

+++ {"id": "EW4S_YYxoGeH"}

### Evaluation et Interprétation

Nous avons constaté que nos modèles ne sont pas bons mais nous n'avons pas pu les visualiser simultanément sur un même graphique.

```{code-cell} ipython3
:id: a87_yXZWodXP

dfr = pd.DataFrame(results)
dfr
```

```{code-cell} ipython3
:id: cHKnLk89o83b

fig, axs = plt.subplots(ncols=2, nrows=3, figsize=(15, 9), sharex=True, sharey=True, constrained_layout=True)

for ax, col in zip(axs.flatten(), dfr.columns[1:]):
  dfr[col].plot(ax=ax, lw=0.8, color="tab:red", title=col)
  dfr["Actual"].plot(ax=ax, lw=0.8, color="tab:blue")
plt.show()
```

```{code-cell} ipython3
#Random forest
from sklearn.ensemble import RandomForestRegressor
params = {
    "max_depth": np.array([4,5,6,8,10]),
    "min_samples_leaf":np.array([1,5,10]),
    "n_estimators":np.array([100,150,200,300])
}
rd = RandomForestRegressor()
cv = GridSearchCV(rd, param_grid=params, n_jobs=11)  # recherche par grille
cv.fit(X_train, y_train)  # apprentissage supervisé

y_pred = cv.predict(X_test)  # prédiction
y_pred = pd.Series(y_pred, index=y_test.index)
results["DecisionTree"] = y_pred

display_result(y_test, y_pred)
```

```{code-cell} ipython3
#Gradient Boosting
from sklearn.ensemble import GradientBoostingRegressor
params = {
    "max_depth": np.array([8,10]),
    "min_samples_leaf":np.array([5,10]),
    "n_estimators":np.array([200,300]),
    "learning_rate":np.array([0.01,0.1])
}
gb = GradientBoostingRegressor(n_estimators=200)
cv = GridSearchCV(gb, param_grid=params, n_jobs=11)
cv.fit(X_train, y_train)  # apprentissage supervisé

y_pred = cv.predict(X_test)  # prédiction
y_pred = pd.Series(y_pred, index=y_test.index)
results["DecisionTree"] = y_pred

display_result(y_test, y_pred)
```

```{code-cell} ipython3
#Création du tableau saison
cos = [np.cos(2*np.pi*(t/365)) for t in range(3285)]
sin = [np.sin(2*np.pi*(t/365)) for t in range(3285)]
Saison = pd.DataFrame([[x,y] for x, y in zip(cos,sin)], columns = [5,6])

#PCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
Xt = pd.concat([ta_fr,tp_fr],axis=1,)
yt = cf_fr.copy()
# Séparation des données d'entrainement et de test
Xt_train, Xt_test, yt_train, yt_test = train_test_split(Xt, yt, test_size=365, shuffle=False)
Saison_train, Saison_test, _,_ = train_test_split(Saison, yt, test_size=365,shuffle=False)
# Normalisation
scaler = StandardScaler().set_output(transform="pandas")
Xt_train_std = scaler.fit_transform(Xt_train)
Xt_test_std = scaler.transform(Xt_test)

results = {"Actual": yt_test}

# PCA
pca = PCA(n_components=5)
X_train_pca = pca.fit_transform(Xt_train_std)
X_test_pca = pca.transform(Xt_test_std)

#Tableau pandas
X_train_pca = pd.DataFrame(X_train_pca)
X_test_pca = pd.DataFrame(X_test_pca)

#Ajout de la donné saison
X_train_pca = pd.concat([X_train_pca,Saison_train],axis=1)
Saison_test = Saison_test.reset_index(drop=True)
X_test_pca = pd.concat([X_test_pca,Saison_test],axis = 1)
X_test_pca
```

```{code-cell} ipython3
#Random forest pca
from sklearn.ensemble import RandomForestRegressor
params = {
    "max_depth": np.array([6,8,10]),
    "min_samples_leaf":np.array([1,5,10]),
    "n_estimators":np.array([100,200,300])
}
rd = RandomForestRegressor()
cv = GridSearchCV(rd, param_grid=params, n_jobs=11)  # recherche par grille
cv.fit(X_train_pca, y_train)  # apprentissage supervisé

y_pred = cv.predict(X_test_pca)  # prédiction
y_pred = pd.Series(y_pred, index=y_test.index)
results["DecisionTree"] = y_pred

print(r2_score(y_test,y_pred))
display_result(y_test, y_pred) 
```

```{code-cell} ipython3
#Gradient Boosting
from sklearn.ensemble import GradientBoostingRegressor
params = {
    "max_depth": np.array([8,10]),
    "min_samples_leaf":np.array([5,10]),
    "n_estimators":np.array([200,300]),
    "learning_rate":np.array([0.01,0.1])
}
gb = GradientBoostingRegressor(n_estimators=200)
cv = GridSearchCV(gb, param_grid=params, n_jobs=11)
cv.fit(X_train_pca, y_train)  # apprentissage supervisé

y_pred = cv.predict(X_test_pca)  # prédiction
y_pred = pd.Series(y_pred, index=y_test.index)
results["DecisionTree"] = y_pred

print(r2_score(y_test,y_pred))
display_result(y_test, y_pred)
```

+++ {"id": "F2S_rzAlV46g"}

# Et maintenant ?

Nous avons vu que nos simples variables ne sont pas suffisantes pour réaliser un modèle performant. Toutefois cela nous a permis de développer rapidement un premier modèle d'apprentissage automatique.

Désormais, il va nous falloir créer des variables explicatives plus en adéquation avec le problème que nous tentons de modéliser.

Sans être hydrologue ou météorologue, il est nécessaire de comprendre un minimum les phénomènes physiques liés au cycle de l'eau :

![Cycle de l'eau](https://geotechniquehse.com/wp-content/uploads/2024/10/hydrogeologie-cycle-de-leau.png)

9. **Que proposeriez vous comme nouvelles variables explicatives ?**

**La réponse a cette question passe par l'étude de la corrélation spatiale et temporelle qui lie les variables climatiques au facteur de charge.**

Quand vous aurez des variables en adéquation avec votre problème, vous pourrez utiliser des modèles plus performants comme les forêts aléatoire et le boosting de gradient, voire des réseaux de neurones.

+++ {"id": "cLVIUYbZBViO"}

# Informations (très) utiles

+++ {"id": "pMuH44LCBpS6"}

## Méthodes d'encodage des données pour l'apprentissage

Tout au long du processus de création des variables, il arrivera que nous devions les mettre sous une autre forme. Voici quelques une des principales transformations :

1. [`OneHotEncoder`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)  
   **Objectif :** Transformer des variables catégorielles en une forme que les modèles peuvent comprendre en créant une colonne binaire pour chaque catégorie.  
   **A utiliser :**
   - pour les variables **nominales** (sans ordre ou relation entre les catégories).
   - lorsqu'il y a des catégories discrètes et qu'il est nécessaire de les traiter indépendamment.

2. [`StandardScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)  
   **Objectif :** Standardiser les données numériques de manière à ce qu’elles soient centrées autour de leur moyenne et comparables par leur écart-type. Cela permet de traiter des variables avec différentes échelles.  
   **A utiliser :**
   - pour les modèles sensibles à l'échelle des données, tels que les régressions linéaires, les SVM, ou les réseaux de neurones.
   - lorsque les données ont des unités différentes ou des amplitudes différentes.

3. [`MinMaxScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)  
   **Objectif :** Mettre à l'échelle les données entre une plage spécifiée, généralement entre 0 et 1.  
   **A utiliser :**
   - lorsqu'il est nécessaire que les données soient dans un intervalle spécifique, surtout pour les modèles sensibles à l'échelle (comme les réseaux de neurones, où l'activation se fait souvent sur des valeurs entre 0 et 1).
   - lorsque les données doivent être dans un certain intervalle.

4. Cyclical Features Encoding  
   **Objectif :** Capturer la relation cyclique des données saisonnières en les transformant sur un cercle unitaire avec les fonctions trigonométriques `sin` et `cos`.  
   **A utiliser :**
     - pour des données cycliques.
     - lorsqu'il est nécessaire de préserver l'ordre temporel et la continuité.

+++ {"id": "o8v9CrbxEp3Q"}

## Méthodes de découpage des données pour la validation croisée

La création de jeux de données de séries temporelles dans le cadre de prévision se fait rarement de manière aléatoire. Cela peut entraîner des problèmes de généralisation et ne représente pas le cas d'usage principal.

1. **[`TimeSeriesSplit`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) : Découpage temporel séquentiel**  
   - Convient pour les séries temporelles où l'ordre chronologique est crucial.  
   - Les données sont découpées de manière progressive : chaque split utilise une portion plus grande des données passées pour l'entraînement, et les données futures pour le test.  
   - Les indices sont respectés pour ne pas mélanger les informations futures dans l'entraînement.  
   - Exemple :
     - Split 1 : Train = [2015], Test = [2016]  
     - Split 2 : Train = [2015, 2016], Test = [2017]  
     - Split 3 : Train = [2015, 2016, 2017], Test = [2018].

2. **[`GroupKFold`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html) : Découpage par groupes (par exemple, années)**  
   - Permet de s'assurer que les groupes (comme les années ou d'autres identifiants logiques) ne sont jamais mélangés entre l'entraînement et le test.  
   - Chaque split utilise des groupes différents pour l'entraînement et le test.  
   - Utile lorsque les données doivent rester groupées par identifiant logique.  
   - Exemple :
     - Split 1 : Train = [2016, 2017, 2018], Test = [2015]  
     - Split 2 : Train = [2015, 2017, 2018], Test = [2016]  
     - Split 3 : Train = [2015, 2016, 2018], Test = [2017].
     - Split 4 : Train = [2015, 2016, 2017], Test = [2018].

> **Note :** Ces deux méthodes peuvent être directement utilisées comme paramètre de [`GridSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) pour optimiser les hyperparamètres tout en respectant les spécificités des données.
