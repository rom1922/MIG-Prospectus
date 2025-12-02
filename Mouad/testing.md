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

CF=pd.read_csv("data/CF_1d.csv")
TA=pd.read_csv("data/TA_1d.csv")
TP=pd.read_csv("data/TP_1d.csv")
print(CF.columns)
print(TA.columns)
print(TP.columns)
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
cf = CF[["Date", "FR"]].copy()   # 👈 très important
# Colonnes de TA qui correspondent à la France
fr_cols_ta = [col for col in TA.columns if "FR" in col]

# Colonnes de TP qui correspondent à la France
fr_cols_tp = [col for col in TP.columns if "FR" in col]

# On garde Date + toutes les colonnes FR
ta = TA.loc[:, ["Date"] + fr_cols_ta]
tp = TP.loc[:, ["Date"] + fr_cols_tp]
ta = ta.set_index("Date")
tp = tp.set_index("Date")
cf = cf.set_index("Date")
```

+++ {"id": "F0OYV_TTEJfp"}

### Exploration et analyse des données

3. Afficher des informations de base sur les dataframes :

```{code-cell} ipython3
:id: 8w6C4seh98Hx

cf.info()
```

+++ {"id": "jM9FVVZnrMWD"}

4. a) Visualiser les données disponibles pour une région :

```{code-cell} ipython3
:id: ghLbqgJLA9vB

# 1) S'assurer que la date est au bon format
cf["Date"] = pd.to_datetime(cf["Date"])

# 2) Tracer l'évolution du facteur de charge pour la France
plt.figure(figsize=(12, 4))
plt.plot(cf["Date"], cf["FR"])
plt.xlabel("Date")
plt.ylabel("Facteur de charge")
plt.title("Évolution du facteur de charge - France")
plt.tight_layout()
plt.show()
```

+++ {"id": "ipN9MFk2aim4"}

4. b) Comment pourriez-vous organiser ces données pour comparer les profils journaliers d'une année à l'autre ? Visualiser ces derniers sous forme de courbes et d'une heatmap.

```{code-cell} ipython3
:id: W9CxhcY-92qM

region = "FR10"  
TA["Date"] = pd.to_datetime(TA["Date"])
TP["Date"] = pd.to_datetime(TP["Date"])
TA["annee"] = TA["Date"].dt.year
TA["jour_annee"] = TA["Date"].dt.dayofyear
TP["annee"] = TP["Date"].dt.year
TP["jour_annee"] = TP["Date"].dt.dayofyear
tp_reg = TP[["annee", "jour_annee", region]].copy()

mat_TP = tp_reg.pivot(index="jour_annee", columns="annee", values=region)

plt.figure(figsize=(10, 6))
sns.heatmap(
    mat_TP,
    cmap="Blues",
    cbar_kws={"label": "Précipitations (mm)"}
)
plt.xlabel("Année")
plt.ylabel("Jour de l'année")
plt.title(f"Précipitations journalières - région {region}")
plt.tight_layout()
plt.show()
```

+++ {"id": "wIfT_MEOdFSo"}

4. c) Comment pourriez-vous résumer statistiquement ces profils sur l’ensemble des années pour chaque jour (quantiles, moyenne, etc) ?

```{code-cell} ipython3
:id: deV9HuXe9tMN

# 4.c) Résumer statistiquement les profils journaliers sur toutes les années
# ici pour la température (mat_TA). Même idée pour mat_TP.

# Stats de base par jour (sur toutes les années)
stats_TP = pd.DataFrame({
    "moyenne": mat_TP.mean(axis=1),
    "mediane": mat_TP.median(axis=1),
    "minimum": mat_TP.min(axis=1),
    "maximum": mat_TP.max(axis=1),
    "ecart_type": mat_TP.std(axis=1),
})

# Quelques quantiles par jour
quantiles_TP = mat_TP.quantile([0.10, 0.25, 0.5, 0.75, 0.90], axis=1).T
quantiles_TP.columns = ["q10", "q25", "q50", "q75", "q90"]

# Tableau final : une ligne = un jour de l'année, colonnes = stats
stats_TP = pd.concat([stats_TP, quantiles_TP], axis=1)
stats_TP.head()
plt.figure(figsize=(12,5))
plt.plot(stats_TP.index, stats_TP["moyenne"], label="Moyenne")
plt.fill_between(
    stats_TP.index,
    stats_TP["q10"],
    stats_TP["q90"],
    alpha=0.3,
    label="[10%, 90%]"
)
plt.xlabel("Jour de l'année")
plt.ylabel("Température")
plt.title("Résumé statistique des profils journaliers (toutes années confondues)")
plt.legend()
plt.tight_layout()
plt.show()
```

+++ {"id": "loLbQfTKrhK7"}

### Feature engineering

L'étape préliminaire dans le processus de développement d'un modèle de machine learning est de construire ses variables de décision pour qualifier ses observations. C'est une étape clé de l'ingénierie des données. Vous verrez que de mauvaises données (brutes, reconstruites ou composées) ne conduisent à aucun bon résultat.

5. Construire un nouveau dataframe `data` de 3 colonnes : les températures moyennes, le cumul moyen des précipitations et le facteur de charge :

```{code-cell} ipython3
:id: kGucUwQ0-T4S

# Colonnes de régions françaises dans ta / tp
cols_fr_ta = [c for c in ta.columns if c.startswith("FR")]
cols_fr_tp = [c for c in tp.columns if c.startswith("FR")]

# Moyenne sur les régions françaises (ligne par ligne)
temp_moy = ta[cols_fr_ta].mean(axis=1)
precip_moy = tp[cols_fr_tp].mean(axis=1)

# DataFrame final
data = pd.DataFrame({
    "T": temp_moy,
    "P": precip_moy,
    "CF": CF["FR"]
})
data.head()
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
    """Affiche les résultats de prédiction / réels."""
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

    # Plot 2
    ax2.set_title("Actual vs Predicted")
    ax2.set_xlabel("Actual")
    ax2.set_ylabel("Predicted")
    ax2.scatter(y_true, y_pred, color="tab:blue", s=10)

    left, right = ax2.get_xlim()
    bottom, top = ax2.get_ylim()
    lb = min(left, bottom) - 0.01
    ub = max(right, top) + 0.01
    ax2.set_ylim(lb, ub)
    ax2.set_xlim(lb, ub)
    ax2.axline((lb, lb), (ub, ub), color="tab:red")

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

display_result(y_test, y_pred)
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

display_result(y_test, y_pred)
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

# votre code ici

from sklearn.linear_model import Ridge
ridge=Ridge(alpha=1000)
ridge.fit(X_train,y_train)
y_pred=ridge.predict(X_test)
r2=r2_score(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
wt,wp= ridge.coef_
b=ridge.intercept_

print(f"R2:{r2:.06f}")
print(f"MSE:{mse:.06f}")
print(f"Wt:{wt:.06f}")
print(f"Wp:{wp:.06f}")
display_result(y_test,y_pred)
```

+++ {"id": "JKtmLIdfix8S"}

**B. Sur données standardisées :**

```{code-cell} ipython3
:id: lwBxWl4O78ev

# votre code ici

from sklearn.linear_model import Ridge
ridge=Ridge(alpha=0.8)
ridge.fit(X_train_std,y_train)
y_pred=ridge.predict(X_test_std)
r2=r2_score(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
wt,wp= ridge.coef_
b=ridge.intercept_

print(f"R2:{r2:.06f}")
print(f"MSE:{mse:.06f}")
print(f"Wt:{wt:.06f}")
print(f"Wp:{wp:.06f}")
display_result(y_test,y_pred)
```

+++ {"id": "tGIXhOgiJBpz"}

8. Observez les prédictions réalisées. Que remarquez vous ?

+++ {"id": "N8gjR0PwA7iS"}

#### Arbre de décision

Même si nous changeons de type de modèle, la méthodologie reste la même. Par contre, il est évident que les paramètres du modèle ne seront plus les mêmes (poids et biais pour la régression linéaire vs variables, seuils et valeurs de prédiction pour l'arbre de décision)

```{code-cell} ipython3
:id: R9kS4Bw8ZCN6

# votre code ici

from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor(
    max_depth=3,          # profondeur maximale
     min_samples_leaf=200,  # nb minimal d'exemples par feuille
    random_state=0
)
dt.fit(X_train,y_train)
y_pred=dt.predict(X_test)
r2=r2_score(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
display_result(y_test,y_pred)
print(f"R2:{r2:.06f}")
print(f"MSE:{mse:.06f}")
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
params
```

+++ {"id": "ny_G5Nr9Zxr0"}

Il est même possible de visualiser facilement l'arbre de décision :

```{code-cell} ipython3
---
id: 51v_uhE-IDgr
editable: true
slideshow:
  slide_type: ''
---
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
yc_pred = pd.Series(y_pred, index=y_test.index)
results["DTCV"] = yc_pred

# Métriques
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"R2: {r2:.06f}")
print(f"MSE: {mse:.06f}")

display_result(y_test, y_pred)
cv.best_estimator_
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

```{code-cell} ipython3
#préparation de DATA
ta.index = pd.to_datetime(ta.index)
tp.index = pd.to_datetime(tp.index)
cf.index = pd.to_datetime(cf.index)
df = cf.join([ta.add_prefix("T_"), tp.add_prefix("P_")], how="inner")
df = df.rename(columns={"FR": "CF"})
hydro_strong = ["FRK2","FRL0","FRI2","FRJ2","FRK1","FRG0","FRJ1","FRI1"]  # Alpes, Pyrénées, Massif Central, Jura
hydro_medium = ["FRF1","FRF3","FRH0","FRB0","FRC1","FRC2"]               # Vosges, Morvan, etc.
hydro_weak   = ["FR10","FRE1","FRE2","FRD1","FRD2","FRF2","FRI3"]        # grandes plaines nord / ouest
def add_season_features(df):
    dayofyear = df.index.dayofyear
    df["day_sin"] = np.sin(2 * np.pi * dayofyear / 365)
    df["day_cos"] = np.cos(2 * np.pi * dayofyear / 365)
    return df

def add_cf_lags(df, lags=[1, 7, 30]):
    for L in lags:
        df[f"CF_lag{L}"] = df["CF"].shift(L)
    return df

def rolling_sum(df, cols, window, min_periods=None, suffix=""):
    if min_periods is None:
        min_periods = window // 3
    for c in cols:
        df[f"{c}_sum{window}{suffix}"] = df[c].rolling(window, min_periods=min_periods).sum()
    return df

def rolling_mean(df, cols, window, min_periods=None, suffix=""):
    if min_periods is None:
        min_periods = window // 3
    for c in cols:
        df[f"{c}_mean{window}{suffix}"] = df[c].rolling(window, min_periods=min_periods).mean()
    return df
```

```{code-cell} ipython3
#Expérience 1
df1 = df.copy()
T_cols = [c for c in df1.columns if c.startswith("T_")]
P_cols = [c for c in df1.columns if c.startswith("P_")]

df1["T_nat"] = df1[T_cols].mean(axis=1)
df1["P_nat"] = df1[P_cols].mean(axis=1)

df1 = add_season_features(df1)
df1 = add_cf_lags(df1, lags=[1, 7, 30])

df1["P_nat_sum7"]   = df1["P_nat"].rolling(7,  min_periods=3).sum()
df1["P_nat_sum30"]  = df1["P_nat"].rolling(30, min_periods=10).sum()
df1["P_nat_sum90"]  = df1["P_nat"].rolling(90, min_periods=30).sum()
df1["T_nat_mean7"]  = df1["T_nat"].rolling(7,  min_periods=3).mean()
df1["T_nat_mean30"] = df1["T_nat"].rolling(30, min_periods=10).mean()

feature_cols_1 = [
    "T_nat","P_nat",
    "T_nat_mean7","T_nat_mean30",
    "P_nat_sum7","P_nat_sum30","P_nat_sum90",
    "day_sin","day_cos",
    "CF_lag1","CF_lag7","CF_lag30"
]

data1 = df1[feature_cols_1 + ["CF"]].dropna()
X1 = data1[feature_cols_1]
y1 = data1["CF"]
```

```{code-cell} ipython3
#Expérience 2
df2 = df.copy()

# 1) Moyennes par groupe
df2["T_hydro_strong"] = df2[[f"T_{r}" for r in hydro_strong]].mean(axis=1)
df2["P_hydro_strong"] = df2[[f"P_{r}" for r in hydro_strong]].mean(axis=1)

df2["T_hydro_medium"] = df2[[f"T_{r}" for r in hydro_medium]].mean(axis=1)
df2["P_hydro_medium"] = df2[[f"P_{r}" for r in hydro_medium]].mean(axis=1)

df2["T_hydro_weak"] = df2[[f"T_{r}" for r in hydro_weak]].mean(axis=1)
df2["P_hydro_weak"] = df2[[f"P_{r}" for r in hydro_weak]].mean(axis=1)

# 2) Rolling sur ces colonnes
for col in ["P_hydro_strong","P_hydro_medium","P_hydro_weak"]:
    df2[f"{col}_sum30"] = df2[col].rolling(30, min_periods=10).sum()
    df2[f"{col}_sum90"] = df2[col].rolling(90, min_periods=30).sum()

for col in ["T_hydro_strong","T_hydro_medium","T_hydro_weak"]:
    df2[f"{col}_mean7"] = df2[col].rolling(7, min_periods=3).mean()

# 3) Saison + lags CF
df2 = add_season_features(df2)
df2 = add_cf_lags(df2, lags=[1,7,30])

feature_cols_2 = [
    "T_hydro_strong","T_hydro_medium","T_hydro_weak",
    "P_hydro_strong","P_hydro_medium","P_hydro_weak",
    "P_hydro_strong_sum30","P_hydro_strong_sum90",
    "P_hydro_medium_sum30","P_hydro_medium_sum90",
    "P_hydro_weak_sum30","P_hydro_weak_sum90",
    "T_hydro_strong_mean7","T_hydro_medium_mean7","T_hydro_weak_mean7",
    "day_sin","day_cos",
    "CF_lag1","CF_lag7","CF_lag30"
]

data2 = df2[feature_cols_2 + ["CF"]].dropna()
X2 = data2[feature_cols_2]
y2 = data2["CF"]
```

```{code-cell} ipython3
#Expérience 3
df_corr = df.copy()

# Ex : cumul de pluie sur 90 jours pour chaque région
P_cols = [c for c in df_corr.columns if c.startswith("P_")]
for c in P_cols:
    df_corr[f"{c}_sum90"] = df_corr[c].rolling(90, min_periods=30).sum()

# Corrélation avec CF
corrs = {}
for c in P_cols:
    corrs[c] = df_corr[f"{c}_sum90"].corr(df_corr["CF"])

# Top 5 en valeur absolue
top_regions = sorted(corrs, key=lambda k: abs(corrs[k]), reverse=True)[:5]
top_regions = [c.replace("P_","") for c in top_regions]  # remettre le code région pur

print("Top régions météo corrélées au CF :", top_regions)

df3 = df.copy()

for r in top_regions:
    df3[f"P_{r}_sum30"] = df3[f"P_{r}"].rolling(30, min_periods=10).sum()
    df3[f"P_{r}_sum90"] = df3[f"P_{r}"].rolling(90, min_periods=30).sum()
    df3[f"T_{r}_mean7"] = df3[f"T_{r}"].rolling(7,  min_periods=3).mean()

df3 = add_season_features(df3)
df3 = add_cf_lags(df3, lags=[1,7,30])

feature_cols_3 = []
for r in top_regions:
    feature_cols_3 += [f"P_{r}_sum30", f"P_{r}_sum90", f"T_{r}_mean7"]

feature_cols_3 += ["day_sin","day_cos","CF_lag1","CF_lag7","CF_lag30"]

data3 = df3[feature_cols_3 + ["CF"]].dropna()
X3 = data3[feature_cols_3]
y3 = data3["CF"]
```

```{code-cell} ipython3
#Expérience 4
df4 = df.copy()

for r in hydro_strong:
    df4[f"P_{r}_sum30"] = df4[f"P_{r}"].rolling(30, min_periods=10).sum()
    df4[f"P_{r}_sum90"] = df4[f"P_{r}"].rolling(90, min_periods=30).sum()
    df4[f"T_{r}_mean7"] = df4[f"T_{r}"].rolling(7,  min_periods=3).mean()

df4 = add_season_features(df4)
df4 = add_cf_lags(df4, lags=[1,7,30])

feature_cols_4 = []
for r in hydro_strong:
    feature_cols_4 += [f"P_{r}_sum30", f"P_{r}_sum90", f"T_{r}_mean7"]

feature_cols_4 += ["day_sin","day_cos","CF_lag1","CF_lag7","CF_lag30"]

data4 = df4[feature_cols_4 + ["CF"]].dropna()
X4 = data4[feature_cols_4]
y4 = data4["CF"]
```

```{code-cell} ipython3
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

def run_rf_experiment(df, feature_cols, param_grid, name="exp"):
    """
    df           : DataFrame complet (index = Date, colonne cible = 'CF')
    feature_cols : liste des colonnes X à utiliser pour cette expérience
    param_grid   : dict d'hyperparamètres pour GridSearchCV
    name         : nom de l'expérience (juste pour l'affichage)
    """

    # --- 1. Découpage Train / Test : on garde les 365 derniers jours pour le test
    train = df.iloc[:-365].copy()
    test  = df.iloc[-365:].copy()

    X_train = train[feature_cols]
    y_train = train["CF"]

    X_test  = test[feature_cols]
    y_test  = test["CF"]

    # --- 2. Modèle de base
    rf_base = RandomForestRegressor(
        bootstrap=True,      # tu voulais le garder à True
        random_state=2024,
        n_jobs=-1            # utilise tous les cœurs dispo
    )

    # TimeSeriesSplit pour respecter le temps dans la CV
    tscv = TimeSeriesSplit(n_splits=5)

    grid = GridSearchCV(
        rf_base,
        param_grid=param_grid,
        cv=tscv,
        scoring="r2",
        n_jobs=-1
    )

    # --- 3. Entraînement + recherche d’hyperparamètres sur le TRAIN
    grid.fit(X_train, y_train)

    best_rf = grid.best_estimator_

    print(f"\n===== {name} =====")
    print("Meilleurs hyperparamètres :", grid.best_params_)
    print(f"R2 moyen en CV : {grid.best_score_:.3f}")

    # --- 4. Évaluation sur le TEST (les 365 derniers jours)
    y_pred = best_rf.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print(f"R2 sur test : {r2:.3f}")
    print(f"MSE sur test : {mse:.6f}")

    # Si tu as déjà la fonction du cours :
    try:
        display_result(y_test, y_pred)
    except NameError:
        pass  # si display_result n'existe pas dans ton notebook

    return best_rf, y_test, y_pred
param_grid_exp1 = {
    "n_estimators": [100, 300, 600],
    "max_depth": [3, 5, 7, None],
    "min_samples_leaf": [1, 3, 5]
}
# Expérience 1
best_rf1, y_test1, y_pred1 = run_rf_experiment(
    df,
    feature_cols=features_exp1,
    param_grid=param_grid_exp1,
    name="Expérience 1 - features simples"
)

# Expérience 2 (autre grille si tu veux)
param_grid_exp2 = {
    "n_estimators": [100, 300, 800],
    "max_depth": [5, 10, None],
    "min_samples_leaf": [1, 2, 4]
}

best_rf2, y_test2, y_pred2 = run_rf_experiment(
    df,
    feature_cols=features_exp2,
    param_grid=param_grid_exp2,
    name="Expérience 2 - moyennes & cumuls"
)
```

```{code-cell} ipython3
#Implémentation du modèle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
params={"n_estimators":np.arange(50,300,50),"max_depth":np.arange(1,6)}
RF=RandomForestRegressor(n_estimators=1000,max_depth=3,bootstrap=True,random_state=2024,n_jobs=6)
#cv=GridSearchCV(RF,param_grid=params)
RF.fit(X_train,y_train)
y_pred=RF.predict(X_test)
r2=r2_score(y_test,y_pred)
print(f"R2:{r2:.06f}")
display_result(y_test,y_pred)
```
