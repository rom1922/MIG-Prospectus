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

+++ {"id": "PcCQhBvLpl09"}

Si nous reprenons l'exemple de démonstration du MLP où nous essayons de faire apprendre à un réseau de neurones que pour une entrée `x=[0.23, 0.82]`, nous devons obtenir en sortie `y=1`.

L'architecture du MLP est la suivante :
- une couche d'entrée à 2 neurones,
- une fonction d'activation (sigmoid)
- une couche cachée à 2 neurones,
- une fonction d'activation (<s>sigmoid</s> relu)
- une couche de sortie à 1 neurone.

```{code-cell} ipython3
import torch
```

+++ {"id": "auoPrqAODhcx"}

## Step 1 : implémentation de la solution en NumPy

La solution NumPy pourrait ressembler à cela :

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: 8C6uKzvCrfGT
outputId: d54ef9f2-941c-46c9-daeb-4ca5c16b3d43
---
import numpy as np

# Data
# Les variables d'entrée
x = np.array([[0.23, 0.82]])

# Les poids de la couche d'entrée
w0 = np.array([[0.1, 0.5],
               [0.4, 0.3]])

# Les poids de la couche cachée
w1 = np.array([[0.2],
               [0.6]])

# La sortie attendue
y = np.array([1])

# Le taux d'apprentissage (learning rate)
n = 0.7

# Apprentissage supervisé
for epoch in range(5):
    print(f"[Epoch {epoch}]")

    ### Forward step  ###

    # Architecture de notre réseau de neurones
    net1 = x @ w0
    o1 = 1 / (1 + np.exp(-net1))  # sigmoid

    net2 = o1 @ w1
    o2 = np.maximum(0,net2) # ReLu

    # Calcul de l'erreur
    E = 0.5 * ((y - o2)**2)
    print(f"E: {E}")
    print(f"o2: {o2}")

    ### Backward step ###

    # Optimisation pour minimiser l'erreur
    #delta2 = (o2 - y) * o2 * (1 - o2) #pour la sigmoïd
    delta2 = (o2 - y) * (net2 > 0).astype(float)
    grad_w1 = o1.T @ delta2

    delta1 = delta2 @ w1.T * o1 * (1 - o1)
    grad_w0 = x.T @ delta1

    # Mise à jour des paramètres
    w0 -= n * grad_w0
    w1 -= n * grad_w1
    print(f"∂E/∂w0:\n {grad_w0}")
    print(f"∂E/∂w1:\n {grad_w1}")

    print(f"w0:\n {w0}")
    print(f"w1:\n {w1}")
    print()
```

+++ {"id": "6_bgHwtVpkzU"}

Nous voyons qu'au bout de la 5ième itération, la valeur de `o2` tend vers 1 et donc très proche de `y` et l'erreur `E` est minime. Notre modèle a maintenant des paramètres (`w0` et `w1`) optimaux.

+++ {"id": "qXNGZcVGDe4k"}

## Step 2 : conversion en PyTorch

Néanmoins, il est fastidieux de devoir calculer les gradients, de mettre à jour les paramètres, etc à chaque epoch. Imaginons que notre modèle contienne des milliers de paramètres (voire des milliards comme les LLMs actuels), ça devient impensable. Evidemment il existe des solutions techniques. Pour ce module, nous utiliserons la librairie [PyTorch](https://pytorch.org/).

> **Note** : [Le tutorial de PyTorch](https://docs.pytorch.org/tutorials/beginner/basics/intro.html) est vraiment très bien fait, consultez le dès que possible.

Nous allons transformer pas à pas le code NumPy précédent en un modèle PyTorch :

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: aLDiqiTA3K7f
outputId: 6e9b4e26-c787-494f-fe97-8358feab3003
---
# Data
# Les variables d'entrée
x = torch.tensor([[0.23, 0.82]])  # (1)

# Les poids de la couche d'entrée
w0 = torch.tensor([[0.1, 0.5],
                   [0.4, 0.3]], requires_grad=True)  # (2)

# Les poids de la couche cachée
w1 = torch.tensor([[0.2],
                   [0.6]], requires_grad=True)

# La sortie attendue
y = torch.Tensor([1])

# Le taux d'apprentissage (learning rate)
n = 0.7

# Apprentissage supervisé
for epoch in range(5):
    print(f"[Epoch {epoch}]")

    ### Forward step ###

    # Architecture de notre réseau de neurones
    net1 = x @ w0
    o1 = torch.sigmoid(net1)  # (3)

    net2 = o1 @ w1
    o2 = torch.relu(net2)

    # Calcul de l'erreur
    E = 0.5 * ((y - o2)**2)
    print(f"E: {E.item()}")  # (4)
    print(f"o2: {o2.item()}")

    ### Backward step ###

    # Réinitialisation des gradients (pas d'accumulation)
    w0.grad = None  # (5)
    w1.grad = None

    # Optimisation pour minimiser l'erreur
    E.backward()  # (6)

    # Mise à jour des paramètres
    with torch.no_grad():  # (7)
        w0 -= n * w0.grad
        w1 -= n * w1.grad
        print(f"∂E/∂w0:\n {w0.grad.numpy()}")
        print(f"∂E/∂w1:\n {w1.grad.numpy()}")

    print(f"w0:\n {w0.detach().numpy()}")  # (8)
    print(f"w1:\n {w1.detach().numpy()}")
    print()
```

+++ {"id": "yfbWXYeKQqpV"}

L'affichage reste le même donc il semble que le modèle reste équivalent. Plusieurs points méritent notre attention :

1. Le [`ndarray`](https://numpy.org/doc/stable/reference/generated/numpy.array.html) NumPy s'est transformé en [`tensor`](https://docs.pytorch.org/docs/stable/generated/torch.tensor.html) PyTorch :
```
torch.tensor([[0.23, 0.82]])
```

2. Les paramètres pour lesquels nous voulons calculer le gradient sont créés avec un argument spécial :
```
w0 = torch.tensor([[0.1, 0.5],
                   [0.4, 0.3]], requires_grad=True)
```

3. Contrairement à NumPy, il existe des fonctions d'activation prêtes à l'emploi dans PyTorch donc nous les utilisons car elles sont optimisées notamment pour le calcul sur GPU/TPU :
```
torch.sigmoid(net1)
...
torch.relu(net2)
```

4. Pour récupérer les valeurs numériques d'un tenseur, nous pouvons utiliser les méthodes [`item()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html), [`tolist()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.tolist.html) et [`numpy()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.numpy.html)
```
E.item()
...
w0.grad.numpy()
```

5. Avant de calculer les gradients, nous devons nous assurer qu'ils sont réinitialisés pour éviter l'accumulation à chaque itération.
```
w0.grad = None
w1.grad = None
```

6. C'est ici que la différence notable se fait. Pour calculer les gradients, il n'y a qu'à appeler la méthode [`backward()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.backward.html) de la feuille du graphe computationnel. Tous les tenseurs qui ont été créés explicitement avec le paramètre `requires_grad=True` ainsi que tous les tenseurs intermédiaires qui sont calculés avec ces derniers auront un attribut `.grad` mis à jour avec la valeur de leur gradient.
```
E.backward()
```

7. Si vous voulez réaliser des opérations sur les tenseurs que PyTorch ne doit pas mémoriser, il faut désactiver l'enregistrement des opérations dans le graphe computationnel avec la fonction [`torch.nograd()`](https://docs.pytorch.org/docs/stable/generated/torch.no_grad.html) :
```
with torch.no_grad():
   # ici les opérations ne sont pas enregistrées dans le graphe computationnel
   ...
# à partir d'ici les opérations sont à nouveau enregistrées
```

8. Quand nous voulons récupérer une valeur numérique d'un tenseur qui nécessite d'avoir son gradient calculé pendant que nous sommes en train d'enregistrer des opérations, cela génère une exception. Il est nécessaire de détacher le tenseur du graphe pour récupérer sa représentation numérique à l'aide de la méthode [`detach()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.detach.html):
```
w0.detach().numpy()
w1.detach().numpy()
```

+++ {"id": "QirNjvCl_BEW"}

## Step 3 : intégration d'un optimiseur

Tout ceci est nettement mieux puisque nous n'avons plus à gérer le calcul des gradients. Toutefois dans le code précédent, il faut encore mettre à jour manuellement les paramètres du modèle par rétropropagation du gradient. Il est possible d'utiliser un optimiseur qui gèrera pour nous la mise à jour des paramètres. Les 2 principaux optimiseurs sont :
- [`SGD`](https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html) : Stochastic Gradient Descent
- [`Adam`](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html) : Adaptive Moment Estimation

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: _--bhJjcBE3I
outputId: 176ace3b-e096-480f-a784-804fdeb1d422
---
import torch
from torch.optim import SGD

# Data
# Les variables d'entrée
x = torch.tensor([[0.23, 0.82]])

# Les poids de la couche d'entrée
w0 = torch.tensor([[0.1, 0.5],
                   [0.4, 0.3]], requires_grad=True)

# Les poids de la couche cachée
w1 = torch.tensor([[0.2],
                   [0.6]], requires_grad=True)

# La sortie attendue
y = torch.Tensor([1])

# Le taux d'apprentissage (learning rate)
n = 0.7

# L'optimiseur
optimizer = SGD([w0, w1], lr=n)  # (1)

# Apprentissage supervisé
for epoch in range(5):
    print(f"[Epoch {epoch}]")

    ### Forward step ###

    # Architecture de notre réseau de neurones
    net1 = x @ w0
    o1 = torch.sigmoid(net1)

    net2 = o1 @ w1
    o2 = torch.relu(net2)

    # Calcul de l'erreur
    E = 0.5 * ((y - o2)**2)
    print(f"E: {E.item()}")
    print(f"o2: {o2.item()}")

    ### Backward step ###

    # Réinitialisation des gradients (pas d'accumulation)
    optimizer.zero_grad()  # (2)

    # Optimisation pour minimiser l'erreur
    E.backward()

    # Mise à jour des paramètres
    optimizer.step()  # (3)

    with torch.no_grad():
        print(f"∂E/∂w0:\n {w0.grad.numpy()}")
        print(f"∂E/∂w1:\n {w1.grad.numpy()}")

    print(f"w0:\n {w0.detach().numpy()}")
    print(f"w1:\n {w1.detach().numpy()}")
    print()
```

+++ {"id": "O9UtvzVc72R_"}

1. Nous instancions l'optimiseur de descente de gradient (`SGD`) avec la liste des paramètres à optimiser (ici `w0` et `w1`) ainsi que le taux d'apprentissage (`learning_rate`) pour indiquer à quel rythme les paramètres doivent être mis à jour à chaque itération :
```
optimizer = SGD([w0, w1], lr=n)
```

2. Comme précédemment, nous ne voulons pas accumuler le gradient entre chaque itération donc nous le réinitialisons avant le calcul des gradients :
```
optimizer.zero_grad()
```

3. Après le calcul des gradients, nous pouvons utiliser ces derniers pour ajuster les paramètres en appliquant une mise à jour proportionnelle au taux d'apprentissage :
```
optimizer.step()
```

## Step 4 : Séparation du modèle

Nous voyons que notre boucle d'apprentissage cherche à optimiser les paramètres de notre modèle de deep learning. Alors que l'architecture de notre modèle peut changer, la boucle d'apprentissage semble rester la même : il serait donc intéressant de séparer fonctionnellement les deux parties d'autant plus que lorsque notre modèle sera réglé, nous n'aurons plus besoin de la boucle d'apprentissage.

Pour ce faire, nous allons créer une classe pour implémenter notre architecture. PyTorch fournit une classe de base [`nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html) pour développer nos modèles :

```{code-cell} ipython3
---
id: IkTpnuv2ppkG
colab:
  base_uri: https://localhost:8080/
outputId: 2a275094-08d5-4384-edb1-edc5784948de
---
import torch
import torch.nn as nn
from torch.optim import SGD

class MyMLP(nn.Module):  # (1)
    def __init__(self):
        super().__init__()

        # Les poids de la couche d'entrée
        w0 = torch.tensor([[0.1, 0.5],
                           [0.4, 0.3]])

        # Les poids de la couche cachée
        w1 = torch.tensor([[0.2],
                           [0.6]])

        self.w0 = nn.Parameter(w0, requires_grad=True)  # (2)
        self.w1 = nn.Parameter(w1, requires_grad=True)

    def forward(self, x):  # (3)
        # Architecture de notre réseau de neurones
        net1 = x @ self.w0
        o1 = torch.sigmoid(net1)

        net2 = o1 @ self.w1
        o2 = torch.relu(net2)

        return o2

# Data
# Les variables d'entrée
x = torch.tensor([[0.23, 0.82]])

# La sortie attendue
y = torch.Tensor([1])

# Le taux d'apprentissage (learning rate)
n = 0.7

# Modèle de régression
model = MyMLP()

# L'optimiseur
optimizer = SGD(model.parameters(), lr=n)

# Apprentissage supervisé
for epoch in range(5):
    print(f"[Epoch {epoch}]")

    ### Forward step ###
    o2 = model(x)

    # Calcul de l'erreur
    E = 0.5 * ((y - o2)**2)
    print(f"E: {E.item()}")
    print(f"o2: {o2.item()}")

    ### Backward step ###

    # Réinitialisation des gradients (pas d'accumulation)
    optimizer.zero_grad()

    # Optimisation pour minimiser l'erreur
    E.backward()

    # Mise à jour des paramètres
    optimizer.step()

    with torch.no_grad():
        print(f"∂E/∂w0:\n {model.w0.grad.numpy()}")
        print(f"∂E/∂w1:\n {model.w1.grad.numpy()}")

    print(f"w0:\n {model.w0.detach().numpy()}")
    print(f"w1:\n {model.w1.detach().numpy()}")
    print()
```

+++ {"id": "OX_zDgoHF3f1"}

1. Il suffit simplement d'hériter de cette classe pour disposer des nombreuses fonctionnalités pensées et éprouvées par les développeurs et l'instancier (à l'instar des `DataFrame` de Pandas) :

```python
class MyMLP(nn.Module):
    ...

model = MyMLP()
```

2. Plutôt que de passer la liste explicite des paramètres de notre modèle à optimiser avec `SGD`, nous pouvons les déclarer en tant que tel grâce à la classe [`nn.Parameter`](https://docs.pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html). Nous utiliserons la méthode [`parameters()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.parameters) de notre modèle pour collecter les paramètres à optimiser par `SGD` :
```python
self.w0 = nn.Parameter(w0, requires_grad=True)
self.w1 = nn.Parameter(w1, requires_grad=True)

...

optimizer = SGD(model.parameters(), lr=n)
```

3. La méthode la plus importante d'un modèle est [`forward()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.forward). Pour pouvoir réaliser le calcul de la sortie à partir des entrées, il ne nous restera plus qu'à appeler l'instance du modèle. C'est PyTorch qui se chargera d'appeler la méthode `forward` pour nous :
```python
def forward(self, x):
    ...

o2 = model(x)
```

+++ {"id": "cSvYl0sP6qMn"}

## Step 5 : Amélioration du modèle

Jusqu'à présent, nous avons géré manuellement deux couches de neurones et optimisé leurs poids à l'aide d'une application linéaire. Mais que se passe-t-il si nos données présentent un biais ? Dans ce cas, il devient pertinent d'introduire un paramètre supplémentaire à optimiser, ce qui revient à passer d'une transformation linéaire à une transformation affine. Plutôt que de continuer à implémenter ce biais à la main, nous allons désormais déléguer cette responsabilité à la couche [`nn.Linear`](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.linear.Linear.html) de PyTorch, qui gère automatiquement les poids et le biais associés.

```{code-cell} ipython3
---
id: tI-a-oo-clxC
colab:
  base_uri: https://localhost:8080/
outputId: 231ebf23-3eee-4d8e-f3ec-094a7c63becd
---
import torch
import torch.nn as nn
from torch.optim import SGD

class MyMLP(nn.Module):
    def __init__(self):
        super().__init__()

        # Les poids de la couche d'entrée
        w0 = torch.tensor([[0.1, 0.5],
                           [0.4, 0.3]])

        # Les poids de la couche cachée
        w1 = torch.tensor([[0.2],
                           [0.6]])

        self.layer0 = nn.Linear(in_features=2, out_features=2, bias=False)  # (1)
        self.layer1 = nn.Linear(in_features=2, out_features=1, bias=False)

        with torch.no_grad():
            self.layer0.weight.copy_(w0.T)  # (2)
            self.layer1.weight.copy_(w1.T)

    def forward(self, x):
        # Architecture de notre réseau de neurones
        net1 = self.layer0(x)  # (3)
        o1 = torch.sigmoid(net1)

        net2 = self.layer1(o1)
        o2 = torch.relu(net2)

        return o2

# Data
# Les variables d'entrée
x = torch.tensor([[0.23, 0.82]])

# La sortie attendue
y = torch.Tensor([1])

# Le taux d'apprentissage (learning rate)
n = 0.7

# Modèle de régression
model = MyMLP()

# L'optimiseur
optimizer = SGD(model.parameters(), lr=n)

# Apprentissage supervisé
for epoch in range(5):
    print(f"[Epoch {epoch}]")

    ### Forward step ###
    o2 = model(x)

    # Calcul de l'erreur
    E = 0.5 * ((y - o2)**2)
    print(f"E: {E.item()}")
    print(f"o2: {o2.item()}")

    ### Backward step ###

    # Réinitialisation des gradients (pas d'accumulation)
    optimizer.zero_grad()

    # Optimisation pour minimiser l'erreur
    E.backward()

    # Mise à jour des paramètres
    optimizer.step()

    with torch.no_grad():
        print(f"∂E/∂w0:\n {model.layer0.weight.grad.numpy().T}")  # (4)
        print(f"∂E/∂w1:\n {model.layer1.weight.grad.numpy().T}")

    print(f"w0:\n {model.layer0.weight.detach().numpy().T}")
    print(f"w1:\n {model.layer1.weight.detach().numpy().T}")
    print()
```

+++ {"id": "3rzA9faZD9rv"}

1. Comme il s'agit d'une couche de neurones générique, il faut lui spécifier à minima les tailles d'entrée et de sortie. Par comptabilité avec notre programme initial, nous désactivons l'apprentissage du biais :
```python
self.layer0 = nn.Linear(in_features=2, out_features=2, bias=False)
self.layer1 = nn.Linear(in_features=2, out_features=1, bias=False)
```

2. Depuis le début, nous avons toujours spécifié les poids initiaux de notre modèle. Nous continuons à faire la même chose avec une légère subtilité quant à la forme de la transformation affine (voir la documentation) :
```python
with torch.no_grad():
    self.layer0.weight.copy_(w0.T)
    self.layer1.weight.copy_(w1.T)
```

3. Désormais, nous ne faisons plus nous même le produit matriciel mais déléguons cette tâche à nos couches de neurones :
```python
net1 = self.layer0(x)
o1 = torch.sigmoid(net1)

net2 = self.layer1(o1)
o2 = torch.relu(net2)
```

4. Attention, nous ne récupérons plus `w0` ou `w1` mais les poids de `layer0` et `layer1` et transposons le résultat pour garder le même affichage :
```python
print(f"∂E/∂w0:\n {model.layer0.weight.grad.numpy().T}")
print(f"∂E/∂w1:\n {model.layer1.weight.grad.numpy().T}")
```

+++ {"id": "1NH1X_i5OICy"}

### Mise en application

Il y a encore de nombreuses améliorations possibles (et indispensables) pour que notre modèle soit entrainé *dans les règles de l'art* mais c'est déjà suffisant pour le moment. En vous inspirant de ce qui vient d'être fait jusqu'à présent, développez un modèle MLP à partir du setup ci-dessous :

$$
\begin{align*}
y &= 10\sin(\pi x_{1} x_{2})
  + 20(x_{3} - 0.5)^{2}
  + 10 x_{4}
  + 5 x_{5}
  + \varepsilon, \\
  &\quad\varepsilon \sim \mathcal{N}(0,\sigma^{2}),
  \qquad x_i \sim \mathcal{U}(0,1) \quad \forall\, i \in \{1,\dots,n\}
\end{align*}
$$

```python
import torch
from sklearn.datasets import make_friedman1
from sklearn.model_selection import train_test_split

X, y = make_friedman1(n_samples=200, n_features=10, noise=1.0, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = torch.from_numpy(X_train.astype("float32"))
X_test  = torch.from_numpy(X_test.astype("float32"))
y_train = torch.from_numpy(y_train.astype("float32")).unsqueeze(dim=1)
y_test  = torch.from_numpy(y_test.astype("float32")).unsqueeze(dim=1)
```

| **Note** : plutôt que de calculer la fonction de perte manuellement, vous pouvez tirer profit de la classe [`MSELoss`](https://docs.pytorch.org/docs/stable/generated/torch.nn.MSELoss.html)

```{code-cell} ipython3
import torch
import torch.nn as nn
from torch.optim import SGD
from sklearn.datasets import make_friedman1
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
```

```{code-cell} ipython3
#Sample
X, y = make_friedman1(n_samples=200, n_features=10, noise=1.0, random_state=42) #entré en 5, sortie 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Nomralisation
scaler = StandardScaler().set_output(transform="pandas")
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

X_train = torch.from_numpy(X_train.astype("float32"))
X_test  = torch.from_numpy(X_test.astype("float32"))
y_train = torch.from_numpy(y_train.astype("float32")).unsqueeze(dim=1)
y_test  = torch.from_numpy(y_test.astype("float32")).unsqueeze(dim=1)


best_score = np.inf
model_save_path = ""
```

```{code-cell} ipython3
class FriedMines(nn.Module):
    def __init__(self):
        super().__init__()
        
        #couche 
        self.layer0 = nn.Linear(in_features=10, out_features=128, bias=True)
        self.layer1 = nn.Linear(in_features=128, out_features=64, bias=True)
        self.layer2 = nn.Linear(in_features=64, out_features=32, bias=True)
        self.layer3 = nn.Linear(in_features=32, out_features=16, bias=True)
        self.layer4 = nn.Linear(in_features=16, out_features=1, bias=False)

    def forward(self, x):
        # Architecture de notre réseau de neurones
        net1 = self.layer0(x)  # (3)
        o1 = torch.relu(net1)

        net2 = self.layer1(o1)
        o2 = torch.relu(net2)

        net3 = self.layer2(o2)
        o3 = torch.relu(net3)

        net4 = self.layer3(o3)
        o4 = torch.relu(net4)

        net5 = self.layer4(o4)
        o5 = net5

        return o5

# Le taux d'apprentissage (learning rate)
n = 0.01

# Modèle de régression
model = FriedMines()

#Modèle de l'erreur
criterion = nn.MSELoss()

# L'optimiseur
optimizer = SGD(model.parameters(), lr=n)

# Apprentissage supervisé
for epoch in range(1000):
    model.train()
    if (epoch)%500 == 0 :
        print(f"[Epoch {epoch}]")

    ### Forward step ###
    predict = model(X_train)

    # Calcul de l'erreur
    loss = criterion(predict, y_train)
    if (epoch)%500 == 0 :
        print(f"E: {loss.item()}")

    ### Backward step ###

    # Réinitialisation des gradients (pas d'accumulation)
    optimizer.zero_grad()

    # Optimisation pour minimiser l'erreur
    loss.backward()

    # Mise à jour des paramètres
    optimizer.step()
print(f'Efinale = {loss.item()}')
```

```{code-cell} ipython3
print(best_score.item())
```

```{code-cell} ipython3
model.eval()
with torch.no_grad():
    y_test_pred = model(X_test)
    loss = criterion(y_test_pred, y_test)
    print(f"Erreur de test = {loss.item()}")
```
