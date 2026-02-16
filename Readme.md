#  Comprendre le fonctionnement d'un neurone

- Pour les détails la roadmap est juste après
- On va utiliser [julia](https://julialang.org/)
  - Manual: [Getting started](https://docs.julialang.org/en/v1/manual/getting-started/)
- Il existe un notebook natif pour Julia:
  - Executer le notebook [Pluto](https://plutojl.org/):
```Julia
using Pluto
Pluto.run()
```
  - mais on peut aussi utilise Jupyter, il faut installer le package `IJulia`. Le Markdown dans jupyter accepte LaTeX ce qui permet d'utiliser une syntax mathématique.
```sh
# jupyter lab --no-browser
```
- Status:
  - [x] Étape 0 — Vision globale
  - [x] 🧮 Étape 1 — Comprendre la dérivée comme "sensibilité"
  - [x] 📉 Étape 2 — Minimisation d’une fonction (descente de gradient 1D)
  - [x] 🌄 Étape 3 — Gradient en dimension 2+
  - [x] 🔗 Étape 4 — Règle de la chaîne (LE cœur du deep learning)
  - [x] ⚙️ Étape 5 — Construire un neurone à la main
  - [ ] 🔁 Étape 6 — Backpropagation (sans l’appeler comme ça)
  - [ ] 🧱 Étape 7 — Plusieurs entrées

---

# 🧠 Roadmap : des dérivées → gradients → neurone réel

Objectif final : comprendre profondément ce qui se passe dans un neurone (type perceptron / MLP), mathématiquement
et intuitivement.

## Étape 0 — Vision globale

Un neurone =

```
entrée → somme pondérée → fonction d’activation → sortie
```

Mathématiquement :

```
y = σ(w₁x₁ + w₂x₂ + ... + b)
```

L’apprentissage consiste à ajuster les poids `w` pour minimiser une erreur.

Et tout repose sur :
→ dérivées
→ gradients
→ descente de gradient

## 🧮 Étape 1 — Comprendre la dérivée comme "sensibilité"

### Objectif

Comprendre :

> "Si je change un paramètre un tout petit peu, qu’est-ce que ça change au résultat ?"

Dans Pluto :

* tracer une fonction simple
* afficher sa dérivée
* voir l’effet local

Exemples à explorer :

```
f(x) = x²
f(x) = sin(x)
f(x) = e^x
```

Expériences :

* zoomer autour d’un point
* visualiser la pente
* approximer numériquement :

```
f'(x) ≈ (f(x+h) - f(x)) / h
```

Intuition clé :

> Une dérivée mesure une influence.

## 📉 Étape 2 — Minimisation d’une fonction (descente de gradient 1D)

### Objectif

Comprendre comment on "apprend" sans neurone.

Problème :

```
minimiser f(x) = (x-3)^2
```

Algorithme :

```
x = x - η * f'(x)
```

Explorer :

* différents learning rates
* convergence
* divergence

Pluto est parfait pour :

* slider sur η
* voir la trajectoire

Concept clé :

> Apprendre = descendre une pente.

## 🌄 Étape 3 — Gradient en dimension 2+

Passer à :

```
f(x, y) = x² + y²
```

Comprendre :

* gradient = direction de plus forte montée
* descente = direction opposée

Expériences :

* champ de vecteurs
* trajectoire d’optimisation

Intuition clé :

> Le gradient indique "où corriger".

## 🔗 Étape 4 — Règle de la chaîne (LE cœur du deep learning)

C’est le moment le plus important.

Comprendre :

```
f(x) = sin(x²)
```

Dérivée :

```
f'(x) = cos(x²) * 2x
```

Puis généraliser :

```
z = f(g(x))
```

La dérivée se propage.

Concept clé :

> L’erreur peut remonter dans un système composé.

C’est exactement :
**backpropagation**

## ⚙️ Étape 5 — Construire un neurone à la main

Créer un neurone simple :

```
y = σ(wx + b)
```

Choisir :

* σ = sigmoid
* ou tanh
* ou ReLU

Puis définir une erreur :

```
L = (y - target)^2
```

Et calculer :

```
dL/dw
dL/db
```

## 🔁 Étape 6 — Backpropagation (sans l’appeler comme ça)

Faire un mini système :

```
x → neurone → erreur
```

Puis :

* calculer dérivées à la main
* ajuster w
* observer l’apprentissage

C’est LE moment "aha".

## 🧱 Étape 7 — Plusieurs entrées

Passer à :

```
y = σ(w1x1 + w2x2 + b)
```

Visualiser :

* plan de décision
* frontière linéaire

Comprendre :

> Un neurone = séparateur linéaire.

## 🌐 Étape 8 — Plusieurs neurones (MLP)

Construire :

```
2 entrées → 3 neurones → 1 sortie
```

Et voir apparaître :

* non-linéarité
* formes complexes
* XOR

Moment clé :

> Un seul neurone est limité.
> Plusieurs = intelligence émergente.

## 🧪 Étape 9 — Pluto vs Clojure (comment mixer)

### Pluto → idéal pour :

* visualisation
* intuition
* maths
* sliders
* exploration

### Clojure → idéal pour :

* live coding
* systèmes dynamiques
* exploration structurelle
* manipulations fonctionnelles

# 🧩 Suggestion d’architecture hybride

## Julia/Pluto

* explorer maths
* tracer gradients
* visualiser optimisation

## Clojure

* implémenter neurones
* expérimenter architectures
* jouer avec propagation

# 🧭 Ordre conseillé (très important)

1. dérivées 1D
2. gradient descent 1D
3. gradients 2D
4. règle de la chaîne
5. 1 neurone
6. apprentissage du neurone
7. multi-neurones

# 💡 Projet fil rouge (excellent pour apprendre)

Créer un neurone qui apprend à approximer :

```
y = x²
```

Puis :

```
y = sin(x)
```

Puis :

```
classifier des points 2D
```

---

Parfait. On va faire la chaîne complète, très lentement, et en gardant exactement ton intuition sur
“les autres variables sont des constantes”.

On part d’un neurone simple.

---

## 1️⃣ Le neurone comme composition de fonctions

On a 3 étapes :

1. Somme pondérée
   [
   z = w_1 x_1 + w_2 x_2 + b
   ]

2. Activation
   [
   y = \sigma(z)
   ]

3. Erreur
   [
   E = \frac12 (y - y_{\text{vrai}})^2
   ]

Donc en réalité :

[
w_i \rightarrow z \rightarrow y \rightarrow E
]

C’est une fonction composée.

---

## 2️⃣ On veut quoi exactement ?

On veut savoir :

[
\frac{\partial E}{\partial w_i}
]

C’est-à-dire :

> Si je change légèrement le poids (w_i), comment l’erreur change ?

---

## 3️⃣ La chaîne complète (version simple)

La règle de la chaîne donne :

[
\frac{\partial E}{\partial w_i}
===============================

\frac{\partial E}{\partial y}
\cdot
\frac{\partial y}{\partial z}
\cdot
\frac{\partial z}{\partial w_i}
]

C’est exactement comme :

[
h(x) = f(g(u(x)))
\Rightarrow
h'(x) = f'(g(u)) \cdot g'(u) \cdot u'(x)
]

---

## 4️⃣ Calcul des 3 morceaux (un par un)

### a) Dérivée de l’erreur par rapport à y

[
E = \frac12 (y - y_{\text{vrai}})^2
]

Donc :

[
\frac{\partial E}{\partial y} = (y - y_{\text{vrai}})
]

Intuition :
si y est trop grand → erreur positive
si y est trop petit → erreur négative

---

### b) Dérivée de y par rapport à z

[
y = \sigma(z)
]

Pour la sigmoid :

[
\frac{\partial y}{\partial z} = \sigma(z)(1 - \sigma(z))
]

Donc :

[
\frac{\partial y}{\partial z} = y(1-y)
]

---

### c) Dérivée de z par rapport à w_i

[
z = w_1 x_1 + w_2 x_2 + b
]

On dérive par rapport à (w_i).

Tous les autres termes sont constants.

Il reste juste :

[
\frac{\partial z}{\partial w_i} = x_i
]

C’est exactement ce que tu as compris avec :

> f(x,y) = x + y → dérivée par rapport à x = 1 car y est constant.

---

## 5️⃣ On assemble tout

[
\frac{\partial E}{\partial w_i}
===============================

(y - y_{\text{vrai}})
\cdot
y(1-y)
\cdot
x_i
]

C’est LE résultat central.

---

## 6️⃣ Interprétation très intuitive

Chaque facteur a un sens :

* (y - y_{\text{vrai}})
  → à quel point on se trompe

* (y(1-y))
  → à quel point le neurone réagit au changement de z

* (x_i)
  → à quel point ce poids participe au résultat

Donc :

> Si (x_i = 0) → ce poids ne joue aucun rôle → gradient = 0

C’est très logique.

---

## 7️⃣ Et là tu vois le lien profond avec la règle de la chaîne

On a littéralement :

[
w_i \rightarrow z \rightarrow y \rightarrow E
]

Et on “remonte” :

* erreur → sortie → somme → poids

C’est exactement ça la **backpropagation**.

---

## 8️⃣ Image mentale importante (la plus juste)

Pense à une chaîne :

```
w_i ──► z ──► y ──► E
```

La dérivée remonte :

```
∂E/∂w_i =
(∂E/∂y)
× (∂y/∂z)
× (∂z/∂w_i)
```

C’est juste :

> la règle de la chaîne appliquée à un système en cascade.

---

## 9️⃣ Le moment clé à retenir

Tu as débloqué le point le plus difficile :

> Quand on dérive par rapport à w_i, tous les autres w_j sont des constantes.

C’est exactement ça qui rend le calcul possible.

Et c’est ça qui permet :

* un gradient pour w1
* un gradient pour w2
* un gradient pour chaque poids

Indépendamment.

---

## 10️⃣ Question pour vérifier un déclic profond

Est-ce que cette phrase te paraît maintenant claire :

> “Backpropagation, c’est juste la règle de la chaîne appliquée à une longue composition de fonctions.”

Ou tu sens encore un petit flou quelque part ?

