- Objectif: Comprendre le fonctionnement d'un neurone
- Run:
  - from [Julia](https://julialang.org) we use Pluto:
```Julia
using Pluto
Pluto.run()
```
  - and load the notebook

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

# 🧮 Étape 1 — Comprendre la dérivée comme "sensibilité"

## Objectif

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

# 📉 Étape 2 — Minimisation d’une fonction (descente de gradient 1D)

## Objectif

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

# 🌄 Étape 3 — Gradient en dimension 2+

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

# 🔗 Étape 4 — Règle de la chaîne (LE cœur du deep learning)

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

# ⚙️ Étape 5 — Construire un neurone à la main

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

# 🔁 Étape 6 — Backpropagation (sans l’appeler comme ça)

Faire un mini système :

```
x → neurone → erreur
```

Puis :

* calculer dérivées à la main
* ajuster w
* observer l’apprentissage

C’est LE moment "aha".

# 🧱 Étape 7 — Plusieurs entrées

Passer à :

```
y = σ(w1x1 + w2x2 + b)
```

Visualiser :

* plan de décision
* frontière linéaire

Comprendre :

> Un neurone = séparateur linéaire.

# 🌐 Étape 8 — Plusieurs neurones (MLP)

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

# 🧪 Étape 9 — Pluto vs Clojure (comment mixer)

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
