# Recommandation Séquentielle Dynamique par Apprentissage par Renforcement

> Système de recommandation intelligent basé sur le Deep Reinforcement Learning,
> modélisant le comportement séquentiel des utilisateurs dans un environnement e-commerce dynamique.

---

## Contexte Business

### Le problème : La "Fatigue de Recommandation"

Les plateformes comme Amazon, Netflix ou Spotify reposent encore largement sur des **filtres statistiques statiques**. Quand vous achetez un iPhone, l'algorithme vous propose un deuxième iPhone — parce qu'il voit une corrélation, pas un besoin.

Ce comportement crée trois effets négatifs mesurables :

| Effet | Impact |
|-------|--------|
| UX dégradée | L'utilisateur se sent harcelé, pas conseillé |
| CTR en baisse | Les recommandations répétitives sont ignorées |
| Manque à gagner | Les besoins complémentaires immédiats sont ratés |

> **Acquérir un nouveau client coûte 5 à 25× plus cher que d'en conserver un.**
> L'enjeu n'est pas le clic — c'est la valeur de toute la session.

### L'approche RL : Optimiser le long terme, pas le clic

Ce projet modélise la recommandation comme un **problème de décision séquentielle** (MDP) :
chaque recommandation influence les interactions futures. L'agent n'optimise pas un clic isolé,
il optimise la **récompense cumulée sur toute la session**.

Concrètement : l'agent apprend qu'il vaut mieux recommander une coque à 20 € qui sera achetée,
plutôt qu'un deuxième téléphone à 800 € qui sera ignoré.

### Leviers de rentabilité

**1. Maximisation du panier moyen (AOV)**
L'agent apprend le *cross-selling* optimal sans règles codées à la main.
L'accesoire pertinent est suggéré dans la fenêtre de quelques secondes après l'achat principal.

**2. Scalabilité industrielle (le saut DQN)**
- Q-Learning → prouve la faisabilité, limité au catalogue de petite taille
- DQN → traite des millions de produits, comprend les relations inter-catégories sans supervision

**3. Rétention et image de marque**
Un agent qui se comporte comme un "vendeur expert" plutôt qu'un robot publicitaire améliore
le NPS et réduit le taux de désabonnement.

---

## Architecture technique

### Formalisation MDP

| Symbole | Définition | Dans ce projet |
|---------|-----------|----------------|
| **S** | États | Index du dernier item vu par l'utilisateur |
| **A** | Actions | Index de l'item à recommander |
| **P** | Transition | Clic → s' = item recommandé |
| **R** | Récompense | +1 clic · +3 achat · 0 ignoré · −0.5 répétition |
| **γ** | Actualisation | 0.90 (sessions courtes) |

### Progression des modèles

```
Phase 0 │ Baseline aléatoire   → lower bound, référence minimale
Phase 1 │ Q-Learning (tabular)  → preuve de concept + démonstration des limites
Phase 2 │ DQN                   → réseau de neurones, scalable, embeddings items
Phase 3 │ GRU + DQN             → encodeur séquentiel, historique utilisateur complet
```

### Simulateur utilisateur

Le système inclut un simulateur qui modélise 4 profils réalistes :

| Profil | Affinité principale | Comportement |
|--------|--------------------|-----------   |
| `tech` | Électronique (85%) | Focalisé, peu de dispersion |
| `fashion` | Mode (90%) | Sensible aux tendances |
| `sport` | Sport (85%) | Achats fonctionnels |
| `random` | Toutes catégories (40%) | Exploration large |

Le simulateur intègre un mécanisme de **fatigue** : recommander la même catégorie en boucle
réduit progressivement la probabilité de clic.

---

## Structure du projet

```
tp_recommandation/
│
├── notebooks/
│   └── recommendation_rl.ipynb     ← notebook principal (toutes les phases)
│
├── env/
│   └── recommendation_env.py       ← environnement MDP (reset / step)
│
├── simulator/                      ← (réservé — simulateur avancé)
│
├── models/                         ← (réservé — DQN, GRU-DQN)
│
├── utils/
│   └── metrics.py                  ← CTR, diversité, hit rate, TD error...
│
├── data/
│   ├── amazon_loader.py            ← pipeline Amazon Electronics
│   ├── raw/                        ← fichiers .gz bruts (gitignorés)
│   └── processed/
│       ├── items.json              ← catalogue réel (après chargement)
│       └── sequences.json          ← séquences utilisateur triées par temps
│
├── app/                            ← (réservé — démo Flask)
│   └── templates/
│
└── README.md
```

---

## Installation

```bash
# Cloner le projet
git clone <repo_url>
cd tp_recommandation

# Créer l'environnement
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate

# Installer les dépendances
pip install numpy pandas matplotlib seaborn torch flask
```

---

## Utilisation

### Mode synthétique (immédiat, sans téléchargement)

```bash
jupyter notebook notebooks/recommendation_rl.ipynb
# Dans le notebook : USE_AMAZON = False
```

### Mode Amazon Electronics (données réelles)

```bash
# Étape 1 : télécharger et préparer le dataset (~15 min)
python data/amazon_loader.py

# Étape 2 : dans le notebook
# USE_AMAZON = True
jupyter notebook notebooks/recommendation_rl.ipynb
```

---

## Métriques d'évaluation

Le notebook collecte deux familles de métriques à chaque épisode :

### Métriques RL

| Métrique | Formule | Ce qu'elle mesure |
|----------|---------|------------------|
| **Reward cumulée** | $G_t = \sum \gamma^k R_{t+k}$ | Qualité globale de la politique |
| **TD Error** | $\|r + \gamma \max Q(s') - Q(s,a)\|$ | Convergence de l'apprentissage |

### Métriques recommandation

| Métrique | Formule | Ce qu'elle mesure |
|----------|---------|------------------|
| **CTR** | clics / recommandations | Pertinence immédiate |
| **Taux d'achat** | achats / recommandations | Conversion réelle |
| **Engagement Rate** | (clics + achats) / recommandations | Engagement global |
| **Hit Rate** | épisodes avec ≥1 clic / total | Fiabilité minimale |
| **Diversité** | items uniques / recommandations | Évitement de la bulle de filtre |
| **Repeat Rate** | items déjà vus / recommandations | Qualité du filtrage |
| **Couverture** | items recommandés au moins 1× / catalogue | Exploitation du catalogue |

---

## Résultats — Phase 1 (Q-Learning)

Résultats sur le catalogue synthétique (50 items, 300 épisodes d'évaluation) :

| Métrique | Random | Q-Learning initial | Q-Learning optimisé |
|----------|--------|-------------------|---------------------|
| Reward moyenne | 1.26 | 1.64 | *après optimisation* |
| CTR | 0.127 | 0.144 | *après optimisation* |
| Diversité | 0.915 | 0.234 | *après optimisation* |
| Couverture | 50/50 | 25/50 | *après optimisation* |

**Optimisations appliquées :**

| Paramètre | Avant | Après | Raison |
|-----------|-------|-------|--------|
| `alpha` | 0.10 | 0.03 | Stabilise les Q-values, TD error converge |
| `gamma` | 0.99 | 0.90 | Adapté aux sessions de 10 steps |
| `epsilon_min` | 0.05 | 0.12 | Maintient la diversité en exploitation |
| `epsilon_decay` | 0.995 | 0.998 | Exploration plus longue |
| `n_episodes` | 1 000 | 3 000 | Convergence avec alpha bas |
| Reward shaping | — | +0.2/nouvelle catégorie | Combat la bulle de filtre |

---

## Roadmap

- [x] Phase 0 — Baseline aléatoire
- [x] Phase 1 — Q-Learning tabulaire + optimisation
- [ ] Phase 2 — Deep Q-Network (DQN)
- [ ] Phase 3 — GRU + DQN (encodeur séquentiel)
- [ ] Phase 4 — Simulation 1 000 utilisateurs
- [ ] Phase 5 — Démo Flask e-commerce

---

## Références

- Mnih et al. (2015) — *Human-level control through deep reinforcement learning* (DQN)
- Ni et al. (2019) — *Justifying Recommendations using Distantly-Labeled Reviews* (Amazon dataset)
- Sutton & Barto (2018) — *Reinforcement Learning: An Introduction*
- Kang & McAuley (2018) — *Self-Attentive Sequential Recommendation* (SASRec)
