# 🔍 Analyse complète — Résultats & Roadmap d'amélioration

## 📊 Tableau de synthèse

| Agent | Reward | CTR | Diversity | Type |
|---|---|---|---|---|
| Random | 1.263 | 12.7% | 0.15 | baseline |
| **QL Initial** | **1.637** | **14.4%** | **0.22** | tabular ✓ MEILLEUR |
| QL Optimisé | 1.443 | 13.6% | 0.229 | tabular |
| DQN | 1.307 | 11.1% | 0.203 | neural |
| GRU+DQN | 1.310 | 13.3% | **0.584** | sequential |

---

## 🚨 Problèmes critiques identifiés

### 1. Régression Q-Learning : 1.637 → 1.443 (-12%)
**"Optimisation" a DÉGRADÉ la performance**
- Grid search a trouvé des mauvais hyperparams
- Les meilleurs params du grid ne sont pas les meilleurs globalement
- Overfit sur le jeu de validation (400 ep)

### 2. DQN PIRE que QL : 1.307 vs 1.443 (-9%)
**Les réseaux de neurones n'apprennent pas mieux que les tables**
- Architecture DQN : `Embedding(50,64) → Linear(64,64) → ReLU → Linear(64,50)`
  - Trop simple
  - Pas de batch norm
  - Pas de résidus
- État = seulement le **dernier item** → zéro contexte
- Pas de normalisation des rewards

### 3. GRU+DQN NE S'AMÉLIORE PAS : 1.310 ≈ 1.307 (±0.2%)
**Ajouter du GRU n'a rien amélioré**
- `seq_len=4` : **TROP COURT** pour exploiter l'historique
- 4 items = 2-3 steps dans le jeu → quasi insignifiant
- Padding avec 0 = ambigu (mélange "début" et "jamais vu")
- GRU apprend mais sur peu de contexte

### 4. Diversité inversement corrélée au reward pour GRU
| Agent | Reward | Diversity | Ratio |
|---|---|---|---|
| QL Initial | 1.637 | 0.22 | **7.4** |
| GRU+DQN | 1.310 | **0.584** | **2.2** |

**GRU recommande des choses différentes mais pas intéressantes.**

- Bonus diversité (+0.2) crée un objectif conflictuel
- GRU choisit : plus de diversité = moins de reward immédiat
- Rational behavior : il vaut mieux 1000 recommandations médiocres & diverses que 100 excellentes redondantes

### 5. Exploration vs Exploitation mal calibrée
- `epsilon_min = 0.10` → 10% aléatoire même en fin d'entraînement
- `epsilon_decay = 0.997` → trop rapide

### 6. Target Network update trop "dur"
- Update tous les 50 steps = stable mais déconnecté
- Le réseau apprend mais la cible saute discontinuément

---

## 🔍 Racines de l'échec

### ROOT CAUSE 1: Apprentissage réseau chaotique

**1a. Warm-up insuffisant**
- Les 200-300 premiers episodes : réseau reçoit du bruit (exploration)
- Q-table : structure prédéfinie, bénéficie d'une "prior"
- Réseau : commence de zéro

**1b. Loss landscape trop chaotique**
- Rewards spars : +1 (click), +3 (buy), -0.5 (repeat), 0 (ignore)
- Réseau prédit 50 Q-valeurs en parallèle
- Q-table : apprentissage indépendant par cellule (N²=2500 cellules)

**1c. Reward shaping conflictuel**
- Bonus diversité +0.2 = noise supplémentaire pour le réseau
- Dualité : maximiser reward OU diversité = confus

**1d. Target network disconnect**
- Cible devient vieille, TD error grandit
- Gradient devient instable

### ROOT CAUSE 2: GRU ne capture pas l'historique

**Seq length trop court (4)**
- Historique de 4 items = contexte minimal
- Pour un user : [item1, item2, item3, item4] = seulement 3 transitions
- Impossible de voir des patterns longs

**Padding ambigu avec 0**
- `[0, 0, 1, 5]` = début session + items 1,5
- `[2, 3, 1, 5]` = fin session différente + même ending
- Réseau ne peut pas distinguer : "pas vu avant" vs "vu longtemps"

**GRU hidden state insuffisant**
- `hidden_dim=64` sur 4 items
- Compression trop forte

### ROOT CAUSE 3: Overfitting sur environment synthétique

- 50 items seulement
- 4 profiles fixes (tech/fashion/sport/random)
- Chaque profile converge sur 5-10 items "bons"
- Réseau apprend simplement ces items → suit la Q-table
- Puis bonus diversité contredit l'apprentissage

**Résultat : GRU + Bonus Diversité = maximise diversité, sacrifie reward**

---

## 📋 Roadmap d'amélioration (6 phases)

### PHASE 1 : Diagnostic DQN (1-2h)
**Vérifier si le problème est architectural ou hyperparams**

- [ ] A. Ablation : entraîner DQN **sans reward bonus diversité**
  - Si reward ↑ de 5% : bonus est le problème
- [ ] B. Monitor lors du training :
  - Loss curve (lissé 50ep)
  - Q-value ranges (min/max/mean)
  - TD error distribution
- [ ] C. Convergence check :
  - Fait-on loss → 0 ? Ou plateau à 0.5+ ?
- [ ] D. Sanity check :
  - Q-values similaires à Q-table ?
  - Ou totalement aléatoires ?

### PHASE 2 : Architecture DQN (2-3h)
**Augmenter capacité et stabilité**

- [ ] A. **Bigger network** :
  ```
  QNetwork v2:
    - embed_dim : 64 → 128
    - layer 1   : 64 → 256
    - layer 2   : 64 → 256
    - Ajouter batch norm après embedding
  ```
- [ ] B. **Reward normalization** :
  ```python
  reward_normalized = reward / 3.0  # max possible reward
  # Range: [-0.166, 1.0] au lieu de [-0.5, 3.0]
  ```
- [ ] C. **Double DQN** :
  ```python
  # Séparer : selection de l'action vs évaluation
  # q_target = r + γ * Q_target(s', argmax_a Q_online(s', a))
  # au lieu de : r + γ * max_a Q_target(s', a)
  ```
- [ ] D. **Dueling DQN** :
  ```python
  # Deux branches : V(s) + A(s, a)
  # Q(s, a) = V(s) + (A(s,a) - mean(A(s, .)))
  # Meilleur pour actions similaires
  ```

### PHASE 3 : GRU — Fixing Sequence (2-3h)
**Vraiment exploiter l'historique**

- [ ] A. **Augmenter seq_len** :
  - `seq_len : 4 → 16` (ou même 32)
  - 16 items = 15 transitions = vrai pattern
- [ ] B. **Remplacer padding 0** :
  ```python
  # Token START spécial
  START_TOKEN = n_items + 1  # index 51
  embedding(n_items + 1)  # +1 pour le token
  seq = [START_TOKEN, START_TOKEN, ..., START_TOKEN]  # au lieu de [0,0,...]
  ```
  - Donne du sens au "vrai rien"
- [ ] C. **Ajouter Attention** :
  ```python
  # GRU + MultiHeadAttention
  # Laisser le réseau choisir quels items importer
  embeddings → GRU → [h1, h2, ..., h16]
            → Attention(h, h, h)  # self-attention
            → weighted sum → context vector
  ```
- [ ] D. **Bidirectional GRU** :
  ```python
  # Encoder dans les deux sens
  gru_fwd = GRU(..., reverse=False)
  gru_bwd = GRU(..., reverse=True)
  h_final = [h_fwd_last; h_bwd_last]  # concatenate
  ```

### PHASE 4 : Hyperparams Tuning (1-2h)
**Stabiliser l'apprentissage**

- [ ] A. **Learning rate scheduling** :
  ```python
  lr = 1e-3
  # Option 1: Decay
  # if ep % 1000 == 0: lr *= 0.5  # → 5e-4, 2.5e-4, ...
  
  # Option 2: Cosine annealing
  # lr(t) = lr_min + 0.5*(lr_max - lr_min)*(1 + cos(π*t/T))
  ```
- [ ] B. **Epsilon decay** :
  - `epsilon_min : 0.10 → 0.01` (moins d'aléatoire)
  - `epsilon_decay : 0.997 → 0.9995` (plus progressif)
- [ ] C. **Prioritized Experience Replay (PER)** :
  ```python
  # Replay les transitions avec high TD error plus souvent
  # p(i) ∝ (|TD_error_i| + ε)^α
  # Improve convergence sur hard transitions
  ```
- [ ] D. **Soft Target Update** :
  ```python
  # Au lieu de : target = online tous les 50 steps
  # Faire : target ← τ*online + (1-τ)*target  (chaque step)
  # τ = 0.001 (très petit)
  # Plus stable, moins de discontinuité
  ```

### PHASE 5 : Reward Redesign (30min)
**Enlever la source de conflits**

- [ ] A. **Supprimer bonus diversité** (-0.2)
  - Cause : GRU sacrifice reward pour diversité
- [ ] B. **Structure claire** :
  ```
  +3.0 : achat
  +1.0 : clic
  -0.5 : repeat
   0.0 : ignore
  ```
- [ ] C. **Bonus optionnel : catégories rares** :
  ```python
  # Au lieu de "toute nouvelle catégorie"
  # Bonus que si catégorie rare dans catalog
  cat_popularity = count_in_catalog[action_cat]
  if cat_popularity < 0.1 * mean_popularity:
      reward += 0.1  # bonus léger pour rare
  ```

### PHASE 6 : Validation & Comparison (2-3h)
**Évaluer vraiment les améliorations**

- [ ] A. Entraîner sur **5000+ episodes** (vs 3000)
- [ ] B. **Early stopping** :
  - Validation reward tous les 500ep
  - Arrêter si plateau 100 episodes
- [ ] C. **Learning curves** :
  - Tracer : DQN v2 vs QL initial vs GRU v2
  - Vérifier convergence
- [ ] D. **A/B test** :
  - Random vs QL initial vs DQN v2 vs GRU+Attn
  - Stats : mean, std, p-value

---

## ⚡ Quick Wins (Priorité haute)

| Change | Impact | Effort | Expected Gain |
|---|---|---|---|
| Remove diversity bonus | Architecture alignment | 5 min | **+5-10%** |
| Increase seq_len 4→16 | More context | 5 min | **+3-7%** |
| Dueling DQN | Better Q-estimates | 20 min | **+2-5%** |
| Soft target update | Stability | 15 min | **+1-3%** |
| Reduce epsilon_min 0.1→0.02 | Better exploitation | 2 min | **+2-4%** |
| Reward normalization | Gradient stability | 10 min | **+1-2%** |

**Priority order :**
1. Remove bonus (5min) → test reward
2. Seq_len 16 (5min) → test GRU
3. Dueling DQN (20min) → structural improvement
4. Soft update (15min) → stability
5. Reward norm (10min) → gradient health

---

## 🎯 Expected outcome après Phase 1-5

**Optimistic** (tous les fixes appliqués) :
- DQN v2 : 1.307 → **1.50+** (+15%)
- GRU v2 : 1.310 → **1.55+** (+18%)
  - Avec seq_len=16 + Attention + no bonus + soft update

**Realistic** (quick wins seulement) :
- DQN : 1.307 → **1.40** (+7%)
- GRU : 1.310 → **1.45** (+11%)

**Conservative** (une seule change) :
- Remove bonus : **+5%**
- Seq_len 16 : **+3%**

---

## 📝 Takeaway

**Le problème n'est PAS "j'ai besoin de plus de neurones"**

C'est une **conjonction de** :
1. Bonus diversité conflictuel
2. Seq_len trop court (4 items)
3. Architecture réseau trop simple
4. Hyperparams conservateurs (epsilon_min=0.1)
5. Target update dur (tous les 50 steps)

**Fix #1 (remove bonus) tout seul pourrait donner +5% immédiatement.**

Le GRU n'est PAS mauvais — il ne reçoit juste pas assez de signal / contexte pour briller.
