"""
Métriques d'évaluation pour les systèmes de recommandation RL.

Deux familles de métriques :
  - RL        : reward cumulée, TD error, stabilité
  - Reco      : CTR, purchase rate, diversité, couverture, hit rate

Usage :
    from utils.metrics import EpisodeMetrics, compute_summary, evaluate_agent
"""

import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict


# ---------------------------------------------------------------------------
# Collecte par épisode
# ---------------------------------------------------------------------------

@dataclass
class EpisodeMetrics:
    """Statistiques collectées pour un seul épisode."""
    total_reward:   float = 0.0
    n_steps:        int   = 0
    n_clicks:       int   = 0
    n_buys:         int   = 0
    n_ignores:      int   = 0
    n_repeats:      int   = 0
    recommended:    list  = field(default_factory=list)   # items recommandés
    td_errors:      list  = field(default_factory=list)   # erreurs TD (si dispo)

    # --- métriques dérivées ---

    @property
    def ctr(self) -> float:
        """Click-Through Rate = clics / recommandations totales."""
        return self.n_clicks / self.n_steps if self.n_steps > 0 else 0.0

    @property
    def purchase_rate(self) -> float:
        """Taux d'achat = achats / recommandations totales."""
        return self.n_buys / self.n_steps if self.n_steps > 0 else 0.0

    @property
    def engagement_rate(self) -> float:
        """(clics + achats) / recommandations totales."""
        return (self.n_clicks + self.n_buys) / self.n_steps if self.n_steps > 0 else 0.0

    @property
    def diversity(self) -> float:
        """Items uniques recommandés / total recommandations (0=répétitif, 1=varié)."""
        if self.n_steps == 0:
            return 0.0
        return len(set(self.recommended)) / self.n_steps

    @property
    def repeat_rate(self) -> float:
        """Taux de recommandations déjà vues (pénalité -0.5)."""
        return self.n_repeats / self.n_steps if self.n_steps > 0 else 0.0

    @property
    def mean_td_error(self) -> float:
        return float(np.mean(self.td_errors)) if self.td_errors else 0.0


# ---------------------------------------------------------------------------
# Agrégation sur N épisodes
# ---------------------------------------------------------------------------

def compute_summary(episodes: list[EpisodeMetrics]) -> dict:
    """
    Calcule les statistiques agrégées sur une liste d'épisodes.
    Retourne un dict prêt pour affichage ou comparaison.
    """
    def _mean(values): return float(np.mean(values)) if values else 0.0
    def _std(values):  return float(np.std(values))  if values else 0.0

    rewards   = [e.total_reward   for e in episodes]
    ctrs      = [e.ctr            for e in episodes]
    prs       = [e.purchase_rate  for e in episodes]
    eng       = [e.engagement_rate for e in episodes]
    divs      = [e.diversity      for e in episodes]
    reps      = [e.repeat_rate    for e in episodes]
    td_errs   = [e.mean_td_error  for e in episodes]

    # Couverture globale : % d'items du catalogue recommandés au moins une fois
    all_recommended = set()
    for e in episodes:
        all_recommended.update(e.recommended)

    # Hit rate : % d'épisodes avec au moins un clic ou achat
    hit_rate = sum(1 for e in episodes if e.n_clicks + e.n_buys > 0) / len(episodes)

    return {
        "n_episodes":      len(episodes),
        # Reward
        "reward_mean":     _mean(rewards),
        "reward_std":      _std(rewards),
        "reward_min":      float(np.min(rewards)),
        "reward_max":      float(np.max(rewards)),
        # Engagement
        "ctr_mean":        _mean(ctrs),
        "purchase_rate":   _mean(prs),
        "engagement_rate": _mean(eng),
        "hit_rate":        hit_rate,
        # Qualité
        "diversity_mean":  _mean(divs),
        "repeat_rate":     _mean(reps),
        # Couverture catalogue
        "coverage":        len(all_recommended),
        # Stabilité RL
        "td_error_mean":   _mean(td_errs),
    }


# ---------------------------------------------------------------------------
# Évaluation d'un agent (mode pure exploitation, epsilon=0)
# ---------------------------------------------------------------------------

def evaluate_agent(agent, env_factory, n_episodes: int = 200, seed: int = 99) -> list[EpisodeMetrics]:
    """
    Évalue un agent en mode exploitation pure (epsilon = 0).
    Compatible avec QLearningAgent et DQNAgent (interface .act()).
    """
    # Sauvegarder et forcer epsilon=0
    original_epsilon = getattr(agent, 'epsilon', 0.0)
    if hasattr(agent, 'epsilon'):
        agent.epsilon = 0.0

    env     = env_factory(seed=seed)
    results = []

    for _ in range(n_episodes):
        state = env.reset()
        ep    = EpisodeMetrics()
        done  = False

        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)

            ep.total_reward += reward
            ep.n_steps      += 1
            ep.recommended.append(action)

            event = info.get("event", "")
            if event == "click":
                ep.n_clicks  += 1
            elif event == "buy":
                ep.n_buys    += 1
            elif event == "ignore":
                ep.n_ignores += 1
            elif event == "repeat":
                ep.n_repeats += 1

            state = next_state

        results.append(ep)

    # Restaurer epsilon
    if hasattr(agent, 'epsilon'):
        agent.epsilon = original_epsilon

    return results


# ---------------------------------------------------------------------------
# Agent aléatoire (baseline)
# ---------------------------------------------------------------------------

class RandomAgent:
    """Agent qui recommande uniformément au hasard. Sert de baseline."""
    def __init__(self, n_actions: int, seed: int = 42):
        self.n_actions = n_actions
        self.epsilon   = 0.0
        self.rng       = np.random.default_rng(seed)

    def act(self, state) -> int:
        return int(self.rng.integers(0, self.n_actions))


# ---------------------------------------------------------------------------
# Collecte des métriques pendant l'entraînement (rolling)
# ---------------------------------------------------------------------------

class TrainingTracker:
    """
    Collecte les métriques épisode par épisode pendant l'entraînement.
    Permet de tracer les courbes d'évolution.
    """
    def __init__(self):
        self.episodes: list[EpisodeMetrics] = []

    def log(self, ep: EpisodeMetrics):
        self.episodes.append(ep)

    def rolling(self, metric: str, window: int = 50) -> np.ndarray:
        """Moyenne mobile d'une métrique."""
        import pandas as pd
        values = [getattr(e, metric) for e in self.episodes]
        return pd.Series(values).rolling(window, min_periods=1).mean().values

    def get(self, metric: str) -> list:
        return [getattr(e, metric) for e in self.episodes]
