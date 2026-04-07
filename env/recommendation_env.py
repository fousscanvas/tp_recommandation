"""
Environnement de recommandation — MDP formalisé.

Supporte deux modes :
  - Synthétique : catalogue généré (50 items) — rapide pour les tests
  - Amazon      : catalogue réel Electronics — après avoir lancé amazon_loader.py

Interface :
  env = RecommendationEnv()                          # mode synthétique
  env = RecommendationEnv.from_amazon()              # mode Amazon

MDP :
  - État   : dernier item vu par l'utilisateur (index entier)
  - Action : item recommandé par l'agent (index entier)
  - Reward : +1 clic, +3 achat, 0 ignoré, -0.5 item déjà vu
  - Done   : fin de session après `max_steps` interactions
"""

import sys
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Catalogue synthétique (fallback / tests rapides)
# ---------------------------------------------------------------------------

CATEGORIES = ["tech", "fashion", "sport", "home", "books"]


def build_synthetic_catalog(n_items: int = 50) -> dict[int, dict]:
    rng = np.random.default_rng(42)
    catalog = {}
    for i in range(n_items):
        category = CATEGORIES[i % len(CATEGORIES)]
        catalog[i] = {
            "id":         i,
            "asin":       f"SYN{i:04d}",
            "title":      f"{category.capitalize()} item {i}",
            "category":   category,
            "price":      round(float(rng.uniform(5, 500)), 2),
            "avg_rating": round(float(rng.uniform(3, 5)), 2),
            "click_prob": round(float(rng.beta(2, 3)), 3),
            "n_reviews":  int(rng.integers(10, 500)),
        }
    return catalog


# ---------------------------------------------------------------------------
# Profils utilisateur
# (valables pour les deux modes — synthétique et Amazon)
# ---------------------------------------------------------------------------

USER_PROFILES = {
    "tech":    {"tech": 0.85, "fashion": 0.10, "sport": 0.20, "home": 0.25, "books": 0.30},
    "fashion": {"tech": 0.10, "fashion": 0.90, "sport": 0.30, "home": 0.40, "books": 0.20},
    "sport":   {"tech": 0.20, "fashion": 0.20, "sport": 0.85, "home": 0.30, "books": 0.20},
    "random":  {"tech": 0.40, "fashion": 0.40, "sport": 0.40, "home": 0.40, "books": 0.40},
}


# ---------------------------------------------------------------------------
# Matrice de similarité inter-items
# ---------------------------------------------------------------------------

def build_similarity_matrix(catalog: dict[int, dict]) -> np.ndarray:
    n = len(catalog)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i][j] = 1.0
            elif catalog[i]["category"] == catalog[j]["category"]:
                matrix[i][j] = 0.6
            else:
                matrix[i][j] = 0.1
    return matrix


# ---------------------------------------------------------------------------
# Environnement principal
# ---------------------------------------------------------------------------

class RecommendationEnv:
    """
    Environnement de recommandation séquentielle.

    Paramètres
    ----------
    catalog         : dict {item_id: {title, category, price, click_prob, ...}}
    user_sequences  : dict {user_id: [item_id_1, item_id_2, ...]}
                      Si None, les épisodes sont entièrement simulés.
    max_steps       : durée max d'une session
    profile         : profil utilisateur fixe (None = aléatoire à chaque reset)
    seed            : reproductibilité
    """

    def __init__(
        self,
        catalog: dict[int, dict] | None = None,
        user_sequences: dict | None = None,
        max_steps: int = 10,
        profile: str | None = None,
        seed: int = 42,
    ):
        self.catalog        = catalog if catalog is not None else build_synthetic_catalog()
        self.user_sequences = user_sequences  # None → simulation pure
        self.max_steps      = max_steps
        self.fixed_profile  = profile
        self.rng            = np.random.default_rng(seed)

        self.n_items = len(self.catalog)
        self.similarity = build_similarity_matrix(self.catalog)

        # Espaces discrets
        self.observation_space_n = self.n_items
        self.action_space_n      = self.n_items

        # État interne
        self.current_state = None
        self.profile       = None
        self.step_count    = 0
        self.seen_items    = set()
        self.fatigue       = 0.0

        # Si on a des séquences réelles, indexer les users disponibles
        self._user_ids = list(user_sequences.keys()) if user_sequences else None

    # ------------------------------------------------------------------
    @classmethod
    def from_amazon(cls, max_steps: int = 10, profile: str | None = None, seed: int = 42):
        """
        Construit l'env à partir du dataset Amazon preprocessé.
        Lance d'abord : python data/amazon_loader.py
        """
        # Import relatif robuste
        loader_path = Path(__file__).parent.parent / "data"
        if str(loader_path) not in sys.path:
            sys.path.insert(0, str(loader_path))

        from amazon_loader import load_processed
        catalog, user_sequences = load_processed()
        print(f"[Amazon] {len(catalog)} items, {len(user_sequences)} users chargés.")
        return cls(catalog=catalog, user_sequences=user_sequences,
                   max_steps=max_steps, profile=profile, seed=seed)

    # ------------------------------------------------------------------
    def reset(self) -> int:
        """
        Démarre un nouvel épisode.
        - Avec séquences réelles : commence sur le premier item d'un user aléatoire
        - Sans séquences        : item de départ aléatoire
        """
        self.step_count = 0
        self.seen_items = set()
        self.fatigue    = 0.0

        # Profil pour cet épisode
        if self.fixed_profile:
            self.profile = self.fixed_profile
        else:
            self.profile = str(self.rng.choice(list(USER_PROFILES.keys())))

        if self._user_ids is not None:
            # Mode Amazon : démarre sur le 1er item de la séquence d'un user réel
            uid         = self._user_ids[int(self.rng.integers(0, len(self._user_ids)))]
            start_item  = self.user_sequences[uid][0]
        else:
            start_item = int(self.rng.integers(0, self.n_items))

        self.current_state = start_item
        self.seen_items.add(start_item)
        return self.current_state

    # ------------------------------------------------------------------
    def step(self, action: int) -> tuple[int, float, bool, dict]:
        """
        L'agent recommande `action` (index item).
        Retourne (next_state, reward, done, info).

        Logique de reward
        -----------------
        - Item déjà vu          : -0.5
        - Item ignoré           :  0.0
        - Clic                  : +1.0
        - Achat (sous-événement): +3.0  (prob = 0.2 × affinité × click_prob item)
        - Fatigue               : réduit la prob de clic si même catégorie répétée
        """
        assert 0 <= action < self.n_items, f"Action invalide : {action}"

        self.step_count += 1
        info = {
            "profile":         self.profile,
            "action_category": self.catalog[action]["category"],
        }

        # --- Pénalité item déjà vu ---
        if action in self.seen_items:
            reward = -0.5
            done   = self.step_count >= self.max_steps
            return self.current_state, reward, done, {**info, "event": "repeat"}

        # --- Pertinence ---
        affinity      = USER_PROFILES[self.profile][self.catalog[action]["category"]]
        context_boost = self.similarity[self.current_state][action] * 0.2
        # click_prob intrinsèque de l'item (calibré depuis les ratings Amazon)
        item_click    = self.catalog[action].get("click_prob", 0.5)
        relevance     = min(1.0, (affinity + context_boost) * item_click)

        # --- Fatigue ---
        current_cat = self.catalog[self.current_state]["category"]
        action_cat  = self.catalog[action]["category"]
        if action_cat == current_cat:
            self.fatigue = min(1.0, self.fatigue + 0.15)
        else:
            self.fatigue = max(0.0, self.fatigue - 0.05)

        click_prob = relevance * (1 - self.fatigue * 0.5)
        clicked    = self.rng.random() < click_prob

        if clicked:
            self.seen_items.add(action)
            self.current_state = action
            buy_prob = 0.2 * relevance
            bought   = self.rng.random() < buy_prob
            reward   = 3.0 if bought else 1.0
            info["event"] = "buy" if bought else "click"
        else:
            reward        = 0.0
            info["event"] = "ignore"

        done = self.step_count >= self.max_steps
        return self.current_state, reward, done, info

    # ------------------------------------------------------------------
    def get_item(self, item_id: int) -> dict:
        return self.catalog[item_id]

    def get_categories(self) -> list[str]:
        return sorted({v["category"] for v in self.catalog.values()})

    def __repr__(self) -> str:
        mode = "Amazon" if self._user_ids else "Synthetic"
        return (f"RecommendationEnv(mode={mode}, n_items={self.n_items}, "
                f"max_steps={self.max_steps}, profile={self.profile})")
