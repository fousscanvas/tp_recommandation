"""
Amazon Electronics Dataset Loader
==================================
Télécharge, nettoie et sérialise le dataset Amazon Electronics (5-core).

Source : Amazon Review Data 2018 — Ni et al., McAuley Lab (UCSD)
  - Reviews  : interactions user→item avec rating et timestamp
  - Metadata : nom, catégorie, prix de chaque item

Outputs dans data/processed/ :
  - items.json      : catalogue d'items {id, asin, title, category, price}
  - sequences.json  : {user_id: [item_id_1, item_id_2, ...]} triés par temps
  - stats.json      : statistiques du dataset

Usage :
  python amazon_loader.py            # télécharge + prépare tout
  python amazon_loader.py --local    # si tu as déjà les .gz en local
"""

import os
import json
import gzip
import argparse
import urllib.request
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR      = Path(__file__).parent
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR       = DATA_DIR / "raw"

# URLs officielles — Amazon Review Data 2018 (McAuley Lab, UCSD)
REVIEWS_URL  = "https://datarepos.nijianmo.github.io/amazon/categoryFiles/Electronics_5.json.gz"
METADATA_URL = "https://datarepos.nijianmo.github.io/amazon/categoryFiles/meta_Electronics.json.gz"

# Paramètres de filtrage
MIN_INTERACTIONS_PER_USER = 5    # séquences trop courtes → inutiles pour RL
MIN_INTERACTIONS_PER_ITEM = 5    # items trop rares → peu fiables
MAX_USERS                 = 5000  # on garde 5000 users pour ne pas exploser la RAM
MAX_ITEMS                 = 1000  # top-1000 items les plus fréquents

RATING_TO_CLICK_PROB = {
    1: 0.05,
    2: 0.15,
    3: 0.40,
    4: 0.75,
    5: 0.95,
}


# ---------------------------------------------------------------------------
# Téléchargement
# ---------------------------------------------------------------------------

def download_file(url: str, dest: Path, label: str = ""):
    """Télécharge un fichier avec barre de progression simple."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  [cache] {dest.name} déjà présent, skip.")
        return

    print(f"  Téléchargement {label} → {dest.name} ...")

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 / total_size)
            print(f"\r  {pct:.1f}%", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print(f"\r  OK — {dest.stat().st_size / 1024**2:.1f} Mo")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_gz_jsonl(path: Path) -> list[dict]:
    """Lit un fichier .json.gz ligne par ligne (format JSONL)."""
    records = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def parse_reviews(path: Path) -> list[dict]:
    """
    Extrait les champs utiles des reviews.
    Retourne une liste de {user_id, item_asin, rating, timestamp}.
    """
    raw = parse_gz_jsonl(path)
    reviews = []
    for r in raw:
        try:
            reviews.append({
                "user_id":   r["reviewerID"],
                "item_asin": r["asin"],
                "rating":    int(r["overall"]),
                "timestamp": int(r["unixReviewTime"]),
            })
        except (KeyError, ValueError):
            continue
    return reviews


def parse_metadata(path: Path) -> dict[str, dict]:
    """
    Extrait les métadonnées des items.
    Retourne un dict {asin: {title, category, price}}.
    """
    raw   = parse_gz_jsonl(path)
    items = {}
    for r in raw:
        asin = r.get("asin", "")
        if not asin:
            continue

        # Catégorie : on prend la plus spécifique (dernier élément de la liste)
        categories = r.get("categories", [[]])[0] if r.get("categories") else []
        category   = categories[-1] if categories else r.get("category", "Unknown")

        # Nettoyer la catégorie → mapper vers nos 5 catégories
        category = normalize_category(category)

        # Prix : peut être une string "$XX.XX"
        price_raw = r.get("price", "0")
        try:
            price = float(str(price_raw).replace("$", "").replace(",", "").strip())
        except ValueError:
            price = 0.0

        title = r.get("title", f"Item {asin}")
        # Nettoyer les titres trop longs
        if len(title) > 80:
            title = title[:77] + "..."

        items[asin] = {
            "asin":     asin,
            "title":    title,
            "category": category,
            "price":    round(price, 2),
        }
    return items


def normalize_category(raw: str) -> str:
    """Mappe les catégories Amazon vers nos 5 catégories internes."""
    raw_lower = raw.lower()
    mapping = {
        "tech":    ["computer", "laptop", "tablet", "phone", "camera",
                    "headphone", "speaker", "monitor", "keyboard", "mouse",
                    "hard drive", "usb", "cable", "charger", "battery",
                    "processor", "memory", "network", "router"],
        "fashion": ["clothing", "shoe", "jewelry", "watch", "bag",
                    "accessori", "apparel", "fashion"],
        "sport":   ["sport", "fitness", "outdoor", "exercise", "gym",
                    "camping", "hiking"],
        "home":    ["home", "kitchen", "furniture", "appliance", "garden",
                    "tool", "lighting", "office"],
        "books":   ["book", "kindle", "music", "movie", "game", "software"],
    }
    for cat, keywords in mapping.items():
        if any(kw in raw_lower for kw in keywords):
            return cat
    return "tech"   # défaut : tech (on est sur Electronics)


# ---------------------------------------------------------------------------
# Filtrage et construction des séquences
# ---------------------------------------------------------------------------

def build_sequences(
    reviews: list[dict],
    metadata: dict[str, dict],
) -> tuple[dict, dict]:
    """
    Construit :
      - item_catalog  : {item_id: {id, asin, title, category, price, click_probs}}
      - user_sequences: {user_id: [item_id_1, item_id_2, ...]}  (triés par timestamp)
    """
    print("  Filtrage des interactions...")

    # 1. Compter les interactions par item et par user
    item_counts = defaultdict(int)
    user_counts = defaultdict(int)
    for r in reviews:
        item_counts[r["item_asin"]] += 1
        user_counts[r["user_id"]]   += 1

    # 2. Garder seulement les items fréquents avec metadata
    valid_items = {
        asin for asin, cnt in item_counts.items()
        if cnt >= MIN_INTERACTIONS_PER_ITEM and asin in metadata
    }
    # Top-MAX_ITEMS par fréquence
    valid_items = set(
        sorted(valid_items, key=lambda a: item_counts[a], reverse=True)[:MAX_ITEMS]
    )

    # 3. Garder seulement les users avec assez d'interactions sur ces items
    user_reviews = defaultdict(list)
    for r in reviews:
        if r["item_asin"] in valid_items:
            user_reviews[r["user_id"]].append(r)

    valid_users = {
        uid for uid, rs in user_reviews.items()
        if len(rs) >= MIN_INTERACTIONS_PER_USER
    }
    # Top-MAX_USERS par nombre d'interactions
    valid_users = set(
        sorted(valid_users, key=lambda u: len(user_reviews[u]), reverse=True)[:MAX_USERS]
    )

    print(f"  Items valides : {len(valid_items)} | Users valides : {len(valid_users)}")

    # 4. Construire le catalogue avec IDs numériques
    asin_to_id = {asin: idx for idx, asin in enumerate(sorted(valid_items))}
    item_catalog = {}
    for asin, idx in asin_to_id.items():
        meta = metadata[asin]
        # Distribution de click_prob basée sur les ratings réels de cet item
        item_ratings = [r["rating"] for r in reviews if r["item_asin"] == asin]
        avg_rating   = sum(item_ratings) / len(item_ratings) if item_ratings else 3.0
        click_prob   = RATING_TO_CLICK_PROB.get(round(avg_rating), 0.4)

        item_catalog[idx] = {
            "id":         idx,
            "asin":       asin,
            "title":      meta["title"],
            "category":   meta["category"],
            "price":      meta["price"],
            "avg_rating": round(avg_rating, 2),
            "click_prob": click_prob,
            "n_reviews":  item_counts[asin],
        }

    # 5. Construire les séquences utilisateur
    user_sequences = {}
    for uid in valid_users:
        # Trier par timestamp, enlever les doublons consécutifs
        sorted_reviews = sorted(user_reviews[uid], key=lambda r: r["timestamp"])
        sequence = []
        prev_item = None
        for r in sorted_reviews:
            item_id = asin_to_id[r["item_asin"]]
            if item_id != prev_item:
                sequence.append(item_id)
                prev_item = item_id
        if len(sequence) >= MIN_INTERACTIONS_PER_USER:
            user_sequences[uid] = sequence

    return item_catalog, user_sequences


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def run(use_local: bool = False):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    reviews_path  = RAW_DIR / "Electronics_5.json.gz"
    metadata_path = RAW_DIR / "meta_Electronics.json.gz"

    # --- Téléchargement ---
    if not use_local:
        print("\n[1/4] Téléchargement des données...")
        download_file(REVIEWS_URL,  reviews_path,  "reviews")
        download_file(METADATA_URL, metadata_path, "metadata")
    else:
        print("\n[1/4] Mode local — skip téléchargement")
        if not reviews_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                f"Fichiers attendus dans {RAW_DIR} :\n"
                f"  - Electronics_5.json.gz\n"
                f"  - meta_Electronics.json.gz\n"
                "Télécharge-les depuis : https://nijianmo.github.io/amazon/"
            )

    # --- Parsing ---
    print("\n[2/4] Parsing des reviews...")
    reviews = parse_reviews(reviews_path)
    print(f"  {len(reviews):,} reviews chargées")

    print("\n[3/4] Parsing des métadonnées...")
    metadata = parse_metadata(metadata_path)
    print(f"  {len(metadata):,} items avec métadonnées")

    # --- Construction ---
    print("\n[4/4] Construction des séquences...")
    item_catalog, user_sequences = build_sequences(reviews, metadata)

    # --- Sauvegarde ---
    items_path     = PROCESSED_DIR / "items.json"
    sequences_path = PROCESSED_DIR / "sequences.json"
    stats_path     = PROCESSED_DIR / "stats.json"

    with open(items_path, "w", encoding="utf-8") as f:
        json.dump(item_catalog, f, ensure_ascii=False, indent=2)

    with open(sequences_path, "w", encoding="utf-8") as f:
        json.dump(user_sequences, f, ensure_ascii=False)

    # Stats
    seq_lengths = [len(s) for s in user_sequences.values()]
    categories  = defaultdict(int)
    for item in item_catalog.values():
        categories[item["category"]] += 1

    stats = {
        "n_items":       len(item_catalog),
        "n_users":       len(user_sequences),
        "n_interactions": sum(seq_lengths),
        "avg_seq_length": round(sum(seq_lengths) / len(seq_lengths), 2) if seq_lengths else 0,
        "min_seq_length": min(seq_lengths) if seq_lengths else 0,
        "max_seq_length": max(seq_lengths) if seq_lengths else 0,
        "categories":    dict(categories),
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print("\n✓ Données prêtes dans data/processed/")
    print(f"  Items     : {stats['n_items']}")
    print(f"  Users     : {stats['n_users']}")
    print(f"  Interactions : {stats['n_interactions']:,}")
    print(f"  Longueur moy. séquence : {stats['avg_seq_length']}")
    print(f"  Catégories : {stats['categories']}")

    return item_catalog, user_sequences


# ---------------------------------------------------------------------------
# Chargement depuis processed/
# ---------------------------------------------------------------------------

def load_processed() -> tuple[dict, dict]:
    """
    Charge les données déjà preprocessées.
    À utiliser dans le notebook et dans l'env.
    """
    items_path     = PROCESSED_DIR / "items.json"
    sequences_path = PROCESSED_DIR / "sequences.json"

    if not items_path.exists() or not sequences_path.exists():
        raise FileNotFoundError(
            "Données non trouvées. Lance d'abord :\n"
            "  python data/amazon_loader.py"
        )

    with open(items_path, encoding="utf-8") as f:
        # Les clés JSON sont des strings → convertir en int
        item_catalog = {int(k): v for k, v in json.load(f).items()}

    with open(sequences_path, encoding="utf-8") as f:
        user_sequences = json.load(f)

    return item_catalog, user_sequences


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true",
                        help="Ne pas télécharger, utiliser les .gz locaux")
    args = parser.parse_args()
    run(use_local=args.local)
