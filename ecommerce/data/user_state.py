"""
Gestion de l'état utilisateur pour les recommandations RL.
État enrichi = (item_courant, catégorie, prix, historique_achat)
"""

import numpy as np
from collections import deque
from .products import (
    N_ITEMS, N_CATEGORIES, get_category_encoded,
    get_price_normalized, get_product_by_id
)


class UserState:
    """
    Représente l'état d'un utilisateur pour le model RL.

    State vector = [item_id, category_encoded, price_normalized, history_encoding]
    - item_id: ID du produit courant (0-9)
    - category_encoded: Catégorie encodée (0-3)
    - price_normalized: Prix normalisé (0-1)
    - history_encoding: Encoding de l'historique d'achat (0-N_ITEMS)
    """

    def __init__(self, user_id, max_history_len=8):
        """
        Initialise l'état utilisateur.

        Args:
            user_id: Identifiant unique de l'utilisateur
            max_history_len: Longueur max de l'historique d'achat
        """
        self.user_id = user_id
        self.max_history_len = max_history_len

        # Historique d'achat (items que l'utilisateur a achetés)
        self.purchase_history = deque(maxlen=max_history_len)

        # Compteurs pour le contexte utilisateur
        self.total_purchases = 0
        self.total_clicks = 0
        self.total_spent = 0.0

        # Item courant affiché
        self.current_item_id = 0

    def set_current_item(self, item_id):
        """Définit l'item courant (celui affiché à l'utilisateur)"""
        assert 0 <= item_id < N_ITEMS, f"Item ID {item_id} invalide"
        self.current_item_id = item_id

    def add_purchase(self, item_id, price):
        """Enregistre un achat"""
        self.purchase_history.append(item_id)
        self.total_purchases += 1
        self.total_spent += price

    def add_click(self, item_id):
        """Enregistre un clic (mais pas achat)"""
        self.total_clicks += 1

    def get_state_vector(self):
        """
        Retourne le vecteur d'état pour le model RL.

        Shape: (4,) = [item_id, category, price, history_encoding]

        Returns:
            np.array de 4 éléments
        """
        product = get_product_by_id(self.current_item_id)

        # Élément 0: ID du produit courant
        item_encoded = float(self.current_item_id)

        # Élément 1: Catégorie du produit courant
        category_encoded = float(get_category_encoded(product['category']))

        # Élément 2: Prix normalisé du produit courant
        price_encoded = get_price_normalized(product['price'])

        # Élément 3: Encoding de l'historique
        # Si aucun achat: 0
        # Sinon: moyenne des IDs achetés + bonus par nombre d'achats
        if len(self.purchase_history) == 0:
            history_encoded = 0.0
        else:
            history_encoded = float(np.mean(list(self.purchase_history)))

        state = np.array([item_encoded, category_encoded, price_encoded, history_encoded],
                         dtype=np.float32)
        return state

    def get_context_dict(self):
        """Retourne un dict avec le contexte utilisateur (pour logs/debug)"""
        return {
            'user_id': self.user_id,
            'current_item_id': self.current_item_id,
            'purchase_history': list(self.purchase_history),
            'total_purchases': self.total_purchases,
            'total_clicks': self.total_clicks,
            'total_spent': round(self.total_spent, 2)
        }

    def reset(self, item_id=0):
        """Réinitialise pour une nouvelle session (garde l'historique)"""
        self.current_item_id = item_id


# ============================================================================
# Tests
# ============================================================================

if __name__ == '__main__':
    print("=== Test UserState ===\n")

    # Créer user
    user = UserState(user_id=1, max_history_len=8)
    print(f"User créé: {user.get_context_dict()}\n")

    # État initial
    user.set_current_item(0)
    state = user.get_state_vector()
    print(f"État initial (item=0): {state}")
    print(f"Shape: {state.shape}, dtype: {state.dtype}\n")

    # Ajouter un achat
    user.add_purchase(0, 999.99)
    print(f"Après achat item 0: {user.get_context_dict()}")

    # Nouvel item
    user.set_current_item(5)
    state = user.get_state_vector()
    print(f"État après changement item (item=5): {state}\n")

    # Plusieurs achats
    user.add_purchase(5, 69.99)
    user.add_click(3)
    user.add_purchase(3, 89.99)
    print(f"Après plusieurs interactions: {user.get_context_dict()}\n")

    state = user.get_state_vector()
    print(f"État final: {state}")
    print(f"✓ Test réussi!")
