"""
Backend API pour le micro e-commerce avec recommandations RL.

Endpoints:
  GET  /api/products          - List tous les produits
  GET  /api/products/<id>     - Détails d'un produit
  POST /api/users/<user_id>   - Créer/initialiser un user
  GET  /api/users/<user_id>   - Récupérer état d'un user
  POST /api/recommend         - Obtenir recommandation (state + model)
  POST /api/purchase          - Enregistrer achat (feedback)
  POST /api/click             - Enregistrer click (feedback)
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import torch
import sys
from pathlib import Path

# Imports locaux
sys.path.insert(0, str(Path(__file__).parent.parent))
from data.products import PRODUCTS, N_ITEMS, get_product_by_id
from data.user_state import UserState
from retrain_enriched import DQNAgentEnriched

# ============================================================================
# Configuration
# ============================================================================

app = Flask(__name__)
CORS(app)  # Enable CORS pour frontend

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Store pour les users (en mémoire, persistent pendant la session)
users_db = {}

# Charger le model DQN entrainé
print("Loading DQN model...")
dqn_agent = DQNAgentEnriched(n_items=N_ITEMS, input_dim=4, hidden=256)
model_path = Path(__file__).parent.parent / 'models' / 'dqn_enriched.pt'

if model_path.exists():
    dqn_agent.online.load_state_dict(torch.load(model_path, map_location=device))
    print(f"✓ Model loaded from {model_path}")
else:
    print(f"⚠ Model not found at {model_path}, using untrained model")

dqn_agent.online.eval()


# ============================================================================
# Utility Functions
# ============================================================================

def get_or_create_user(user_id):
    """
    Récupère ou crée un user.
    """
    if user_id not in users_db:
        users_db[user_id] = UserState(user_id=user_id)
    return users_db[user_id]


def get_recommendation(user_state, model='dqn'):
    """
    Obtient une recommandation du model RL.

    Args:
        user_state: UserState object
        model: 'dqn' (défaut)

    Returns:
        item_id (int)
    """
    state_vec = user_state.get_state_vector()

    if model == 'dqn':
        # Inference avec DQN
        state_t = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = dqn_agent.online(state_t)
        return int(q_values.argmax(dim=1).item())
    else:
        # Fallback: random
        return np.random.randint(0, N_ITEMS)


def product_to_dict(product):
    """Convertit un produit en dict JSON-safe"""
    return {
        'id': product['id'],
        'name': product['name'],
        'category': product['category'],
        'price': product['price'],
        'image': product['image'],
        'description': product['description']
    }


# ============================================================================
# Routes - Products
# ============================================================================

@app.route('/api/products', methods=['GET'])
def list_products():
    """
    Retourne tous les produits.

    Response:
        {
            "products": [
                {"id": 0, "name": "Laptop Pro", "category": "electronics", ...},
                ...
            ]
        }
    """
    return jsonify({
        'products': [product_to_dict(p) for p in PRODUCTS]
    })


@app.route('/api/products/<int:product_id>', methods=['GET'])
def get_product(product_id):
    """
    Retourne les détails d'un produit.

    Response:
        {"product": {...}}  ou
        {"error": "Not found"}, 404
    """
    product = get_product_by_id(product_id)
    if product is None:
        return jsonify({'error': f'Product {product_id} not found'}), 404

    return jsonify({'product': product_to_dict(product)})


# ============================================================================
# Routes - Users
# ============================================================================

@app.route('/api/users/<user_id>', methods=['POST'])
def create_user(user_id):
    """
    Crée/initialise un user avec un item de départ aléatoire.

    Response:
        {
            "user_id": "user123",
            "state": [item_id, category, price, history],
            "current_product": {...}
        }
    """
    user = get_or_create_user(user_id)

    # Item aléatoire de départ
    initial_item = np.random.randint(0, N_ITEMS)
    user.set_current_item(initial_item)

    state_vec = user.get_state_vector().tolist()
    current_product = get_product_by_id(initial_item)

    return jsonify({
        'user_id': user_id,
        'state': state_vec,
        'current_product': product_to_dict(current_product)
    })


@app.route('/api/users/<user_id>', methods=['GET'])
def get_user(user_id):
    """
    Récupère l'état d'un user.

    Response:
        {
            "user_id": "user123",
            "state": [...],
            "purchase_history": [0, 5, 3],
            "total_purchases": 3,
            "total_spent": 1159.97
        }
    """
    user = get_or_create_user(user_id)
    context = user.get_context_dict()
    context['state'] = user.get_state_vector().tolist()

    return jsonify(context)


# ============================================================================
# Routes - Recommendations & Feedback
# ============================================================================

@app.route('/api/recommend', methods=['POST'])
def recommend():
    """
    Obtient une recommandation pour un user basée sur le DQN.

    Request body:
        {
            "user_id": "user123",
            "current_item_id": 5
        }

    Response:
        {
            "recommended_item_id": 7,
            "product": {...},
            "confidence": 0.85  (max Q-value / max possible)
        }
    """
    data = request.get_json()
    user_id = data.get('user_id')
    current_item_id = data.get('current_item_id', 0)

    if user_id is None:
        return jsonify({'error': 'user_id required'}), 400

    # Get/create user
    user = get_or_create_user(user_id)
    user.set_current_item(int(current_item_id))

    # Get recommendation
    state_vec = user.get_state_vector()
    recommended_item_id = get_recommendation(user, model='dqn')
    recommended_product = get_product_by_id(recommended_item_id)

    # Compute confidence (normalized Q-value)
    state_t = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        q_values = dqn_agent.online(state_t)
    max_q = q_values.max().item()
    confidence = max(0.0, min(1.0, (max_q + 5) / 10))  # Normalize to [0,1]

    return jsonify({
        'recommended_item_id': recommended_item_id,
        'product': product_to_dict(recommended_product),
        'confidence': round(confidence, 2)
    })


@app.route('/api/purchase', methods=['POST'])
def purchase():
    """
    Enregistre un achat (feedback positif).

    Request body:
        {
            "user_id": "user123",
            "product_id": 5
        }

    Response:
        {
            "success": true,
            "reward": 3.0,
            "user_state": {...}
        }
    """
    data = request.get_json()
    user_id = data.get('user_id')
    product_id = data.get('product_id')

    if user_id is None or product_id is None:
        return jsonify({'error': 'user_id and product_id required'}), 400

    user = get_or_create_user(user_id)
    product = get_product_by_id(int(product_id))

    if product is None:
        return jsonify({'error': f'Product {product_id} not found'}), 404

    # Record purchase
    user.add_purchase(int(product_id), product['price'])

    return jsonify({
        'success': True,
        'reward': 3.0,
        'product_name': product['name'],
        'user_context': user.get_context_dict()
    })


@app.route('/api/click', methods=['POST'])
def click():
    """
    Enregistre un click (feedback neutre).

    Request body:
        {
            "user_id": "user123",
            "product_id": 5
        }

    Response:
        {
            "success": true,
            "reward": 1.0
        }
    """
    data = request.get_json()
    user_id = data.get('user_id')
    product_id = data.get('product_id')

    if user_id is None or product_id is None:
        return jsonify({'error': 'user_id and product_id required'}), 400

    user = get_or_create_user(user_id)
    user.add_click(int(product_id))

    return jsonify({
        'success': True,
        'reward': 1.0,
        'user_context': user.get_context_dict()
    })


# ============================================================================
# Health Check
# ============================================================================

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'model': 'dqn',
        'device': str(device),
        'n_items': N_ITEMS
    })


# ============================================================================
# Error Handlers
# ============================================================================

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error', 'message': str(e)}), 500


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("E-COMMERCE BACKEND API")
    print("="*60)
    print(f"Device: {device}")
    print(f"Products: {N_ITEMS}")
    print("Running on http://localhost:5000")
    print("="*60 + "\n")

    app.run(debug=False, host='127.0.0.1', port=5000)
