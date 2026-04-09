"""
Test script pour l'API backend.
Lance le serveur et teste les endpoints.
"""

import requests
import json
import time
from multiprocessing import Process
import sys
from pathlib import Path

# Import le backend
sys.path.insert(0, str(Path(__file__).parent))
from backend.app import app

BASE_URL = 'http://127.0.0.1:5000/api'


def run_server():
    """Lance le serveur Flask"""
    app.run(debug=False, host='127.0.0.1', port=5000, use_reloader=False)


def test_api():
    """Teste tous les endpoints"""
    print("\n" + "="*60)
    print("TESTING E-COMMERCE API")
    print("="*60 + "\n")

    # Attendre que le serveur démarre
    time.sleep(3)

    try:
        # 1. Health check
        print("1. Health check...")
        resp = requests.get(f'{BASE_URL}/health')
        assert resp.status_code == 200, f"Health check failed: {resp.status_code}"
        health = resp.json()
        print(f"   ✓ Server running on {health['device']}")
        print(f"   ✓ Model: {health['model']}")
        print(f"   ✓ N_items: {health['n_items']}\n")

        # 2. List products
        print("2. List products...")
        resp = requests.get(f'{BASE_URL}/products')
        assert resp.status_code == 200
        data = resp.json()
        n_products = len(data['products'])
        print(f"   ✓ Got {n_products} products")
        print(f"   ✓ Example: {data['products'][0]['name']}\n")

        # 3. Get single product
        print("3. Get product #0...")
        resp = requests.get(f'{BASE_URL}/products/0')
        assert resp.status_code == 200
        product = resp.json()['product']
        print(f"   ✓ {product['name']} (${product['price']})\n")

        # 4. Create user
        print("4. Create user...")
        resp = requests.post(f'{BASE_URL}/users/test_user_1', json={})
        assert resp.status_code == 200
        user_data = resp.json()
        print(f"   ✓ User created: {user_data['user_id']}")
        print(f"   ✓ Starting product: {user_data['current_product']['name']}")
        print(f"   ✓ State vector: {user_data['state']}\n")

        # 5. Get user state
        print("5. Get user state...")
        resp = requests.get(f'{BASE_URL}/users/test_user_1')
        assert resp.status_code == 200
        user_state = resp.json()
        print(f"   ✓ User: {user_state['user_id']}")
        print(f"   ✓ Purchases: {user_state['total_purchases']}")
        print(f"   ✓ Total spent: ${user_state['total_spent']}\n")

        # 6. Get recommendation
        print("6. Get recommendation...")
        resp = requests.post(f'{BASE_URL}/recommend',
                           json={'user_id': 'test_user_1', 'current_item_id': 3})
        assert resp.status_code == 200
        rec = resp.json()
        print(f"   ✓ Recommended: {rec['product']['name']}")
        print(f"   ✓ Confidence: {rec['confidence']}\n")

        # 7. Record purchase
        print("7. Record purchase...")
        resp = requests.post(f'{BASE_URL}/purchase',
                           json={'user_id': 'test_user_1', 'product_id': 3})
        assert resp.status_code == 200
        purchase = resp.json()
        print(f"   ✓ Purchase recorded")
        print(f"   ✓ Reward: {purchase['reward']}")
        print(f"   ✓ Total purchases now: {purchase['user_context']['total_purchases']}\n")

        # 8. Record click
        print("8. Record click...")
        resp = requests.post(f'{BASE_URL}/click',
                           json={'user_id': 'test_user_1', 'product_id': 5})
        assert resp.status_code == 200
        click = resp.json()
        print(f"   ✓ Click recorded")
        print(f"   ✓ Reward: {click['reward']}\n")

        # 9. Get updated user state
        print("9. Get updated user state...")
        resp = requests.get(f'{BASE_URL}/users/test_user_1')
        assert resp.status_code == 200
        user_state = resp.json()
        print(f"   ✓ Total purchases: {user_state['total_purchases']}")
        print(f"   ✓ Total clicks: {user_state['total_clicks']}")
        print(f"   ✓ Total spent: ${user_state['total_spent']}")
        print(f"   ✓ Purchase history: {user_state['purchase_history']}\n")

        # 10. Error cases
        print("10. Error handling...")
        resp = requests.get(f'{BASE_URL}/products/999')
        assert resp.status_code == 404, "Should 404 for invalid product"
        print(f"   ✓ 404 on invalid product\n")

        resp = requests.post(f'{BASE_URL}/recommend', json={})
        assert resp.status_code == 400, "Should 400 for missing user_id"
        print(f"   ✓ 400 on missing user_id\n")

        print("="*60)
        print("✓ ALL TESTS PASSED")
        print("="*60 + "\n")

    except AssertionError as e:
        print(f"✗ Test failed: {e}\n")
        return False
    except Exception as e:
        print(f"✗ Error: {e}\n")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == '__main__':
    print("Starting server in background...\n")

    # Démarrer le serveur dans un processus séparé
    server_process = Process(target=run_server, daemon=True)
    server_process.start()

    # Tester l'API
    success = test_api()

    # Arrêter le serveur
    server_process.terminate()
    server_process.join(timeout=2)

    sys.exit(0 if success else 1)
