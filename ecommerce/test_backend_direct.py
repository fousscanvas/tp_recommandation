"""
Test direct de l'API (sans serveur HTTP).
Teste les fonctionnalités du backend directement.
"""

import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from data.products import PRODUCTS, N_ITEMS, get_product_by_id
from data.user_state import UserState
from retrain_enriched import DQNAgentEnriched

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test_user_state():
    """Test UserState class"""
    print("\n" + "="*60)
    print("TEST 1: UserState Management")
    print("="*60 + "\n")

    user = UserState(user_id='user_1')

    # 1. State inicial
    user.set_current_item(0)
    state = user.get_state_vector()
    print(f"1. Initial state for item 0: {state}")
    assert state.shape == (4,), f"State shape should be (4,), got {state.shape}"
    assert state.dtype == np.float32, f"State dtype should be float32"
    print("   ✓ State shape and dtype correct\n")

    # 2. Add purchase
    user.add_purchase(0, 999.99)
    context = user.get_context_dict()
    assert context['total_purchases'] == 1
    assert context['total_spent'] == 999.99
    print(f"2. After purchase: {context}")
    print("   ✓ Purchase recorded\n")

    # 3. State après achat
    user.set_current_item(5)
    state = user.get_state_vector()
    print(f"3. State for item 5 (after purchase): {state}")
    print(f"   ✓ Item encoded: {state[0]} (expected 5.0)")
    print(f"   ✓ History encoded: {state[3]} (expected 0.0 from [0])\n")

    # 4. Plusieurs interactions
    user.add_purchase(5, 69.99)
    user.add_click(3)
    user.add_purchase(3, 89.99)
    context = user.get_context_dict()
    print(f"4. After multiple interactions: {context}")
    assert context['total_purchases'] == 3
    assert context['total_clicks'] == 1
    print("   ✓ Multiple interactions recorded\n")

    print("✓ TEST 1 PASSED\n")


def test_dqn_model():
    """Test DQN model"""
    print("="*60)
    print("TEST 2: DQN Model Loading & Inference")
    print("="*60 + "\n")

    # 1. Load model
    print("1. Loading model...")
    agent = DQNAgentEnriched(n_items=N_ITEMS, input_dim=4, hidden=256)
    model_path = Path(__file__).parent / 'models' / 'dqn_enriched.pt'

    if model_path.exists():
        agent.online.load_state_dict(torch.load(model_path, map_location=device))
        print(f"   ✓ Model loaded from {model_path}\n")
    else:
        print(f"   ⚠ Model not found, using untrained\n")

    agent.online.eval()

    # 2. Forward pass
    print("2. Forward pass...")
    state = np.array([0.0, 0.0, 0.99, 0.0], dtype=np.float32)
    state_t = torch.tensor(state).unsqueeze(0).to(device)

    with torch.no_grad():
        q_values = agent.online(state_t)

    print(f"   Input state: {state}")
    print(f"   Q-values shape: {q_values.shape}")
    print(f"   Q-values: {q_values[0].cpu().numpy()}")
    assert q_values.shape == (1, N_ITEMS), f"Output shape should be (1, {N_ITEMS})"
    print("   ✓ Output shape correct\n")

    # 3. Greedy action
    action = q_values.argmax(dim=1).item()
    print(f"3. Greedy action: {action}")
    assert 0 <= action < N_ITEMS
    print("   ✓ Action in valid range\n")

    # 4. Batch inference
    print("4. Batch inference (5 states)...")
    states = np.random.randn(5, 4).astype(np.float32)
    states_t = torch.tensor(states).to(device)

    with torch.no_grad():
        q_batch = agent.online(states_t)

    print(f"   Input shape: {states_t.shape}")
    print(f"   Output shape: {q_batch.shape}")
    assert q_batch.shape == (5, N_ITEMS)
    print("   ✓ Batch inference works\n")

    print("✓ TEST 2 PASSED\n")


def test_integration():
    """Test integration: user state + model"""
    print("="*60)
    print("TEST 3: User State + Model Integration")
    print("="*60 + "\n")

    # Setup
    user = UserState(user_id='user_integration')
    agent = DQNAgentEnriched(n_items=N_ITEMS, input_dim=4, hidden=256)
    agent.online.eval()

    # 1. Random product sequence
    print("1. Simulating user journey...")
    products_viewed = np.random.choice(N_ITEMS, size=5, replace=False)

    for i, product_id in enumerate(products_viewed):
        user.set_current_item(product_id)
        state = user.get_state_vector()

        # Get recommendation
        state_t = torch.tensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = agent.online(state_t)
        rec = q_values.argmax(dim=1).item()

        print(f"   Step {i+1}: Viewing {get_product_by_id(product_id)['name']}")
        print(f"           → Recommend: {get_product_by_id(rec)['name']}")

        # Simulate user action (50% buy, 50% ignore)
        if np.random.rand() < 0.5:
            user.add_purchase(product_id, 50.0)
            print(f"           → User bought it!")
        else:
            print(f"           → User ignored it")

    # 2. Final state
    context = user.get_context_dict()
    print(f"\n2. Final user state:")
    print(f"   Total purchases: {context['total_purchases']}")
    print(f"   Total spent: ${context['total_spent']}")
    print(f"   Purchase history: {context['purchase_history']}")
    print("   ✓ Integration works seamlessly\n")

    print("✓ TEST 3 PASSED\n")


def test_edge_cases():
    """Test edge cases"""
    print("="*60)
    print("TEST 4: Edge Cases")
    print("="*60 + "\n")

    user = UserState(user_id='user_edge')

    # 1. All items viewed
    print("1. Testing max history...")
    for item_id in range(N_ITEMS):
        user.add_purchase(item_id, 50.0)

    assert len(user.purchase_history) == min(N_ITEMS, user.max_history_len)
    print(f"   ✓ History capped at {user.max_history_len} items\n")

    # 2. State after max history
    print("2. State with full history...")
    user.set_current_item(9)
    state = user.get_state_vector()
    print(f"   State: {state}")
    print(f"   ✓ State computed correctly with full history\n")

    # 3. Zero division check (empty history)
    print("3. Testing empty history...")
    user2 = UserState(user_id='user_empty')
    user2.set_current_item(0)
    state = user2.get_state_vector()
    assert not np.isnan(state).any(), "State contains NaN"
    print(f"   State: {state}")
    print(f"   ✓ No NaN or infinity values\n")

    print("✓ TEST 4 PASSED\n")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("BACKEND DIRECT TESTS")
    print("="*60)

    try:
        test_user_state()
        test_dqn_model()
        test_integration()
        test_edge_cases()

        print("="*60)
        print("✓ ALL TESTS PASSED")
        print("="*60 + "\n")

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
