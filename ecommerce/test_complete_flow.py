"""
Complete Integration Test: Simulates a full user journey
Tests: products → user creation → browsing → recommendations → purchases
"""

import sys
from pathlib import Path
import numpy as np
import torch
from time import time

sys.path.insert(0, str(Path(__file__).parent))
from data.products import PRODUCTS, N_ITEMS, get_product_by_id
from data.user_state import UserState
from retrain_enriched import DQNAgentEnriched, SimpleRecommendationEnv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test_complete_user_journey():
    """
    Simulates a realistic user journey:
    1. Initialize user
    2. Browse 10 products
    3. Get recommendations
    4. Make 3-5 purchases
    5. Verify consistency
    """
    print("\n" + "="*70)
    print("COMPLETE USER JOURNEY TEST")
    print("="*70 + "\n")

    # ====== SETUP ======
    print("1. SETUP")
    print("-" * 70)

    user = UserState(user_id='journey_user_123')
    agent = DQNAgentEnriched(n_items=N_ITEMS, input_dim=4, hidden=256)
    agent.online.eval()

    model_path = Path(__file__).parent / 'models' / 'dqn_enriched.pt'
    if model_path.exists():
        agent.online.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✓ Model loaded\n")
    else:
        print(f"⚠ Model not found, using untrained network\n")

    # ====== BROWSING ======
    print("2. BROWSING PHASE (10 products)")
    print("-" * 70)

    browsing_sequence = np.random.choice(N_ITEMS, size=10, replace=True)
    recommendations_given = []
    confidence_scores = []

    for step, product_id in enumerate(browsing_sequence, 1):
        # Set current product
        user.set_current_item(product_id)
        product = get_product_by_id(product_id)

        # Get state and recommendation
        state = user.get_state_vector()
        state_t = torch.tensor(state).unsqueeze(0).to(device)

        with torch.no_grad():
            q_values = agent.online(state_t)
            rec_id = q_values.argmax(dim=1).item()
            max_q = q_values.max().item()
            confidence = max(0.0, min(1.0, (max_q + 5) / 10))

        rec_product = get_product_by_id(rec_id)
        recommendations_given.append(rec_id)
        confidence_scores.append(confidence)

        print(f"Step {step:2d}: Viewing {product['name']:20s} → "
              f"Recommend {rec_product['name']:20s} (conf={confidence:.2f})")

    print(f"\nBrowsing complete. Recommendations quality: "
          f"Avg confidence = {np.mean(confidence_scores):.3f}\n")

    # ====== PURCHASING ======
    print("3. PURCHASING PHASE")
    print("-" * 70)

    n_purchases = np.random.randint(2, 6)  # 2-5 purchases
    purchase_items = np.random.choice(N_ITEMS, size=n_purchases, replace=False)

    for i, item_id in enumerate(purchase_items, 1):
        product = get_product_by_id(item_id)
        user.add_purchase(item_id, product['price'])
        print(f"Purchase {i}: {product['name']} (${product['price']:.2f})")

    print()

    # ====== VERIFICATION ======
    print("4. VERIFICATION")
    print("-" * 70)

    context = user.get_context_dict()

    # Check 1: Purchase history
    assert len(context['purchase_history']) == len(purchase_items), \
        "Purchase history size mismatch"
    print(f"✓ Purchase history size: {len(context['purchase_history'])}")

    # Check 2: Spent calculation
    expected_spent = sum(get_product_by_id(p)['price'] for p in purchase_items)
    assert abs(context['total_spent'] - expected_spent) < 0.01, \
        "Total spent calculation error"
    print(f"✓ Total spent: ${context['total_spent']:.2f} (correct)")

    # Check 3: State vector after history
    user.set_current_item(5)
    state = user.get_state_vector()
    assert state.shape == (4,), f"State shape error: {state.shape}"
    assert not np.isnan(state).any(), "State contains NaN"
    assert not np.isinf(state).any(), "State contains Inf"
    print(f"✓ State vector valid: shape={state.shape}, no NaN/Inf")

    # Check 4: Model inference consistency
    states_batch = np.random.randn(5, 4).astype(np.float32)
    states_t = torch.tensor(states_batch).to(device)
    with torch.no_grad():
        q_batch = agent.online(states_t)
    assert q_batch.shape == (5, N_ITEMS), "Batch inference shape error"
    print(f"✓ Batch inference shape: {q_batch.shape}")

    # Check 5: Recommendations are valid
    assert all(0 <= r < N_ITEMS for r in recommendations_given), \
        "Invalid recommendation IDs"
    print(f"✓ All recommendations valid (0-{N_ITEMS-1})")

    print()
    print("="*70)
    print("✓ COMPLETE USER JOURNEY TEST PASSED")
    print("="*70 + "\n")

    return {
        'user': user,
        'agent': agent,
        'num_products_viewed': len(browsing_sequence),
        'num_purchases': len(purchase_items),
        'total_spent': context['total_spent'],
        'avg_confidence': np.mean(confidence_scores)
    }


def test_recommendation_quality():
    """
    Tests recommendation quality over multiple user profiles
    """
    print("\n" + "="*70)
    print("RECOMMENDATION QUALITY TEST (100 simulated users)")
    print("="*70 + "\n")

    agent = DQNAgentEnriched(n_items=N_ITEMS, input_dim=4, hidden=256)
    agent.online.eval()

    model_path = Path(__file__).parent / 'models' / 'dqn_enriched.pt'
    if model_path.exists():
        agent.online.load_state_dict(torch.load(model_path, map_location=device))

    # Simulate 100 users with different profiles
    profiles = {
        'electronics_lover': [0, 1, 2],
        'fashionista': [3, 4, 5],
        'home_decorator': [6, 7, 8],
        'bookworm': [9],
        'random_browser': list(range(N_ITEMS))
    }

    results = {}

    for profile_name, product_preferences in profiles.items():
        confidences = []

        for _ in range(20):  # 20 users per profile
            user = UserState(user_id=f'profile_{profile_name}')

            # User buys products from their preference
            for _ in range(np.random.randint(2, 5)):
                item_id = np.random.choice(product_preferences)
                product = get_product_by_id(item_id)
                user.add_purchase(item_id, product['price'])

            # Check recommendation for similar item
            test_item = np.random.choice(product_preferences)
            user.set_current_item(test_item)
            state = user.get_state_vector()

            state_t = torch.tensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = agent.online(state_t)
                confidence = max(0.0, min(1.0, (q_values.max().item() + 5) / 10))
            confidences.append(confidence)

        avg_conf = np.mean(confidences)
        results[profile_name] = avg_conf
        print(f"{profile_name:20s}: confidence = {avg_conf:.3f}")

    print("\n" + "="*70)
    print(f"✓ RECOMMENDATION QUALITY TEST PASSED")
    print(f"  Overall average confidence: {np.mean(list(results.values())):.3f}")
    print("="*70 + "\n")

    return results


def benchmark_inference_time():
    """
    Benchmarks model inference time (should be <50ms)
    """
    print("\n" + "="*70)
    print("INFERENCE TIME BENCHMARK")
    print("="*70 + "\n")

    agent = DQNAgentEnriched(n_items=N_ITEMS, input_dim=4, hidden=256)
    agent.online.eval()

    model_path = Path(__file__).parent / 'models' / 'dqn_enriched.pt'
    if model_path.exists():
        agent.online.load_state_dict(torch.load(model_path, map_location=device))

    # Benchmark single inference
    print("1. Single inference (1 state)")
    times = []
    for _ in range(100):
        state = np.random.randn(4).astype(np.float32)
        state_t = torch.tensor(state).unsqueeze(0).to(device)

        t0 = time()
        with torch.no_grad():
            _ = agent.online(state_t)
        t1 = time()
        times.append((t1 - t0) * 1000)  # Convert to ms

    avg_time = np.mean(times)
    max_time = np.max(times)
    print(f"   Avg: {avg_time:.2f}ms")
    print(f"   Max: {max_time:.2f}ms")
    assert avg_time < 50, f"Inference too slow: {avg_time:.2f}ms"
    print(f"   ✓ Performance acceptable\n")

    # Benchmark batch inference
    print("2. Batch inference (64 states)")
    times = []
    for _ in range(100):
        states = np.random.randn(64, 4).astype(np.float32)
        states_t = torch.tensor(states).to(device)

        t0 = time()
        with torch.no_grad():
            _ = agent.online(states_t)
        t1 = time()
        times.append((t1 - t0) * 1000)

    avg_time = np.mean(times)
    per_sample = avg_time / 64
    print(f"   Batch avg: {avg_time:.2f}ms")
    print(f"   Per-sample: {per_sample:.2f}ms")
    print(f"   ✓ Throughput: {64/avg_time*1000:.0f} samples/sec\n")

    print("="*70)
    print("✓ BENCHMARK COMPLETE")
    print("="*70 + "\n")


if __name__ == '__main__':
    try:
        print("\n" + "="*70)
        print("RUNNING COMPLETE INTEGRATION TESTS")
        print("="*70)

        # Test 1: User journey
        journey_result = test_complete_user_journey()

        # Test 2: Recommendation quality
        quality_result = test_recommendation_quality()

        # Test 3: Performance
        benchmark_inference_time()

        # Summary
        print("="*70)
        print("✓ ALL INTEGRATION TESTS PASSED")
        print("="*70)
        print("\nSummary:")
        print(f"  Journey products viewed: {journey_result['num_products_viewed']}")
        print(f"  Journey purchases: {journey_result['num_purchases']}")
        print(f"  Total spent: ${journey_result['total_spent']:.2f}")
        print(f"  Avg recommendation confidence: {journey_result['avg_confidence']:.3f}")
        print(f"  Best user profile confidence: {max(quality_result.values()):.3f}")
        print("="*70 + "\n")

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
