# Quick Start Guide

Get the Smart E-Commerce platform up and running in 5 minutes.

## Prerequisites

```bash
# Python 3.8+
python3 --version

# Install dependencies
pip3 install torch numpy flask flask-cors
```

## 3-Step Setup

### Step 1: Train Models (10 min)

```bash
cd ecommerce
python3 retrain_enriched.py
```

**What happens:**
- Q-Learning trains for 500 episodes
- DQN trains for 300 episodes
- Model saved to `models/dqn_enriched.pt`

**Output:**
```
Device: cpu
============================================================
Training Q-Learning (enriched) - 500 episodes
============================================================
Episode  50/500 | Avg Reward: 5.08 | ε: 0.7783
...
Final reward (last 50 eps): 4.66

============================================================
Training DQN (enriched, GELU) - 300 episodes
============================================================
...
Final reward (last 50 eps): 4.88

✓ DQN model saved to models/dqn_enriched.pt
```

### Step 2: Start Backend API (Terminal 2)

```bash
cd ecommerce
python3 backend/app.py
```

**Output:**
```
Device: cpu
============================================================
E-COMMERCE BACKEND API
============================================================
Running on http://localhost:5000
```

### Step 3: Open Frontend (Browser)

```bash
open frontend/index.html
```

Or manually open:
```
file:///path/to/ecommerce/frontend/index.html
```

## What You'll See

1. **Product Grid** - 10 products with emojis
2. **Currently Viewing** - Shows selected product details
3. **AI Recommendation** - DQN recommendation with confidence score
4. **Purchase History** - Track your purchases
5. **Real-time Stats** - Total spent, items purchased

## Try These Actions

### Browse Products
- Click any product in the grid
- Current product details appear on the right

### Get Recommendations
- View a product → Automatic recommendation appears
- Green "💳 Buy" button → Records purchase
- Blue "👁️ More Info" button → View that product

### Track Stats
- See purchases update in real-time
- Total spent increases as you buy
- History shows all purchased items

## Example User Journey

1. **Start**: System shows random product
2. **Browse**: Click 5-10 products
3. **Recommendations**: DQN suggests related items
4. **Purchase**: Click "Buy" on recommendations
5. **See Results**: Purchase history and stats update

## Testing (Optional)

### Run All Tests
```bash
python3 test_backend_direct.py    # Backend unit tests
python3 test_complete_flow.py     # Integration tests
```

Expected output:
```
✓ ALL TESTS PASSED
```

## Troubleshooting

### Issue: "Model not found"
```
⚠ Model not found at models/dqn_enriched.pt, using untrained model
```
**Fix**: Run `python3 retrain_enriched.py` first

### Issue: "API Connection Error"
**Fix**: Make sure backend is running: `python3 backend/app.py`

### Issue: Browser shows "Connecting..."
1. Check backend terminal for errors
2. Check browser console (F12) for error messages
3. Ensure API_URL is correct: `http://127.0.0.1:5000/api`

### Issue: Very Slow Inference
**Normal**: First few inferences might be slow while model loads
**After**: Should be <1ms per recommendation

## Key Files to Know

| File | Purpose |
|------|---------|
| `retrain_enriched.py` | Train the DQN model |
| `backend/app.py` | Start the API server |
| `frontend/index.html` | Open this in browser |
| `data/products.py` | 10-product catalog |
| `data/user_state.py` | User state management |
| `models/dqn_enriched.pt` | Trained model (auto-generated) |

## Next Steps

### To Improve Results
Change in `retrain_enriched.py`:
```python
# Line 343: Change from 500 to 5000
train_ql(n_episodes=5000)

# Line 344: Change from 300 to 5000
train_dqn(n_episodes=5000)
```

Then retrain: `python3 retrain_enriched.py`

Expected: Better recommendations after ~2-3 hours

### To Deploy to Production
See `PRODUCTION_GUIDE.md` for:
- Database setup
- Production server (Gunicorn)
- Domain & SSL
- Auto-scaling

## Architecture at a Glance

```
Frontend (HTML/JS)
       ↓
   Flask API
       ↓
    DQN Model
       ↓
  User State & Products
```

## Key Metrics

- **Products**: 10
- **Categories**: 4 (electronics, clothing, home, books)
- **Model Inference**: 0.27ms
- **Throughput**: 162k samples/sec
- **Average Confidence**: 73.4%

## That's It! 🎉

You now have a working AI e-commerce recommendation system!

- ✅ Recommendation model running
- ✅ Backend API responding
- ✅ Frontend displaying products
- ✅ Real-time tracking

**Next**: Try purchasing some items and see how recommendations improve!

---

For more details, see:
- `README.md` - Full documentation
- `IMPLEMENTATION_SUMMARY.md` - Architecture details
- `PRODUCTION_GUIDE.md` - Deployment instructions
