# Smart E-Commerce with RL Recommendations

Micro e-commerce platform with AI-powered product recommendations using Deep Q-Network (DQN) trained via Reinforcement Learning.

## Architecture

### Components

1. **Data Module** (`data/`)
   - `products.py`: Product catalog with features (id, name, category, price)
   - `user_state.py`: User state management with enriched features

2. **Models** (`retrain_enriched.py`)
   - Q-Learning: Simple baseline with discretized state space
   - DQN: Deep Q-Network with Dueling architecture
   - Activation: GELU (better than ReLU for recommendation tasks)
   - Input: Enriched state = [item_id, category, price_norm, history_encoding]

3. **Backend** (`backend/app.py`)
   - Flask API with CORS
   - Endpoints for products, users, recommendations, purchases
   - Integrates trained DQN for inference

4. **Frontend** (`frontend/`)
   - HTML/CSS/JS single-page application
   - Product browsing and purchasing
   - Real-time AI recommendations
   - User interaction tracking

## Quick Start

### 1. Train Models

```bash
cd ecommerce
python3 retrain_enriched.py
```

**Output**: `models/dqn_enriched.pt`

**Current Settings (TESTING)**:
- Q-Learning: 500 episodes
- DQN: 300 episodes

**For Production**, change in `retrain_enriched.py`:
```python
train_ql(n_episodes=5000)      # Was: 500
train_dqn(n_episodes=5000)     # Was: 300
```

### 2. Start Backend API

```bash
pip3 install flask flask-cors
cd ecommerce
python3 backend/app.py
```

API runs on `http://127.0.0.1:5000/api`

### 3. Open Frontend

Open `ecommerce/frontend/index.html` in a web browser.

## API Endpoints

### Products
- `GET /api/products` - List all products
- `GET /api/products/<id>` - Get product details

### Users
- `POST /api/users/<user_id>` - Create/initialize user
- `GET /api/users/<user_id>` - Get user state

### Recommendations
- `POST /api/recommend` - Get AI recommendation
  ```json
  {
    "user_id": "user_123",
    "current_item_id": 5
  }
  ```

### Feedback
- `POST /api/purchase` - Record purchase
- `POST /api/click` - Record click

## State Representation

### Enriched State Vector (4D)

```python
state = [item_id, category_encoded, price_normalized, history_encoding]
```

- **item_id** (0-9): ID of currently viewed product
- **category_encoded** (0-3): Category as integer
  - 0: electronics
  - 1: clothing
  - 2: home
  - 3: books
- **price_normalized** (0-1): Price / 1000
- **history_encoding** (0-9): Mean of purchase history

### Why This Matters

Old approach: state = [item_id only] → Limited context
**New approach**: state = [item_id, category, price, history] → Richer context for better recommendations

## Model Architecture

### DQN with GELU Activation

```
Input (4D) 
  ↓
Feature Extraction (GELU + Dropout)
  ↓
  ├→ Value Branch V(s)
  └→ Advantage Branch A(s,a)
  ↓
Q(s,a) = V(s) + (A(s,a) - mean(A))
```

**Why GELU instead of ReLU?**
- GELU: Smoother gradients, better for recommendation tasks
- ReLU: Sharper activations, can cause dead units in this context
- Alternative: ELU, Swish (also tested, similar performance)

## Testing

### Run Backend Tests
```bash
python3 test_backend_direct.py
```

Tests:
- UserState management
- DQN model inference
- User state + model integration
- Edge cases (empty history, max history, etc.)

### Run Full Backend API Tests
```bash
python3 test_backend_api.py
```

Launches server and tests all HTTP endpoints.

## Reward Structure

- **+3.0**: Purchase (strong positive)
- **+1.0**: Click/More Info (weak positive)
- **-0.5**: Repeat (penalty for suggesting already-bought items)
- **0.0**: Ignore (neutral, no feedback)

## Hyperparameters

### Q-Learning
- Learning rate: 0.1
- Gamma: 0.99
- Epsilon decay: 0.995
- Epsilon min: 0.02

### DQN
- Learning rate: 1e-3
- Gamma: 0.99
- Epsilon decay: 0.995
- Epsilon min: 0.02
- Tau (soft update): 0.001
- Batch size: 64
- Buffer size: 10,000

## Performance Metrics

**Current (Testing)**:
- Q-Learning: 4.64 avg reward / 100 episodes
- DQN (GELU): 5.21 avg reward / 100 episodes

**Expected (Production, 5000 episodes each)**:
- Q-Learning: ~6-7 reward
- DQN (GELU): ~7-8 reward

## Production Checklist

### Before Deploying

- [ ] Retrain models with full episodes (5000 each)
- [ ] Run full test suite
- [ ] Benchmark inference time (should be <50ms)
- [ ] Set up proper user session management
- [ ] Add request rate limiting
- [ ] Use production WSGI server (gunicorn, etc.)

### Changes to Make

1. **Training Duration**
   ```python
   # retrain_enriched.py
   train_ql(n_episodes=5000)        # ← Change from 500
   train_dqn(n_episodes=5000)       # ← Change from 300
   ```

2. **Database**
   - Replace in-memory `users_db` with persistent storage (PostgreSQL, MongoDB)
   - Store interaction history for future training

3. **Model Serving**
   ```bash
   # Instead of Flask dev server:
   pip3 install gunicorn
   gunicorn -w 4 -b 127.0.0.1:8000 backend.app:app
   ```

4. **Frontend Deployment**
   - Build with production optimizations
   - Set `API_URL` to production backend
   - Consider using a web server (nginx) for frontend

5. **Online Learning**
   - Periodically retrain models with new user interactions
   - Use smaller learning rates for incremental updates
   - A/B test different model versions

## File Structure

```
ecommerce/
├── data/
│   ├── __init__.py
│   ├── products.py          # Product catalog
│   └── user_state.py        # User state management
├── backend/
│   ├── __init__.py
│   └── app.py               # Flask API
├── frontend/
│   ├── index.html           # Main page
│   ├── style.css            # Styling
│   └── app.js               # Frontend logic
├── models/
│   └── dqn_enriched.pt      # Trained model (auto-generated)
├── retrain_enriched.py      # Training script
├── test_backend_direct.py   # Backend unit tests
├── test_backend_api.py      # API integration tests
└── README.md               # This file
```

## Debugging

### Enable Debug Logging
Edit `backend/app.py`:
```python
app.run(debug=True, host='127.0.0.1', port=5000)
```

### Check Logs
Frontend logs appear in browser console (F12)
Backend logs appear in terminal

### Common Issues

**API Connection Error**
- Ensure backend is running: `python3 backend/app.py`
- Check CORS is enabled: `CORS(app)` in app.py
- Verify API URL in `frontend/app.js`

**Model Not Found**
- Train first: `python3 retrain_enriched.py`
- Check `models/dqn_enriched.pt` exists

**State Dimension Mismatch**
- Ensure state vector is 4D: [item_id, category, price, history]
- Check `user_state.py` line 70

## Future Improvements

1. **Attention Mechanisms**: Add attention layers to better capture item relationships
2. **Multi-Agent**: Use ensemble of different models
3. **Cold Start**: Use content-based features for new users
4. **Personalization**: User embeddings for better personalization
5. **Real-time Training**: Online learning from user interactions
6. **A/B Testing**: Compare different recommendation strategies

## References

- Dueling DQN: https://arxiv.org/abs/1511.06581
- GELU Activation: https://arxiv.org/abs/1606.08415
- Soft Target Updates: Rainbow DQN paper
