# Production Deployment Guide

Complete guide for moving the Smart E-Commerce platform from testing to production.

## Current State (Testing/Development)

✓ Model trained: 300-500 episodes each (fast iterations)
✓ Backend API: Flask development server
✓ Frontend: Pure HTML/CSS/JS (no build process)
✓ Storage: In-memory only (resets on restart)
✓ Testing: All unit & integration tests passing

## Production Checklist

### 1. Model Training (2-3 hours)

**Current**: 500 episodes Q-Learning, 300 episodes DQN

**Change to Production**:
```bash
cd ecommerce
```

Edit `retrain_enriched.py`, find these lines:

```python
# Line ~360: Change from 500 to 5000
train_ql(n_episodes=5000)  # Was: 500

# Line ~361: Change from 300 to 5000  
train_dqn(n_episodes=5000)  # Was: 300
```

Then retrain:
```bash
python3 retrain_enriched.py
```

Expected time: ~2-3 hours (laptop), <30 min (GPU)
Output: `models/dqn_enriched.pt` (updated with better training)

### 2. Database Setup (Persistent Storage)

**Current**: In-memory dictionary
```python
users_db = {}  # Resets on restart
```

**Production**: Use PostgreSQL or MongoDB

#### Option A: PostgreSQL (Recommended)

Install:
```bash
pip3 install psycopg2-binary sqlalchemy
```

Create `backend/models.py`:
```python
from sqlalchemy import Column, Integer, String, Float, JSON, DateTime
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(String, primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    purchase_history = Column(JSON)
    total_purchases = Column(Integer)
    total_spent = Column(Float)
    total_clicks = Column(Integer)
```

Update `backend/app.py`:
```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "postgresql://user:password@localhost/ecommerce_db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

# Use SessionLocal() instead of users_db dict
```

#### Option B: MongoDB

Install:
```bash
pip3 install pymongo
```

Update `backend/app.py`:
```python
from pymongo import MongoClient

client = MongoClient('mongodb://localhost:27017/')
db = client['ecommerce_db']
users_collection = db['users']
```

### 3. Production Server Setup

**Current**: Flask development server (single-threaded, slow)
```bash
python3 backend/app.py
```

**Production**: Use Gunicorn + Nginx

Install:
```bash
pip3 install gunicorn
```

Start Gunicorn (4 workers, port 8000):
```bash
cd ecommerce
gunicorn -w 4 -b 127.0.0.1:8000 backend.app:app
```

**Nginx Configuration** (`/etc/nginx/sites-available/ecommerce`):
```nginx
server {
    listen 80;
    server_name yourdomain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_buffering off;
    }

    location /frontend {
        alias /path/to/ecommerce/frontend;
    }
}
```

Enable:
```bash
sudo ln -s /etc/nginx/sites-available/ecommerce /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### 4. Frontend Build & Optimization

**Current**: Plain HTML/CSS/JS files

**Production Build**:
```bash
npm init -y
npm install --save-dev webpack webpack-cli
npm install --save-dev terser-webpack-plugin
```

Create `webpack.config.js`:
```javascript
module.exports = {
  mode: 'production',
  entry: './frontend/app.js',
  output: {
    filename: 'bundle.min.js',
    path: __dirname + '/frontend/dist'
  },
  optimization: {
    minimize: true
  }
};
```

Build:
```bash
npm run build
```

Update `frontend/index.html`:
```html
<!-- Change from -->
<script src="app.js"></script>

<!-- To -->
<script src="dist/bundle.min.js"></script>
```

### 5. API Configuration

Update `backend/app.py`:

```python
# Development
API_CORS_ORIGINS = ["http://localhost:3000"]
DEBUG = True

# Production
API_CORS_ORIGINS = ["https://yourdomain.com"]
DEBUG = False

CORS(app, origins=API_CORS_ORIGINS)
```

Update `frontend/app.js`:
```javascript
// Development
const API_URL = 'http://127.0.0.1:5000/api';

// Production
const API_URL = 'https://yourdomain.com/api';
```

### 6. Model Serving Options

#### Option A: TorchServe (Recommended for ML)

```bash
pip3 install torchserve torch-model-archiver torch-workflow-archiver

# Archive model
torch-model-archiver \
  --model-name dqn_recommender \
  --version 1.0 \
  --model-file models/dqn_model.py \
  --serialized-file models/dqn_enriched.pt \
  --handler models/handler.py

# Start server
torchserve --start --model-store model_store --ncs
```

#### Option B: TensorFlow Serving

Convert PyTorch to TensorFlow:
```python
import onnx
import onnx_tf.backend as onnx_backend

# Convert PyTorch → ONNX → TensorFlow
```

### 7. Monitoring & Logging

Install:
```bash
pip3 install python-json-logger prometheus-client
```

Add to `backend/app.py`:
```python
import logging
from pythonjsonlogger import jsonlogger

logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter()
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)
```

### 8. Auto-Scaling

Deploy with Docker:
```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY ecommerce /app

RUN pip install -r requirements.txt

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8000", "backend.app:app"]
```

Build & run:
```bash
docker build -t ecommerce:latest .
docker run -p 8000:8000 ecommerce:latest
```

### 9. Security Hardening

#### Enable HTTPS
```python
# backend/app.py
from flask_talisman import Talisman

Talisman(app, 
    force_https=True,
    strict_transport_security=True,
    strict_transport_security_max_age=31536000
)
```

#### Rate Limiting
```bash
pip3 install flask-limiter
```

```python
from flask_limiter import Limiter

limiter = Limiter(app)

@app.route('/api/recommend', methods=['POST'])
@limiter.limit("100/minute")
def recommend():
    ...
```

#### Input Validation
```python
from pydantic import BaseModel, validator

class PurchaseRequest(BaseModel):
    user_id: str
    product_id: int
    
    @validator('product_id')
    def product_id_valid(cls, v):
        if not 0 <= v < N_ITEMS:
            raise ValueError('Invalid product ID')
        return v
```

### 10. Online Learning Setup

Periodically retrain on new data:
```python
# backend/scheduled_retraining.py
from apscheduler.schedulers.background import BackgroundScheduler
import retrain_enriched

scheduler = BackgroundScheduler()

@scheduler.scheduled_job('interval', days=7)  # Weekly retraining
def retrain_models():
    print("Starting scheduled retraining...")
    agent, _ = retrain_enriched.train_dqn(n_episodes=1000)
    torch.save(agent.online.state_dict(), 'models/dqn_enriched_updated.pt')
    print("Retraining complete")

scheduler.start()
```

## Deployment Scenarios

### Scenario 1: Single Server (Cheap)
- Nginx (frontend + API proxy)
- Gunicorn (4 workers)
- SQLite or PostgreSQL
- Startup: ~10 min
- Cost: ~$5-20/month (DigitalOcean, Heroku)

### Scenario 2: Microservices (Scalable)
- Frontend: AWS S3 + CloudFront
- API: ECS + Load Balancer
- Model: SageMaker
- Database: RDS (PostgreSQL)
- Startup: ~1-2 hours setup
- Cost: ~$50-200/month

### Scenario 3: Serverless (Elastic)
- Frontend: Vercel / Netlify
- API: AWS Lambda
- Model: SageMaker Endpoint or Lambda
- Database: DynamoDB
- Cost: ~$10-30/month (with usage limits)

## Performance Targets

### Before Optimization
- Inference: 0.27ms (single)
- Throughput: 162k samples/sec
- Model size: ~2MB
- API response: <100ms

### After Optimization
- Inference: <0.1ms (quantized)
- Throughput: >500k samples/sec
- Model size: <500KB (pruned)
- API response: <50ms (with caching)

### Optimization Steps

1. **Model Quantization**
```python
# Convert FP32 → INT8
quantized_model = torch.quantization.quantize_dynamic(
    agent.online,
    {torch.nn.Linear},
    dtype=torch.qint8
)
```

2. **ONNX Export**
```python
torch.onnx.export(agent.online, 
                  torch.randn(1, 4), 
                  "models/dqn.onnx")
```

3. **Caching**
```python
from functools import lru_cache

@lru_cache(maxsize=10000)
def cached_recommendation(state_tuple):
    state = np.array(state_tuple)
    return get_recommendation(state)
```

## Performance Testing

Load test with Apache Bench:
```bash
# 10,000 requests, 100 concurrent
ab -n 10000 -c 100 https://yourdomain.com/api/health

# Expected: >1000 req/sec
```

Load test with Locust:
```bash
pip3 install locust
```

Create `locustfile.py`:
```python
from locust import HttpUser, task

class EcommerceUser(HttpUser):
    @task
    def recommend(self):
        self.client.post("/api/recommend", 
            json={"user_id": "user1", "current_item_id": 5})
```

Run:
```bash
locust -f locustfile.py --host=https://yourdomain.com
```

## Troubleshooting

### Issue: Slow Inference
- Check GPU availability: `torch.cuda.is_available()`
- Profile with: `python3 -m cProfile backend/app.py`
- Consider model quantization

### Issue: Memory Leak
- Monitor with: `memory_profiler`
- Check for dangling tensors: `torch.cuda.empty_cache()`

### Issue: API Timeout
- Increase Gunicorn timeout: `gunicorn --timeout 120`
- Add request queuing: Redis/Celery

### Issue: Database Slowness
- Add indexes on frequently queried columns
- Use connection pooling: SQLAlchemy pool_size
- Consider caching layer: Redis

## Monitoring Commands

```bash
# Check API health
curl https://yourdomain.com/api/health

# Monitor logs
tail -f /var/log/nginx/error.log
tail -f /var/log/gunicorn.log

# Check GPU usage
nvidia-smi

# Monitor system
htop
```

## Rollback Plan

If production fails:
1. Redirect traffic to previous version (DNS change)
2. Restore database from backup (daily automatic)
3. Use A/B testing to detect bad models
4. Keep 2-3 model versions available

## Timeline

- Week 1: Database setup + testing
- Week 2: Production deployment + monitoring
- Week 3: Load testing + optimization
- Week 4: Go live + feedback collection
- Ongoing: Auto-retraining + A/B testing

## Success Metrics

- **Uptime**: >99.5%
- **Response time**: <200ms p95
- **Inference accuracy**: >73% confidence
- **Cost per 1000 recommendations**: <$0.10
- **User satisfaction**: >4.0/5 stars (from feedback)

## Further Reading

- [Flask Production Deployment](https://flask.palletsprojects.com/deployment/)
- [PyTorch Model Serving](https://pytorch.org/serve/)
- [AWS Best Practices](https://docs.aws.amazon.com/)
