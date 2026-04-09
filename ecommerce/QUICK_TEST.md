# Quick Test Guide - 5 Minutes

## ✅ Backend: Ready
```bash
# Backend is running on http://127.0.0.1:5000
# DQN model loaded with 36 products
# All API endpoints functional
```

## 🖥️ Frontend: Ready to Test

### Option 1: Open File Directly
```
File: ecommerce/frontend/index.html
Action: Open in any browser
```

### Option 2: Use VS Code Live Server
- Right-click `frontend/index.html` → Open with Live Server
- Browser opens automatically

### Option 3: Python HTTP Server
```bash
cd ecommerce/frontend
python3 -m http.server 8000
# Visit: http://localhost:8000
```

---

## 📋 Test Checklist

### Pagination (3 Pages Total)
- [ ] Page 1 shows 12 products
- [ ] "Next" button visible on page 1
- [ ] "Previous" button hidden on page 1
- [ ] Page 2 shows products 13-24
- [ ] Both "Previous" and "Next" visible on page 2
- [ ] Page 3 shows products 25-36
- [ ] "Next" hidden on page 3, "Previous" visible

### Images
- [ ] All product cards show placeholder images
- [ ] No console errors (F12 → Console)
- [ ] No CORS errors (was: NS_BINDING_ABORTED)

### Product Interactions
- [ ] Click a product card → Details shown on left
- [ ] Product image, name, category, price visible
- [ ] "Buy Now" button works
- [ ] "More Info" on recommendation works

### Purchases
- [ ] Buy a product → Alert appears with name
- [ ] Stats update: "Purchases: 1" appears
- [ ] "Spent: $X.XX" shows correct total
- [ ] Purchase history lists bought item

### Recommendations
- [ ] Recommendation card shows after selecting product
- [ ] "Confidence: X%" badge appears
- [ ] Confidence changes between 0-100%
- [ ] Two buttons: Buy and More Info

### User Tracking
- [ ] User ID shown (first 12 chars)
- [ ] Stats update on each action
- [ ] Total clicks incremented (not just purchases)

---

## 🐛 If Something Breaks

### Issue: No products shown
```bash
# Restart backend
pkill -f "python3 backend/app.py"
cd ecommerce && python3 backend/app.py
```

### Issue: Images still showing CORS errors
```bash
# Verify images are placeholder.com
curl http://127.0.0.1:5000/api/products | grep image | head -1
# Should show: "https://via.placeholder.com/400x300?text=Product"
```

### Issue: No recommendations appear
```bash
# Test API directly
curl -X POST http://127.0.0.1:5000/api/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": "test", "current_item_id": 0}'
# Should return JSON with "recommended_item_id" and "confidence"
```

### Issue: Pagination buttons missing
- Check browser console (F12 → Console)
- Look for JavaScript errors
- Verify `frontend/app.js` has pagination code (should have `currentPage`, `PRODUCTS_PER_PAGE`)

---

## 📊 Success Indicators

When working correctly, you should see:
- ✅ 36 products across 3 pages
- ✅ All images load (placeholder circles)
- ✅ Recommendation appears with 50-95% confidence
- ✅ Stats increase when you buy/click
- ✅ No console errors
- ✅ Smooth pagination navigation

---

## 🚀 Backend Status Command

```bash
# Quick health check
curl http://127.0.0.1:5000/api/health

# Expected output:
# {"device":"cpu","model":"dqn","n_items":36,"status":"ok"}
```

---

**That's it!** Open the frontend and explore. The system learns from your clicks and purchases.
