/**
 * Smart E-Commerce Frontend
 *
 * Manages:
 * - Product browsing
 * - AI recommendations (via DQN model)
 * - Purchase tracking
 * - User interaction logging
 */

// ============================================================
// Configuration
// ============================================================

const API_URL = 'http://127.0.0.1:5000/api';
const USER_ID = `user_${Date.now()}`; // Unique user ID per session

// Global state
let currentProductId = null;
let userState = null;
let allProducts = [];

// ============================================================
// Utility: API Calls
// ============================================================

async function apiCall(endpoint, method = 'GET', data = null) {
    try {
        const options = {
            method,
            headers: {
                'Content-Type': 'application/json'
            }
        };

        if (data) {
            options.body = JSON.stringify(data);
        }

        const response = await fetch(`${API_URL}${endpoint}`, options);
        const result = await response.json();

        if (!response.ok) {
            console.error(`API Error (${response.status}):`, result);
            return null;
        }

        return result;
    } catch (error) {
        console.error('API Call Error:', error);
        return null;
    }
}

// ============================================================
// UI Updates
// ============================================================

function updateAPIStatus(status) {
    const statusEl = document.getElementById('api-status');
    if (status === 'ok') {
        statusEl.textContent = '✓ Connected';
        statusEl.className = 'status-ok';
    } else if (status === 'error') {
        statusEl.textContent = '✗ Error';
        statusEl.className = 'status-error';
    } else {
        statusEl.textContent = '⏳ Connecting...';
        statusEl.className = 'status-pending';
    }
}

function renderProductGrid() {
    const grid = document.getElementById('products-grid');
    grid.innerHTML = '';

    allProducts.forEach(product => {
        const card = document.createElement('div');
        card.className = 'product-card';
        if (product.id === currentProductId) {
            card.classList.add('selected');
        }

        // Generate a consistent emoji for each product based on category
        const categoryEmojis = {
            'electronics': '📱',
            'clothing': '👕',
            'home': '🏠',
            'books': '📚'
        };
        const emoji = categoryEmojis[product.category] || '📦';

        card.innerHTML = `
            <div class="product-image">${emoji}</div>
            <div class="product-info">
                <div class="product-name">${product.name}</div>
                <div class="product-category">${product.category}</div>
                <div class="product-price">$${product.price.toFixed(2)}</div>
                <button class="btn-buy-grid" style="width: 100%; margin-top: 8px; padding: 6px;">💳 Buy</button>
            </div>
        `;

        // View product on click (anywhere except buy button)
        card.addEventListener('click', (e) => {
            if (e.target.classList.contains('btn-buy-grid')) {
                e.stopPropagation();
                purchaseRecommended(product.id);
            } else {
                selectProduct(product.id);
            }
        });

        grid.appendChild(card);
    });
}

function updateUserStats() {
    if (!userState) return;

    const { total_purchases, total_spent, total_clicks } = userState;
    document.getElementById('user-id-display').textContent = `User: ${USER_ID.substring(0, 12)}...`;
    document.getElementById('user-stats').textContent =
        `Purchases: ${total_purchases} | Spent: $${total_spent.toFixed(2)}`;

    document.getElementById('total-items').textContent = total_purchases;
    document.getElementById('total-clicks').textContent = total_clicks;
    document.getElementById('total-spent').textContent = `$${total_spent.toFixed(2)}`;
}

function updateCurrentProductDisplay() {
    if (!allProducts.length || currentProductId === null) return;

    const product = allProducts.find(p => p.id === currentProductId);
    if (!product) return;

    const categoryEmojis = {
        'electronics': '📱',
        'clothing': '👕',
        'home': '🏠',
        'books': '📚'
    };

    const display = document.getElementById('current-product');
    display.innerHTML = `
        <div style="margin-bottom: 15px; font-size: 2em;">
            ${categoryEmojis[product.category] || '📦'}
        </div>
        <div class="product-name">${product.name}</div>
        <div class="product-category">${product.category}</div>
        <div class="product-price">$${product.price.toFixed(2)}</div>
        <p style="margin-top: 12px; font-size: 0.9em; opacity: 0.9;">
            ${product.description}
        </p>
        <button class="btn-buy" style="width: 100%; margin-top: 12px; padding: 10px; font-weight: 600;" onclick="purchaseRecommended(${product.id})">
            💳 Buy Now
        </button>
    `;
}

async function updateRecommendation() {
    if (currentProductId === null) {
        document.getElementById('recommendation').innerHTML =
            '<div class="placeholder">View a product to get recommendation</div>';
        return;
    }

    const rec = await apiCall('/recommend', 'POST', {
        user_id: USER_ID,
        current_item_id: currentProductId
    });

    if (!rec) {
        updateAPIStatus('error');
        return;
    }

    updateAPIStatus('ok');

    const categoryEmojis = {
        'electronics': '📱',
        'clothing': '👕',
        'home': '🏠',
        'books': '📚'
    };

    const html = `
        <div style="background: white; padding: 12px; border-radius: 6px;">
            <div style="font-size: 1.8em; margin-bottom: 8px;">
                ${categoryEmojis[rec.product.category] || '📦'}
            </div>
            <div class="product-name">${rec.product.name}</div>
            <div class="product-category">${rec.product.category}</div>
            <div class="product-price">$${rec.product.price.toFixed(2)}</div>
            <span class="confidence-badge">Confidence: ${Math.round(rec.confidence * 100)}%</span>
            <div class="action-buttons">
                <button class="btn-buy" onclick="purchaseRecommended(${rec.recommended_item_id})">
                    💳 Buy
                </button>
                <button class="btn-click" onclick="clickRecommended(${rec.recommended_item_id})">
                    👁️ More Info
                </button>
            </div>
        </div>
    `;

    document.getElementById('recommendation').innerHTML = html;
}

function updatePurchaseHistory() {
    if (!userState || !userState.purchase_history.length) {
        document.getElementById('purchase-history').innerHTML =
            '<p class="empty-state">No purchases yet</p>';
        return;
    }

    let html = '<div class="history-list">';
    userState.purchase_history.forEach(itemId => {
        const product = allProducts.find(p => p.id === itemId);
        if (product) {
            html += `
                <div class="history-item">
                    ${product.name} <strong style="float: right;">$${product.price.toFixed(2)}</strong>
                </div>
            `;
        }
    });
    html += '</div>';

    document.getElementById('purchase-history').innerHTML = html;
}

function updateDebugInfo() {
    if (!userState) {
        document.getElementById('debug-user').textContent = '-';
        document.getElementById('debug-item').textContent = '-';
        document.getElementById('debug-confidence').textContent = '-';
        return;
    }

    document.getElementById('debug-user').textContent = USER_ID.substring(0, 16) + '...';
    document.getElementById('debug-item').textContent = currentProductId || '-';
}

// ============================================================
// User Actions
// ============================================================

async function selectProduct(productId) {
    currentProductId = productId;

    // Update UI
    renderProductGrid();
    updateCurrentProductDisplay();
    updateDebugInfo();

    // Get recommendation from DQN
    await updateRecommendation();
}

async function purchaseRecommended(productId) {
    const result = await apiCall('/purchase', 'POST', {
        user_id: USER_ID,
        product_id: productId
    });

    if (!result) {
        alert('Error recording purchase');
        return;
    }

    // Update user state
    userState = await apiCall(`/users/${USER_ID}`);
    updateUserStats();
    updatePurchaseHistory();

    alert(`✓ Purchased: ${result.product_name}`);
}

async function clickRecommended(productId) {
    const result = await apiCall('/click', 'POST', {
        user_id: USER_ID,
        product_id: productId
    });

    if (!result) {
        alert('Error recording click');
        return;
    }

    // Update and view that product
    userState = await apiCall(`/users/${USER_ID}`);
    updateUserStats();

    await selectProduct(productId);
}

// ============================================================
// Initialization
// ============================================================

async function init() {
    console.log('Initializing app...');
    updateAPIStatus('');

    // 1. Load products
    const productsData = await apiCall('/products');
    if (!productsData) {
        updateAPIStatus('error');
        alert('Failed to load products');
        return;
    }
    allProducts = productsData.products;
    console.log(`✓ Loaded ${allProducts.length} products`);
    renderProductGrid();

    // 2. Create user
    const userData = await apiCall(`/users/${USER_ID}`, 'POST', {});
    if (!userData) {
        updateAPIStatus('error');
        alert('Failed to create user');
        return;
    }
    console.log(`✓ User created: ${USER_ID}`);

    // 3. Get user state
    userState = await apiCall(`/users/${USER_ID}`);
    if (!userState) {
        updateAPIStatus('error');
        alert('Failed to load user state');
        return;
    }
    console.log('✓ User state loaded');
    updateUserStats();

    // 4. Select random initial product
    if (userData.current_product) {
        await selectProduct(userData.current_product.id);
    }

    updateAPIStatus('ok');
    console.log('✓ App initialized successfully');
}

// ============================================================
// Start App
// ============================================================

document.addEventListener('DOMContentLoaded', init);
