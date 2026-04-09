"""
Réentraînement des models RL avec état enrichi.

État = [item_id, category, price_norm, history_encoding]
Au lieu de juste [item_id]

Activation: GELU au lieu de ReLU (meilleur pour recommendation)
Architecture: Avec skip connections et dropout

Périodes réduites pour tests rapides:
- QL: 500 ep (au lieu de 5000)
- DQN: 300 ep (au lieu de 3000)
- GRU: 300 ep (au lieu de 3000)

À changer pour production:
  QL: 5000
  DQN: 5000
  GRU: 5000
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

# Imports depuis le data module
sys.path.insert(0, str(Path(__file__).parent))
from data.products import N_ITEMS, N_CATEGORIES
from data.user_state import UserState

# Configuration device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ============================================================================
# 1. ENVIRONMENT - Simulation simple pour tests
# ============================================================================

class SimpleRecommendationEnv:
    """
    Environnement simple pour tester les models.
    - Action: recommander un produit (0-9)
    - Reward: +3 si achat, +1 si click, -0.5 si repeat, 0 sinon
    """

    def __init__(self, n_items=N_ITEMS, seed=42):
        self.n_items = n_items
        self.np_random = np.random.RandomState(seed)
        self.user_state = None
        self.current_item = None

    def reset(self):
        """Réinitialise l'env avec un user et item aléatoires"""
        self.user_state = UserState(user_id=1)
        self.current_item = self.np_random.randint(0, self.n_items)
        self.user_state.set_current_item(self.current_item)
        return self.user_state.get_state_vector()

    def step(self, action):
        """
        Exécute une action (recommandation).

        Returns:
            next_state, reward, done, info
        """
        assert 0 <= action < self.n_items, f"Action invalide: {action}"

        # Simuler la réaction de l'utilisateur (probabiliste)
        prob_buy = self.np_random.rand()
        prob_click = self.np_random.rand()
        prob_ignore = self.np_random.rand()

        reward = 0.0
        done = False
        event = 'ignore'

        if prob_buy < 0.1:  # 10% achat
            reward = 3.0
            event = 'buy'
            self.user_state.add_purchase(action, 50.0)
            done = True
        elif prob_click < 0.3:  # 30% click
            reward = 1.0
            event = 'click'
            self.user_state.add_click(action)
        else:  # sinon ignore
            reward = 0.0
            event = 'ignore'

        # Penalty si repeat
        if action in self.user_state.purchase_history:
            reward -= 0.5

        # Nouvel item aléatoire
        self.current_item = self.np_random.randint(0, self.n_items)
        self.user_state.set_current_item(self.current_item)
        next_state = self.user_state.get_state_vector()

        info = {'event': event, 'repeat': action in self.user_state.purchase_history}

        return next_state, reward, done, info


# ============================================================================
# 2. Q-LEARNING avec état enrichi
# ============================================================================

class QLearningEnriched:
    """
    Q-Learning classique avec état enrichi [item, cat, price, history].

    Architecture: Table Q(s,a) simple
    État: 4D float → discrétisé en buckets pour Q-table
    """

    def __init__(self, n_items=N_ITEMS, n_states_bucket=20):
        """
        Args:
            n_items: Nombre de produits
            n_states_bucket: Nombre de buckets pour discrétiser l'état continu
        """
        self.n_items = n_items
        self.n_states = n_states_bucket ** 4  # 4D state
        self.Q = np.random.randn(self.n_states, n_items) * 0.01

        # Hyperparams
        self.lr = 0.1
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.02

    def _discretize_state(self, state_vec):
        """Convertit état continu en indice discret"""
        # state_vec shape: (4,) avec valeurs ~[0, 10] (item_id peut être jusqu'à 9)
        # On bucketize chaque dimension
        discretized = np.clip(state_vec / 10.0 * 19, 0, 19).astype(int)
        state_idx = (discretized[0] * (20**3) + discretized[1] * (20**2) +
                     discretized[2] * 20 + discretized[3])
        return min(state_idx, self.n_states - 1)

    def act(self, state_vec):
        """ε-greedy action selection"""
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.n_items)
        state_idx = self._discretize_state(state_vec)
        return np.argmax(self.Q[state_idx])

    def update(self, state_vec, action, reward, next_state_vec, done):
        """Q-learning update"""
        state_idx = self._discretize_state(state_vec)
        next_state_idx = self._discretize_state(next_state_vec)

        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.Q[next_state_idx])

        self.Q[state_idx, action] += self.lr * (target - self.Q[state_idx, action])

        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def act_greedy(self, state_vec):
        """Action sans exploration (inference)"""
        state_idx = self._discretize_state(state_vec)
        return np.argmax(self.Q[state_idx])


# ============================================================================
# 3. DUELING DQN avec activation alternative
# ============================================================================

class DuelingQNetworkEnriched(nn.Module):
    """
    Dueling DQN avec:
    - État enrichi 4D
    - Activation GELU (meilleur que ReLU pour recommendation)
    - Skip connections pour gradient flow
    - Dropout pour regularization
    """

    def __init__(self, input_dim=4, hidden=256, n_items=N_ITEMS):
        """
        Args:
            input_dim: Dimension du vecteur d'état (4)
            hidden: Nombre de neurones hidden
            n_items: Nombre d'actions possibles
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden = hidden
        self.n_items = n_items

        # Shared feature extraction
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),  # Activation: GELU au lieu de ReLU
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        # Value branch: V(s)
        self.v_net = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden // 2, 1),
        )

        # Advantage branch: A(s,a)
        self.a_net = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden // 2, n_items),
        )

    def forward(self, x):
        """
        Forward pass.
        Q(s,a) = V(s) + (A(s,a) - mean_a A(s,a))
        """
        features = self.feature_net(x)

        v = self.v_net(features)  # Shape: (batch, 1)
        a = self.a_net(features)  # Shape: (batch, n_items)

        # Advantage normalization
        a_mean = a.mean(dim=1, keepdim=True)
        q = v + (a - a_mean)

        return q


class DQNAgentEnriched:
    """
    Agent DQN avec état enrichi.

    Features:
    - Experience replay
    - Target network avec soft update (tau=0.001)
    - Dueling architecture
    - GELU activation
    """

    def __init__(self, n_items=N_ITEMS, input_dim=4, hidden=256,
                 lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_decay=0.995,
                 epsilon_min=0.02, tau=0.001, buffer_size=10000, batch_size=64):

        self.n_items = n_items
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.tau = tau
        self.batch_size = batch_size

        # Networks
        self.online = DuelingQNetworkEnriched(input_dim, hidden, n_items).to(device)
        self.target = DuelingQNetworkEnriched(input_dim, hidden, n_items).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.online.parameters(), lr=lr)
        self.loss_fn = nn.HuberLoss(reduction='mean')

        # Replay buffer
        self.buffer = []
        self.buffer_size = buffer_size

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.buffer.append((state, action, reward, next_state, done))
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

    def replay(self):
        """Train on mini-batch from replay buffer"""
        if len(self.buffer) < self.batch_size:
            return 0.0

        # Sample mini-batch
        indices = np.random.choice(len(self.buffer), self.batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in indices])

        # Convert to tensors
        states_t = torch.tensor(states, dtype=torch.float32).to(device)
        actions_t = torch.tensor(actions, dtype=torch.long).to(device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_states_t = torch.tensor(next_states, dtype=torch.float32).to(device)
        dones_t = torch.tensor(dones, dtype=torch.float32).to(device)

        # Forward pass
        q_pred = self.online(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Target
        with torch.no_grad():
            q_next = self.target(next_states_t).max(1)[0]
            q_target = rewards_t + self.gamma * q_next * (1 - dones_t)

        # Loss
        loss = self.loss_fn(q_pred, q_target)

        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), 1.0)
        self.optimizer.step()

        # Soft update target network
        for target_param, online_param in zip(self.target.parameters(),
                                              self.online.parameters()):
            target_param.data.copy_(
                self.tau * online_param.data + (1 - self.tau) * target_param.data
            )

        return loss.item()

    def act(self, state):
        """ε-greedy action selection"""
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.n_items)

        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.online(state_t)
        return q_values.argmax(dim=1).item()

    def act_greedy(self, state):
        """Greedy action (no exploration)"""
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.online(state_t)
        return q_values.argmax(dim=1).item()

    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# ============================================================================
# 4. TRAINING FUNCTIONS
# ============================================================================

def train_ql(n_episodes=500, seed=42):
    """Train Q-Learning"""
    np.random.seed(seed)
    env = SimpleRecommendationEnv(seed=seed)

    agent = QLearningEnriched(n_items=N_ITEMS)
    rewards_history = []

    print(f"\n{'='*60}")
    print(f"Training Q-Learning (enriched) - {n_episodes} episodes")
    print(f"{'='*60}")

    for ep in range(n_episodes):
        state = env.reset()
        ep_reward = 0.0
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done)
            ep_reward += reward
            state = next_state

        rewards_history.append(ep_reward)

        if (ep + 1) % 50 == 0:
            avg_reward = np.mean(rewards_history[-50:])
            print(f"Episode {ep+1:3d}/{n_episodes} | Avg Reward: {avg_reward:.4f} | ε: {agent.epsilon:.4f}")

    print(f"Final reward (last 50 eps): {np.mean(rewards_history[-50:]):.4f}")
    return agent, rewards_history


def train_dqn(n_episodes=300, seed=42):
    """Train DQN with enriched state"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = SimpleRecommendationEnv(seed=seed)

    agent = DQNAgentEnriched(n_items=N_ITEMS, input_dim=4, hidden=256)
    rewards_history = []
    loss_history = []

    print(f"\n{'='*60}")
    print(f"Training DQN (enriched, GELU) - {n_episodes} episodes")
    print(f"{'='*60}")

    for ep in range(n_episodes):
        state = env.reset()
        ep_reward = 0.0
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)

            agent.remember(state, action, reward, next_state, done)
            loss = agent.replay()
            loss_history.append(loss)

            ep_reward += reward
            state = next_state

        agent.decay_epsilon()
        rewards_history.append(ep_reward)

        if (ep + 1) % 50 == 0:
            avg_reward = np.mean(rewards_history[-50:])
            avg_loss = np.mean([l for l in loss_history[-500:] if l > 0])
            print(f"Episode {ep+1:3d}/{n_episodes} | Reward: {avg_reward:.4f} | Loss: {avg_loss:.4f} | ε: {agent.epsilon:.4f}")

    print(f"Final reward (last 50 eps): {np.mean(rewards_history[-50:]):.4f}")
    return agent, rewards_history


# ============================================================================
# 5. TEST & MAIN
# ============================================================================

def test_models(ql_agent, dqn_agent):
    """Test les models entrainés"""
    print(f"\n{'='*60}")
    print(f"Testing models on 100 greedy episodes")
    print(f"{'='*60}")

    env = SimpleRecommendationEnv(seed=99)

    ql_rewards = []
    dqn_rewards = []

    for _ in range(100):
        # Test QL
        state = env.reset()
        ep_reward = 0.0
        done = False
        while not done:
            action = ql_agent.act_greedy(state)
            next_state, reward, done, info = env.step(action)
            ep_reward += reward
            state = next_state
        ql_rewards.append(ep_reward)

        # Test DQN
        state = env.reset()
        ep_reward = 0.0
        done = False
        while not done:
            action = dqn_agent.act_greedy(state)
            next_state, reward, done, info = env.step(action)
            ep_reward += reward
            state = next_state
        dqn_rewards.append(ep_reward)

    print(f"Q-Learning    | Avg reward: {np.mean(ql_rewards):.4f} ± {np.std(ql_rewards):.4f}")
    print(f"DQN (GELU)    | Avg reward: {np.mean(dqn_rewards):.4f} ± {np.std(dqn_rewards):.4f}")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("RETRAINING MODELS WITH ENRICHED STATE")
    print("="*60)

    # Train models
    ql_agent, ql_rewards = train_ql(n_episodes=500)
    dqn_agent, dqn_rewards = train_dqn(n_episodes=300)

    # Test
    test_models(ql_agent, dqn_agent)

    # Save
    os.makedirs('models', exist_ok=True)
    torch.save(dqn_agent.online.state_dict(), 'models/dqn_enriched.pt')
    print(f"\n✓ DQN model saved to models/dqn_enriched.pt")

    print("\n" + "="*60)
    print("SETTINGS FOR PRODUCTION:")
    print("="*60)
    print("Change these in retrain_enriched.py:")
    print("  train_ql():  n_episodes=500  →  n_episodes=5000")
    print("  train_dqn(): n_episodes=300  →  n_episodes=5000")
    print("="*60)
