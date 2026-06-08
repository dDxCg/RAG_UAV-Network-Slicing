# mappo_agents.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Dirichlet, Categorical
from collections import deque
import random
from typing import Dict, List, Tuple, Optional
from utils import Configuration


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for agent communication"""
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations and split into heads
        Q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.embed_dim
        )
        
        output = self.out_linear(context)
        return output, attention_weights


class ActorNetworkPPO(nn.Module):
    """
    Stochastic Actor for MAPPO with modular encoders (matching MADDPG architecture)
    Outputs distribution parameters instead of deterministic actions
    
    Total obs_dim = 80:
      - UAV state: 5 dims
      - DA info: 63 dims (9 DAs x 7 features)
      - Handover state: 4 dims
      - Surrounding: 8 dims
    """
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        
        # Observation component dimensions (same as MADDPG)
        self.uav_state_dim = 5
        self.da_info_dim = 63      # 9 DAs x 7 features
        self.handover_dim = 4
        self.load_pred_dim = 0
        self.surrounding_dim = 8
        
        # Verify total
        total_expected = (self.uav_state_dim + self.da_info_dim + 
                         self.handover_dim + self.load_pred_dim + 
                         self.surrounding_dim)
        assert obs_dim == total_expected, \
            f"obs_dim mismatch: expected {total_expected}, got {obs_dim}"
        
        # ============================================
        # Separate Encoders for Each Component (same as MADDPG)
        # ============================================
        
        # 1. UAV State Encoder (position, power, battery)
        self.uav_encoder = nn.Sequential(
            nn.Linear(self.uav_state_dim, 32),
            nn.ReLU(),
            nn.LayerNorm(32)
        )
        
        # 2. DA Info Encoder (CRITICAL for bandwidth allocation!)
        self.da_encoder = nn.Sequential(
            nn.Linear(self.da_info_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.LayerNorm(64)
        )
        
        # 3. Handover State Encoder (CRITICAL for movement decisions!)
        self.handover_encoder = nn.Sequential(
            nn.Linear(self.handover_dim, 16),
            nn.ReLU()
        )
        
        # 4. Surrounding Encoder (for coordination)
        self.surrounding_encoder = nn.Sequential(
            nn.Linear(self.surrounding_dim, 16),
            nn.ReLU()
        )
        
        # ============================================
        # Fusion and Policy Network
        # ============================================
        
        # Combined feature dimension
        combined_dim = 32 + 64 + 16 + 16  # = 128
        
        # Attention mechanism for DA features (helps bandwidth allocation)
        self.da_attention = nn.MultiheadAttention(
            embed_dim=64, 
            num_heads=4,
            batch_first=True
        )
        
        # Shared policy network
        self.policy_net = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
        )
        
        # ============================================
        # Distribution Parameter Heads (MAPPO-specific)
        # ============================================
        
        # Position distribution (3D movement) - Gaussian
        self.position_mean_head = nn.Linear(128, 3)
        self.position_log_std = nn.Parameter(torch.zeros(3))
        
        # Power distribution (1D) - Gaussian
        self.power_mean_head = nn.Linear(128, 1)
        self.power_log_std = nn.Parameter(torch.zeros(1))
        
        # Bandwidth allocation (9 DAs) - Dirichlet or Categorical
        # Use alpha parameters for Dirichlet distribution
        num_das = action_dim - 4  # action_dim = 13, so bandwidth is 9 dims
        self.bandwidth_alpha_head = nn.Linear(128, num_das)
        
        # Learnable temperature for bandwidth concentration
        self.log_bandwidth_temperature = nn.Parameter(
            torch.log(torch.tensor(5.0))
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Orthogonal initialization for PPO"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        
        # Small initialization for output heads
        nn.init.orthogonal_(self.position_mean_head.weight, gain=0.01)
        nn.init.orthogonal_(self.power_mean_head.weight, gain=0.01)
        nn.init.orthogonal_(self.bandwidth_alpha_head.weight, gain=0.01)
    
    def forward(self, obs):
        """
        Forward pass to get distribution parameters
        
        Returns:
            Dictionary with distribution parameters
        """
        batch_size = obs.size(0)
        
        # ============================================
        # Split observation into components
        # ============================================
        idx = 0
        
        # UAV state (5)
        uav_state = obs[:, idx:idx+self.uav_state_dim]
        idx += self.uav_state_dim
        
        # DA info (63)
        da_info = obs[:, idx:idx+self.da_info_dim]
        idx += self.da_info_dim
        
        # Handover state (4)
        handover_state = obs[:, idx:idx+self.handover_dim]
        idx += self.handover_dim
        
        # Surrounding (8)
        surrounding_state = obs[:, idx:idx+self.surrounding_dim]
        
        # ============================================
        # Encode each component
        # ============================================
        
        uav_encoded = self.uav_encoder(uav_state)
        handover_encoded = self.handover_encoder(handover_state)
        surrounding_encoded = self.surrounding_encoder(surrounding_state)
        
        # DA encoding with self-attention
        da_encoded = self.da_encoder(da_info)
        
        # Apply attention to DA features
        da_attended, _ = self.da_attention(
            da_encoded.unsqueeze(1),
            da_encoded.unsqueeze(1),
            da_encoded.unsqueeze(1)
        )
        da_attended = da_attended.squeeze(1)
        
        # ============================================
        # Combine all features
        # ============================================
        
        combined = torch.cat([
            uav_encoded,         # 32
            da_attended,         # 64
            handover_encoded,    # 16
            surrounding_encoded  # 16
        ], dim=-1)  # Total: 128
        
        # ============================================
        # Generate distribution parameters
        # ============================================
        
        features = self.policy_net(combined)
        
        # Position distribution (Gaussian)
        pos_mean = torch.tanh(self.position_mean_head(features))  # [-1, 1]
        pos_std = torch.exp(self.position_log_std.clamp(-20, 2))
        
        # Power distribution (Gaussian)
        power_mean = torch.sigmoid(self.power_mean_head(features))  # [0, 1]
        power_std = torch.exp(self.power_log_std.clamp(-20, 2))
        
        # Bandwidth distribution (Dirichlet)
        # Alpha parameters must be > 0
        temperature = torch.exp(self.log_bandwidth_temperature).clamp(2.0, 20.0)
        bandwidth_alpha = F.softplus(self.bandwidth_alpha_head(features)) + 0.1
        bandwidth_alpha = bandwidth_alpha * temperature  # Scale concentration
        
        return {
            'pos_mean': pos_mean,
            'pos_std': pos_std,
            'power_mean': power_mean,
            'power_std': power_std,
            'bandwidth_alpha': bandwidth_alpha
        }
    
    def get_action(self, obs, deterministic=False):
        """
        Sample action from the policy
        
        Args:
            obs: Observation tensor
            deterministic: If True, return mean (for evaluation)
        
        Returns:
            action: Sampled action
            log_prob: Log probability of the action
        """
        params = self.forward(obs)
        
        if deterministic:
            # Use mean for deterministic action
            position = params['pos_mean']
            power = params['power_mean']
            # For Dirichlet, use normalized alpha as mean
            bandwidth = params['bandwidth_alpha'] / params['bandwidth_alpha'].sum(dim=-1, keepdim=True)
            
            # No log prob for deterministic
            log_prob = torch.zeros(obs.shape[0], 1, device=obs.device)
        else:
            # Sample from distributions
            # Position
            pos_dist = Normal(params['pos_mean'], params['pos_std'])
            position = pos_dist.sample()
            position = torch.clamp(position, -1, 1)
            pos_log_prob = pos_dist.log_prob(position).sum(dim=-1, keepdim=True)
            
            # Power
            power_dist = Normal(params['power_mean'], params['power_std'])
            power = power_dist.sample()
            power = torch.clamp(power, 0.05, 1.0)
            power_log_prob = power_dist.log_prob(power)
            
            # Bandwidth (Dirichlet)
            bandwidth_dist = Dirichlet(params['bandwidth_alpha'])
            bandwidth = bandwidth_dist.sample()
            bandwidth_log_prob = bandwidth_dist.log_prob(bandwidth).unsqueeze(-1)
            
            # Total log probability
            log_prob = pos_log_prob + power_log_prob + bandwidth_log_prob
        
        # Concatenate action components
        action = torch.cat([position, power, bandwidth], dim=-1)
        
        return action, log_prob
    
    def evaluate_actions(self, obs, actions):
        """
        Evaluate log probabilities and entropy of given actions
        
        Args:
            obs: Observation tensor
            actions: Action tensor
        
        Returns:
            log_prob: Log probability of actions
            entropy: Entropy of the policy
        """
        params = self.forward(obs)
        
        # Extract action components
        position = actions[:, :3]
        power = actions[:, 3:4]
        bandwidth = actions[:, 4:]
        
        # Position log probability and entropy
        pos_dist = Normal(params['pos_mean'], params['pos_std'])
        pos_log_prob = pos_dist.log_prob(position).sum(dim=-1, keepdim=True)
        pos_entropy = pos_dist.entropy().sum(dim=-1, keepdim=True)
        
        # Power log probability and entropy
        power_dist = Normal(params['power_mean'], params['power_std'])
        power_log_prob = power_dist.log_prob(power)
        power_entropy = power_dist.entropy()
        
        # Bandwidth log probability and entropy
        bandwidth_dist = Dirichlet(params['bandwidth_alpha'])
        bandwidth_log_prob = bandwidth_dist.log_prob(bandwidth).unsqueeze(-1)
        bandwidth_entropy = bandwidth_dist.entropy().unsqueeze(-1)
        
        # Total log probability and entropy
        log_prob = pos_log_prob + power_log_prob + bandwidth_log_prob
        entropy = pos_entropy + power_entropy + bandwidth_entropy
        
        return log_prob, entropy


class CriticNetworkPPO(nn.Module):
    """
    Centralized Value Network for MAPPO
    Only takes states (no actions, since on-policy)
    
    Input: all states (obs_dim × num_agents)
    Output: scalar value estimate
    """
    def __init__(self, state_dim, num_agents=5):
        super().__init__()
        
        # Input: concatenated states from all agents
        input_dim = state_dim * num_agents
        # For 5 agents: 80 × 5 = 400
        
        # Deeper network for better value estimation
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Dropout(0.1),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            
            nn.Linear(64, 1)
        )
        
        # Initialize with orthogonal weights
        self._init_weights()
    
    def _init_weights(self):
        """Orthogonal initialization for PPO"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        
        # Small initialization for output layer
        self.net[-1].weight.data.mul_(0.1)
        self.net[-1].bias.data.mul_(0.1)
    
    def forward(self, states):
        """
        Args:
            states: (batch, num_agents * state_dim) OR (batch, num_agents, state_dim)
        Returns:
            values: (batch, 1)
        """
        if states.dim() == 3:
            # Flatten agents dimension
            states = states.flatten(1)
        
        return self.net(states)


class RolloutBuffer:
    """On-policy rollout buffer for MAPPO"""
    def __init__(self, buffer_size: int, num_agents: int, obs_dim: int, action_dim: int):
        self.buffer_size = buffer_size
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.clear()
    
    def clear(self):
        """Clear the buffer"""
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.position = 0
    
    def push(self, obs, action, log_prob, reward, done, value):
        """Store transition"""
        self.observations.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.position += 1
    
    def get(self):
        """Get all stored transitions and clear buffer"""
        data = {
            'observations': np.array(self.observations),
            'actions': np.array(self.actions),
            'log_probs': np.array(self.log_probs),
            'rewards': np.array(self.rewards),
            'dones': np.array(self.dones),
            'values': np.array(self.values)
        }
        self.clear()
        return data
    
    def __len__(self):
        return len(self.observations)


class MAPPOAgent:
    """Multi-Agent Proximal Policy Optimization Agent for UAV Network Slicing"""
    def __init__(self, 
                 num_agents: int,
                 obs_dim: int,
                 action_dim: int,
                 training: bool = True,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 config_path = "./config/agents/mappo_default.yaml"):
        
        self.config = Configuration(config_path)
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = torch.device(device)
        
        # PPO hyperparameters
        self.gamma = getattr(self.config.mappo, 'gamma', 0.99)
        self.gae_lambda = getattr(self.config.mappo, 'gae_lambda', 0.95)
        self.clip_epsilon = getattr(self.config.mappo, 'clip_epsilon', 0.2)
        self.value_loss_coef = getattr(self.config.mappo, 'value_loss_coef', 0.5)
        self.entropy_coef = getattr(self.config.mappo, 'entropy_coef', 0.01)
        self.max_grad_norm = getattr(self.config.mappo, 'max_grad_norm', 0.5)
        self.n_epochs = getattr(self.config.mappo, 'n_epochs', 10)
        self.n_steps = getattr(self.config.mappo, 'n_steps', 2048)
        self.batch_size = getattr(self.config.mappo, 'batch_size', 64)
        self.lr_actor = getattr(self.config.mappo, 'actor_lr', 3e-4)
        self.lr_critic = getattr(self.config.mappo, 'critic_lr', 1e-3)
        
        # Initialize actor networks for each agent
        self.actors = []
        self.actor_optimizers = []
        
        for i in range(num_agents):
            actor = ActorNetworkPPO(obs_dim, action_dim).to(self.device)
            self.actors.append(actor)
            
            if self.device.type == 'cuda':
                self.actor_optimizers.append(
                    optim.AdamW(actor.parameters(), lr=self.lr_actor, fused=True)
                )
            else:
                self.actor_optimizers.append(
                    optim.Adam(actor.parameters(), lr=self.lr_actor)
                )
        
        # Initialize centralized critic
        self.critic = CriticNetworkPPO(obs_dim, num_agents).to(self.device)
        
        if self.device.type == 'cuda':
            self.critic_optimizer = optim.AdamW(
                self.critic.parameters(), lr=self.lr_critic, fused=True
            )
        else:
            self.critic_optimizer = optim.Adam(
                self.critic.parameters(), lr=self.lr_critic
            )
        
        # Rollout buffer
        self.buffer = RolloutBuffer(self.n_steps, num_agents, obs_dim, action_dim)
        
        self.training_mode = training
        self.pin_memory = (self.device.type == 'cuda')
        
        # Print initialization info (matching MADDPG style)
        print(f"  Rollout steps: {self.n_steps}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Pin memory: {self.pin_memory}")
        print(f"  Agent Device: {self.device}")
        if self.device.type == 'cuda':
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("      WARNING: Using CPU! Training will be slow!")
        print(f"  PPO epochs: {self.n_epochs}")
        print(f"  Clip epsilon: {self.clip_epsilon}")
        print(f"  GAE lambda: {self.gae_lambda}")

    def select_actions(self, observations: Dict[int, np.ndarray], 
                      deterministic: bool = False) -> Tuple[Dict[int, np.ndarray], 
                                                            Dict[int, float],
                                                            float]:
        """
        Select actions for all agents (matching MADDPG API)
        
        Args:
            observations: Dictionary of observations for each agent
            deterministic: If True, use mean actions (for evaluation)
        
        Returns:
            actions: Dictionary of actions for each agent
            log_probs: Dictionary of log probabilities for each agent
            value: Estimated value of the joint state
        """
        actions = {}
        log_probs = {}
        
        # Get actions for each agent
        obs_list = []
        for agent_id in range(self.num_agents):
            obs = observations[agent_id]
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            obs_list.append(obs_tensor)
            
            self.actors[agent_id].eval()
            with torch.no_grad():
                action, log_prob = self.actors[agent_id].get_action(
                    obs_tensor, 
                    deterministic=deterministic
                )
            
            actions[agent_id] = action.cpu().numpy()[0]
            log_probs[agent_id] = log_prob.cpu().numpy()[0, 0]
        
        # Get value estimate from centralized critic
        self.critic.eval()
        with torch.no_grad():
            obs_cat = torch.cat(obs_list, dim=-1)
            value = self.critic(obs_cat).cpu().numpy()[0, 0]
        
        return actions, log_probs, value

    def store_transition(self, observations: Dict[int, np.ndarray],
                        actions: Dict[int, np.ndarray],
                        log_probs: Dict[int, float],
                        reward: float,
                        done: bool,
                        value: float):
        """
        Store transition in rollout buffer (matching MADDPG API)
        
        Args:
            observations: Dictionary of observations for each agent
            actions: Dictionary of actions for each agent
            log_probs: Dictionary of log probabilities for each agent
            reward: Shared reward
            done: Episode done flag
            value: Value estimate
        """
        # Convert dictionaries to arrays
        obs_array = np.concatenate([observations[i] for i in range(self.num_agents)])
        action_array = np.concatenate([actions[i] for i in range(self.num_agents)])
        log_prob_array = np.array([log_probs[i] for i in range(self.num_agents)])
        
        self.buffer.push(obs_array, action_array, log_prob_array, reward, done, value)

    def compute_gae(self, rewards, values, dones, next_value):
        """
        Compute Generalized Advantage Estimation (GAE)
        
        Args:
            rewards: Array of rewards [n_steps]
            values: Array of value estimates [n_steps]
            dones: Array of done flags [n_steps]
            next_value: Value estimate of the next state
        
        Returns:
            advantages: Array of advantage estimates [n_steps]
            returns: Array of discounted returns [n_steps]
        """
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value_t = next_value
            else:
                next_value_t = values[t + 1]
            
            # Mask if episode ended
            mask = 1.0 - dones[t]
            
            # TD error
            delta = rewards[t] + self.gamma * next_value_t * mask - values[t]
            
            # GAE
            last_gae = delta + self.gamma * self.gae_lambda * mask * last_gae
            advantages[t] = last_gae
        
        # Returns = advantages + values
        returns = advantages + values
        
        return advantages, returns

    def train(self, next_observations: Dict[int, np.ndarray]):
        """
        Update policy using PPO (matching MADDPG API)
        
        Args:
            next_observations: Observations from the next state (for value bootstrapping)
        
        Returns:
            Dictionary with training metrics (matching MADDPG format)
        """
        if len(self.buffer) < self.batch_size:
            return {}
        
        # Get rollout data
        data = self.buffer.get()
        
        # Convert to tensors
        observations = torch.FloatTensor(data['observations']).to(self.device)
        actions = torch.FloatTensor(data['actions']).to(self.device)
        old_log_probs = torch.FloatTensor(data['log_probs']).to(self.device)
        rewards = data['rewards']
        dones = data['dones']
        values = data['values']
        
        # Get next value for GAE computation
        with torch.no_grad():
            next_obs_list = []
            for agent_id in range(self.num_agents):
                next_obs = torch.FloatTensor(next_observations[agent_id]).unsqueeze(0).to(self.device)
                next_obs_list.append(next_obs)
            next_obs_cat = torch.cat(next_obs_list, dim=-1)
            next_value = self.critic(next_obs_cat).cpu().numpy()[0, 0]
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(rewards, values, dones, next_value)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Reshape for individual agents
        n_steps = len(rewards)
        obs_per_agent = observations.view(n_steps, self.num_agents, self.obs_dim)
        actions_per_agent = actions.view(n_steps, self.num_agents, self.action_dim)
        old_log_probs_per_agent = old_log_probs.view(n_steps, self.num_agents)
        
        # Training metrics (matching MADDPG format)
        actor_losses = []  # To match MADDPG's actor_losses
        critic_loss_total = 0
        policy_losses = []
        entropy_losses = []
        approx_kls = []
        clip_fractions = []
        n_updates = 0
        
        # Multiple epochs of optimization
        for epoch in range(self.n_epochs):
            # Generate random mini-batches
            indices = np.random.permutation(n_steps)
            
            for start_idx in range(0, n_steps, self.batch_size):
                end_idx = min(start_idx + self.batch_size, n_steps)
                batch_indices = indices[start_idx:end_idx]
                
                # Sample batch
                batch_obs = observations[batch_indices]
                batch_obs_per_agent = obs_per_agent[batch_indices]
                batch_actions_per_agent = actions_per_agent[batch_indices]
                batch_old_log_probs_per_agent = old_log_probs_per_agent[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # ============================================
                # Update Actors (Policy) - Per Agent
                # ============================================
                epoch_policy_losses = []
                epoch_entropy_losses = []
                epoch_approx_kls = []
                epoch_clip_fractions = []
                
                for i in range(self.num_agents):
                    # Set to training mode
                    self.actors[i].train()
                    
                    # Evaluate actions
                    new_log_probs, entropy = self.actors[i].evaluate_actions(
                        batch_obs_per_agent[:, i, :],
                        batch_actions_per_agent[:, i, :]
                    )
                    
                    # Get old log probs for this agent
                    old_log_probs_agent = batch_old_log_probs_per_agent[:, i:i+1]
                    
                    # Ratio for PPO
                    ratio = torch.exp(new_log_probs - old_log_probs_agent)
                    
                    # Clipped surrogate objective
                    surr1 = ratio * batch_advantages.unsqueeze(-1)
                    surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages.unsqueeze(-1)
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    # Entropy bonus (for exploration)
                    entropy_loss = -entropy.mean()
                    
                    # Total actor loss
                    actor_loss = policy_loss + self.entropy_coef * entropy_loss
                    
                    # Update actor
                    self.actor_optimizers[i].zero_grad()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), self.max_grad_norm)
                    self.actor_optimizers[i].step()
                    
                    # Logging
                    epoch_policy_losses.append(policy_loss.item())
                    epoch_entropy_losses.append(entropy.mean().item())
                    
                    # Approximate KL divergence
                    with torch.no_grad():
                        approx_kl = (old_log_probs_agent - new_log_probs).mean().item()
                        epoch_approx_kls.append(approx_kl)
                        
                        # Clip fraction
                        clip_fraction = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean().item()
                        epoch_clip_fractions.append(clip_fraction)
                
                # Store actor loss (mean across agents) to match MADDPG format
                actor_losses.append(np.mean(epoch_policy_losses))
                
                # ============================================
                # Update Critic (Value Function)
                # ============================================
                self.critic.train()
                
                # Predict values
                predicted_values = self.critic(batch_obs)
                
                # Value loss (MSE)
                value_loss = F.mse_loss(predicted_values.squeeze(-1), batch_returns)
                
                # Update critic
                self.critic_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
                
                # Accumulate metrics
                critic_loss_total += value_loss.item()
                policy_losses.extend(epoch_policy_losses)
                entropy_losses.extend(epoch_entropy_losses)
                approx_kls.extend(epoch_approx_kls)
                clip_fractions.extend(epoch_clip_fractions)
                n_updates += 1
        
        # Average metrics (matching MADDPG return format)
        metrics = {
            'actor_losses': actor_losses,  # List of losses (matching MADDPG)
            'critic_loss': critic_loss_total / n_updates,  # Single value (matching MADDPG)
            'policy_loss': np.mean(policy_losses),
            'value_loss': critic_loss_total / n_updates,
            'entropy': np.mean(entropy_losses),
            'approx_kl': np.mean(approx_kls),
            'clip_fraction': np.mean(clip_fractions),
            'explained_variance': 1 - np.var(returns.cpu().numpy() - values) / (np.var(returns.cpu().numpy()) + 1e-8)
        }
        
        return metrics

    def save_models(self, path: str):
        """Save all models (matching MADDPG API)"""
        checkpoint = {
            'critic': self.critic.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'actors': [actor.state_dict() for actor in self.actors],
            'actor_optimizers': [opt.state_dict() for opt in self.actor_optimizers],
        }
        torch.save(checkpoint, path)
        print(f"Models saved to {path}")
    
    def load_models(self, path: str):
        """Load all models (matching MADDPG API)"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        
        for i in range(self.num_agents):
            self.actors[i].load_state_dict(checkpoint['actors'][i])
            self.actor_optimizers[i].load_state_dict(checkpoint['actor_optimizers'][i])
        
        print(f"Models loaded from {path}")

    def reset_buffer(self):
        """Reset the rollout buffer"""
        self.buffer.clear()

