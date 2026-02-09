import numpy as np
import matplotlib.pyplot as plt
from environment import NetworkSlicingEnv
from agents import MADDPGAgent
from agents_mappo import MAPPOAgent
from utils import Configuration
import os
from typing import Dict, List, Tuple
from abc import ABC, abstractmethod
import json
from datetime import datetime
from tqdm import tqdm
import glob
import pandas as pd


class BaselineAgent(ABC):
    """Abstract base class for baseline agents"""
    
    def __init__(self, num_agents: int, obs_dim: int, action_dim: int):
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
    
    def select_actions(self, observations: Dict[int, np.ndarray], explore: bool = False) -> Dict[int, np.ndarray]:
        """Select actions for all agents"""
        pass
    
    def get_name(self) -> str:
        """Return agent name for logging"""
        return self.__class__.__name__

class RandomAgent(BaselineAgent):
    """Random baseline that selects actions uniformly at random with reasonable constraints"""
    
    def select_actions(self, observations: Dict[int, np.ndarray], explore: bool = False) -> Dict[int, np.ndarray]:
        """Select random actions with constraints to avoid invalid states"""
        actions = {}
        for agent_id in observations.keys():
            action = np.zeros(self.action_dim)
            
            # Position changes: random but reasonable [-1, 1]
            action[0:3] = np.random.uniform(-1, 1, 3)
            # Power: random but not too low (minimum 0% power to avoid SINR issues)
            action[3] = np.random.uniform(0, 1.0)
            
            # Bandwidth allocations: random positive values
            if self.action_dim > 4:
                action[4:] = np.random.uniform(0.01, 1.0, self.action_dim - 4)
                # Normalize to sum to reasonable value
                if np.sum(action[4:]) > 0:
                    action[4:] = action[4:] / np.sum(action[4:])
            
            actions[agent_id] = action
        return actions

class CoverageMaximizationGreedyAgent(BaselineAgent):
    """
    Coverage Maximization Strategy:
    - Primary goal: Maximize number of UEs within beam coverage
    - Adjusts height to ensure all UEs are covered by beam angle
    - Sacrifices some SINR for broader coverage
    """
    
    def __init__(self, num_agents: int, obs_dim: int, action_dim: int, env):
        super().__init__(num_agents, obs_dim, action_dim)
        self.env = env
    
    def select_actions(self, observations: Dict[int, np.ndarray], explore: bool = False) -> Dict[int, np.ndarray]:
        actions = {}
        
        for uav_id, obs in observations.items():
            action = np.zeros(self.action_dim)
            uav = self.env.uavs[uav_id]
            
            assigned_ues = [ue for ue in self.env.ues.values() 
                           if ue.assigned_uav == uav_id and ue.is_active]
            
            if not assigned_ues:
                action[3] = 0.4
                action[4:] = 1.0 / (self.action_dim - 4)
                actions[uav_id] = action
                continue
            
            # 1. Find height that covers ALL users
            target_height = self._compute_coverage_maximizing_height(uav, assigned_ues)
            
            # 2. Position towards coverage center
            position_action = self._compute_coverage_position(uav, assigned_ues, target_height)
            action[0:3] = position_action
            
            # 3. Power for wide coverage
            action[3] = self._compute_coverage_power(uav, assigned_ues, target_height)
            
            # 4. Bandwidth prioritizing uncovered users
            action[4:] = self._compute_coverage_bandwidth(uav)
            
            actions[uav_id] = action
        
        return actions
    
    def _compute_coverage_maximizing_height(self, uav, assigned_ues) -> float:
        """
        Find minimum height that covers all assigned UEs.
        Strategy: h = max_distance / tan(beam_angle)
        """
        if not assigned_ues:
            return self.env.uav_fly_range_h[0]
        
        # Find center of all users
        ue_positions = np.array([ue.position[:2] for ue in assigned_ues])
        center_2d = np.mean(ue_positions, axis=0)
        
        # Find maximum distance from center
        max_distance = max([np.linalg.norm(ue.position[:2] - center_2d) for ue in assigned_ues])
        
        # Calculate required height for beam coverage
        beam_angle_rad = np.radians(uav.beam_angle)
        
        if beam_angle_rad > 0:
            # h = d / tan(θ), add 10% margin for safety
            required_height = (max_distance / np.tan(beam_angle_rad)) * 1.1
        else:
            required_height = self.env.uav_fly_range_h[1]
        
        # Clamp to allowed range
        target_height = np.clip(required_height, 
                               self.env.uav_fly_range_h[0], 
                               self.env.uav_fly_range_h[1])
        
        return target_height
    
    def _compute_coverage_position(self, uav, assigned_ues, target_height) -> np.ndarray:
        """Move to position that maximizes coverage"""
        if not assigned_ues:
            return np.zeros(3)
        
        # Target is centroid of all users
        ue_positions = np.array([ue.position[:2] for ue in assigned_ues])
        centroid = np.mean(ue_positions, axis=0)
        
        # XY direction
        direction_xy = centroid - uav.position[:2]
        distance_xy = np.linalg.norm(direction_xy)
        
        if distance_xy > 5.0:
            direction_xy = (direction_xy / distance_xy) * 0.45
        else:
            direction_xy = np.zeros(2)
        
        # Z adjustment
        height_diff = target_height - uav.position[2]
        height_action = np.clip(height_diff / 80.0, -0.4, 0.4)
        
        return np.array([
            np.clip(direction_xy[0], -1.0, 1.0),
            np.clip(direction_xy[1], -1.0, 1.0),
            height_action
        ])
    
    def _compute_coverage_power(self, uav, assigned_ues, target_height) -> float:
        """High power for wide coverage"""
        # Higher altitude needs more power
        height_ratio = (target_height - self.env.uav_fly_range_h[0]) / \
                      (self.env.uav_fly_range_h[1] - self.env.uav_fly_range_h[0])

        base_power = np.sin(height_ratio * np.pi / 2) * 0.4  # Between 0.4 and 1.0

        
        return np.clip(base_power, 0, 1)
    
    def _compute_coverage_bandwidth(self, uav) -> np.ndarray:
        """Prioritize users that are harder to cover (Far distance)"""
        uav_das = [da for da in self.env.demand_areas.values() if da.uav_id == uav.id]
        
        if not uav_das:
            return np.ones(self.action_dim - 4) / (self.action_dim - 4)
        
        priorities = []
        for da in sorted(uav_das, key=lambda d: d.id):
            if len(da.user_ids) == 0:
                priorities.append(0.01)
            else:
                # Extra weight for Far users (harder to cover)
                distance_weight = {'Near': 1.0, 'Medium': 1.3, 'Far': 2.0}[da.distance_level]
                priority = len(da.user_ids) * self.env.slice_weights[da.slice_type] * distance_weight
                priorities.append(priority)
        
        allocations = np.array(priorities) / (sum(priorities) + 1e-6)
        return allocations / allocations.sum()

class QoSAwareHeightGreedyAgent(BaselineAgent):
    """
    QoS-Aware Height Strategy:
    - URLLC users → Fly lower (better SINR, lower latency)
    - eMBB users → Medium height (balance throughput and coverage)
    - mMTC users → Can fly higher (less demanding QoS)
    - Dynamically adjusts height based on slice type mix
    """
    
    def __init__(self, num_agents: int, obs_dim: int, action_dim: int, env):
        super().__init__(num_agents, obs_dim, action_dim)
        self.env = env
        
        # Preferred heights for each slice type
        self.preferred_heights = {
            'urllc': 0.3,   # 30% of height range (low altitude)
            'embb': 0.5,    # 50% of height range (medium)
            'mmtc': 0.7     # 70% of height range (high)
        }
    
    def select_actions(self, observations: Dict[int, np.ndarray], explore: bool = False) -> Dict[int, np.ndarray]:
        actions = {}
        
        for uav_id, obs in observations.items():
            action = np.zeros(self.action_dim)
            uav = self.env.uavs[uav_id]
            
            assigned_ues = [ue for ue in self.env.ues.values() 
                           if ue.assigned_uav == uav_id and ue.is_active]
            
            if not assigned_ues:
                action[3] = 0.4
                action[4:] = 1.0 / (self.action_dim - 4)
                actions[uav_id] = action
                continue
            
            # 1. Calculate QoS-aware optimal height
            target_height = self._compute_qos_aware_height(uav, assigned_ues)
            
            # 2. Position with QoS priority weighting
            position_action = self._compute_qos_position(uav, assigned_ues, target_height)
            action[0:3] = position_action
            
            # 3. Power based on QoS requirements
            action[3] = self._compute_qos_power(uav, assigned_ues, target_height)
            
            # 4. Bandwidth with QoS priorities
            action[4:] = self._compute_qos_bandwidth(uav)
            
            actions[uav_id] = action
        
        return actions
    
    def _compute_qos_aware_height(self, uav, assigned_ues) -> float:
        """Calculate height based on slice type distribution"""
        if not assigned_ues:
            return (self.env.uav_fly_range_h[0] + self.env.uav_fly_range_h[1]) / 2
        
        # Count users by slice type
        slice_counts = {'urllc': 0, 'embb': 0, 'mmtc': 0}
        for ue in assigned_ues:
            slice_counts[ue.slice_type] += 1
        
        total_users = len(assigned_ues)
        
        # Weighted average of preferred heights
        target_height_ratio = 0.0
        for slice_type, count in slice_counts.items():
            if count > 0:
                weight = count / total_users
                # Extra weight for URLLC (critical)
                if slice_type == 'urllc':
                    weight *= 2.0
                target_height_ratio += weight * self.preferred_heights[slice_type]
        
        # Normalize if URLLC was over-weighted
        target_height_ratio = min(target_height_ratio, 1.0)
        
        # Convert ratio to actual height
        height_range = self.env.uav_fly_range_h[1] - self.env.uav_fly_range_h[0]
        target_height = self.env.uav_fly_range_h[0] + height_range * target_height_ratio
        
        # Also consider beam coverage constraint
        ue_positions = np.array([ue.position[:2] for ue in assigned_ues])
        centroid = np.mean(ue_positions, axis=0)
        max_distance = max([np.linalg.norm(ue.position[:2] - centroid) for ue in assigned_ues])
        
        beam_angle_rad = np.radians(uav.beam_angle)
        min_height_for_coverage = (max_distance / np.tan(beam_angle_rad)) * 1.05 if beam_angle_rad > 0 else self.env.uav_fly_range_h[0]
        
        # Ensure coverage, but prefer QoS-optimal height
        target_height = max(target_height, min_height_for_coverage)
        
        return np.clip(target_height, self.env.uav_fly_range_h[0], self.env.uav_fly_range_h[1])
    
    def _compute_qos_position(self, uav, assigned_ues, target_height) -> np.ndarray:
        """Position prioritizing high-QoS users"""
        if not assigned_ues:
            return np.zeros(3)
        
        # Weighted centroid with QoS priority
        weighted_positions = []
        total_weight = 0.0
        
        for ue in assigned_ues:
            # URLLC gets 3x weight, eMBB gets 2x, mMTC gets 1x
            qos_weight = {'urllc': 3.0, 'embb': 2.0, 'mmtc': 1.0}[ue.slice_type]
            slice_weight = self.env.slice_weights[ue.slice_type]
            
            combined_weight = qos_weight * slice_weight
            weighted_positions.append(ue.position[:2] * combined_weight)
            total_weight += combined_weight
        
        target_xy = sum(weighted_positions) / total_weight if total_weight > 0 else uav.position[:2]
        
        # XY movement
        direction_xy = target_xy - uav.position[:2]
        distance_xy = np.linalg.norm(direction_xy)
        
        if distance_xy > 5.0:
            direction_xy = (direction_xy / distance_xy) * 0.4
        else:
            direction_xy = np.zeros(2)
        
        # Z movement (prioritize reaching QoS-optimal height)
        height_diff = target_height - uav.position[2]
        height_action = np.clip(height_diff / 70.0, -0.35, 0.35)
        
        return np.array([
            np.clip(direction_xy[0], -1.0, 1.0),
            np.clip(direction_xy[1], -1.0, 1.0),
            height_action
        ])
    
    def _compute_qos_power(self, uav, assigned_ues, target_height) -> float:
        """Power allocation for QoS satisfaction"""
        if not assigned_ues:
            return 0.4
        
        # Check if we have URLLC users (need high power)
        has_urllc = any(ue.slice_type == 'urllc' for ue in assigned_ues)
        
        # Base power higher for URLLC
        base_power = 0.8 if has_urllc else 0.65
        
        # Height adjustment
        height_ratio = (target_height - self.env.uav_fly_range_h[0]) / \
                      (self.env.uav_fly_range_h[1] - self.env.uav_fly_range_h[0])
        power_boost = 0.15 * height_ratio
        
        # Battery constraint
        battery_ratio = uav.current_battery / uav.battery_capacity
        battery_factor = 0.7 if battery_ratio < 0.25 else 1.0
        battery_factor = 0.7
        
        return np.clip((base_power + power_boost) * battery_factor, 0.5, 0.95)

    # def _compute_qos_power(self, uav, assigned_ues, target_height) -> float:
    #     """Power allocation for QoS satisfaction considering all slice types"""
    #     if not assigned_ues:
    #         return 0.4
        
    #     # Count users by slice type
    #     slice_counts = {'urllc': 0, 'embb': 0, 'mmtc': 0}
    #     for ue in assigned_ues:
    #         slice_type = ue.slice_type.lower()
    #         if slice_type in slice_counts:
    #             slice_counts[slice_type] += 1
        
    #     total_ues = len(assigned_ues)
        
    #     # Calculate slice type ratios
    #     urllc_ratio = slice_counts['urllc'] / total_ues
    #     embb_ratio = slice_counts['embb'] / total_ues
    #     mmtc_ratio = slice_counts['mmtc'] / total_ues
        
    #     # Base power levels for each slice type (normalized 0-1)
    #     # URLLC: Highest power for reliability and low latency
    #     # eMBB: High power for throughput
    #     # mMTC: Lower power (many devices, lower individual rates)
    #     power_urllc = 0.85
    #     power_embb = 0.70
    #     power_mmtc = 0.50
        
    #     # Weighted average based on slice composition
    #     base_power = (urllc_ratio * power_urllc + 
    #                 embb_ratio * power_embb + 
    #                 mmtc_ratio * power_mmtc)
        
    #     # Ensure minimum power level
    #     base_power = max(base_power, 0.45)
        
    #     # Height adjustment: Higher altitude requires more power
    #     height_ratio = (target_height - self.env.uav_fly_range_h[0]) / \
    #                 (self.env.uav_fly_range_h[1] - self.env.uav_fly_range_h[0])
    #     power_boost = 0.15 * height_ratio
        
    #     # Battery constraint: Reduce power when battery is low
    #     battery_ratio = uav.current_battery / uav.battery_capacity
    #     if battery_ratio < 0.25:
    #         battery_factor = 0.7
    #     elif battery_ratio < 0.50:
    #         battery_factor = 0.85
    #     else:
    #         battery_factor = 1.0
        
    #     # Final power calculation
    #     final_power = (base_power + power_boost) * battery_factor
        
    #     # Clip to valid range [0, 1]
    #     return max(0.0, min(1.0, final_power))
    
    def _compute_qos_bandwidth(self, uav) -> np.ndarray:
        """Bandwidth with strong QoS priorities"""
        uav_das = [da for da in self.env.demand_areas.values() if da.uav_id == uav.id]
        
        if not uav_das:
            return np.ones(self.action_dim - 4) / (self.action_dim - 4)
        
        priorities = []
        for da in sorted(uav_das, key=lambda d: d.id):
            if len(da.user_ids) == 0:
                priorities.append(0.01)
            else:
                # Strong emphasis on slice weights (URLLC >> eMBB > mMTC)
                slice_priority = self.env.slice_weights[da.slice_type] ** 2  # Squared for emphasis
                user_count = len(da.user_ids)
                distance_weight = {'Near': 1.0, 'Medium': 1.2, 'Far': 1.5}[da.distance_level]
                
                priority = user_count * slice_priority * distance_weight
                priorities.append(priority)
        
        allocations = np.array(priorities) / (sum(priorities) + 1e-6)
        
        # Ensure minimum for non-empty DAs
        for i, da in enumerate(sorted(uav_das, key=lambda d: d.id)):
            if len(da.user_ids) > 0:
                allocations[i] = max(allocations[i], 0.02)
        
        return allocations / allocations.sum()

class MAPPOWrapper(BaselineAgent):
    def __init__(self, mappo_agent: MAPPOAgent):
        self.agent = mappo_agent
        self.num_agents = mappo_agent.num_agents
        self.obs_dim = mappo_agent.obs_dim
        self.action_dim = mappo_agent.action_dim
    
    def select_actions(self, observations, explore=False):
        actions, _, _ = self.agent.select_actions(
            observations, deterministic=True
        )
        return actions
    
    def get_name(self):
        return "MAPPO"

class EvaluationFramework:
    """Framework for evaluating and comparing agents"""
    
    def __init__(self, env_config_path: str, results_dir: str = "baseline_results"):
        self.env_config_path = env_config_path
        self.env = NetworkSlicingEnv(config_path=env_config_path)
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # Create timestamp for this evaluation run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(results_dir, f"run_{self.timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)
    
    def evaluate_agent(self, agent, agent_name: str, env: NetworkSlicingEnv, 
                      max_steps: int = 500) -> Dict:
        """Evaluate an agent on a single episode with fixed environment"""
        
        print(f"\n{'='*60}")
        print(f"Evaluating {agent_name}")
        print(f"{'='*60}")
        
        # Reset environment (will use same seed as set externally)
        obs = env.reset()
        
        # Storage for step-by-step metrics
        metrics = {
            'steps': [],
            'rewards': [],
            'cumulative_rewards': [],
            'qos': [],
            'energy': [],
            'fairness': [],
            'active_ues': []
        }

        for u in range(len(env.uavs.values())):
            metrics[f'uav_{u}_pos_x'] = []
            metrics[f'uav_{u}_pos_y'] = []
            metrics[f'uav_{u}_pos_z'] = []
        
        cumulative_reward = 0
        
        for step in tqdm(range(max_steps), desc=f"{agent_name}", leave=True):
            # Select actions
            actions = agent.select_actions(obs, explore=False)
            
            # Step environment
            next_obs, reward, done, info = env.step(actions)

            # Store UAV positions
            for uav_id, uav in env.uavs.items():
                metrics[f'uav_{uav_id}_pos_x'].append(uav.position[0])
                metrics[f'uav_{uav_id}_pos_y'].append(uav.position[1])
                metrics[f'uav_{uav_id}_pos_z'].append(uav.position[2])
            
            cumulative_reward += reward
            
            # Store metrics
            metrics['steps'].append(step)
            metrics['rewards'].append(reward)
            metrics['cumulative_rewards'].append(cumulative_reward)
            metrics['qos'].append(info['qos_satisfaction'])
            metrics['energy'].append(info['energy_usage_level'])
            metrics['fairness'].append(info['fairness_level'])
            metrics['active_ues'].append(info['active_ues'])
            
            obs = next_obs
            
            if done:
                break
        
        # Compile results
        results = {
            'agent_name': agent_name,
            'metrics': metrics,
            'summary': {
                'total_reward': cumulative_reward,
                'mean_reward': np.mean(metrics['rewards']),
                'mean_qos': np.mean(metrics['qos']),
                'mean_energy': np.mean(metrics['energy']),
                'mean_fairness': np.mean(metrics['fairness']),
                'mean_active_ues': np.mean(metrics['active_ues']),
                'final_qos': metrics['qos'][-1] if metrics['qos'] else 0,
                'final_energy': metrics['energy'][-1] if metrics['energy'] else 0,
                'final_fairness': metrics['fairness'][-1] if metrics['fairness'] else 0
            }
        }
        
        # Print summary
        print(f"\n{agent_name} Results:")
        print(f"  Total Reward: {results['summary']['total_reward']:.4f}")
        print(f"  Mean Step Reward: {results['summary']['mean_reward']:.4f}")
        print(f"  Mean QoS: {results['summary']['mean_qos']:.4f}")
        print(f"  Mean Energy Efficiency: {results['summary']['mean_energy']:.4f}")
        print(f"  Mean Fairness: {results['summary']['mean_fairness']:.4f}")
        print(f"  Steps Completed: {len(metrics['steps'])}")
        
        return results
    
    def compare_agents(self, results_dir: str):
        """Generate comparison plots for multiple agents"""

        # Find all CSV files in the directory
        csv_files = glob.glob(os.path.join(results_dir, '*.csv'))
        
        if not csv_files:
            raise ValueError(f"No CSV files found in {results_dir}")
        
        print(f"Found {len(csv_files)} agent result files")
        
        # Load data from each CSV file
        results_list = []
        
        for csv_path in csv_files:
            # Extract agent name from filename
            filename = os.path.basename(csv_path)
            agent_name = filename.replace('.csv', '').replace('_', ' ')
            
            print(f"  Loading: {agent_name}")
            
            # Read CSV file
            try:
                df = pd.read_csv(csv_path)
                
                # Extract metrics from dataframe
                metrics = {
                    'steps': df['step'].values,
                    'rewards': df['reward'].values,
                    'cumulative_rewards': df['cumulative_reward'].values,
                    'qos': df['qos'].values,
                    'energy': df['energy'].values,
                    'fairness': df['fairness'].values,
                    'active_ues': df['active_ues'].values
                }
                
                # Calculate summary statistics
                summary = {
                    'total_reward': float(metrics['cumulative_rewards'][-1]) if len(metrics['cumulative_rewards']) > 0 else 0.0,
                    'mean_qos': float(np.mean(metrics['qos'])),
                    'mean_energy': float(np.mean(metrics['energy'])),
                    'mean_fairness': float(np.mean(metrics['fairness'])),
                    'mean_active_ues': float(np.mean(metrics['active_ues']))
                }
                
                # Create result dictionary
                result = {
                    'agent_name': agent_name,
                    'metrics': metrics,
                    'summary': summary
                }
                
                results_list.append(result)
                
                print(f"    ✓ Loaded {len(metrics['steps'])} steps")
                print(f"    Total reward: {summary['total_reward']:.2f}")
                
            except Exception as e:
                print(f"    ⚠️  Error loading {filename}: {e}")
                continue
        
        if not results_list:
            raise ValueError("No valid agent results could be loaded")
        
        # Sort by agent name for consistent ordering
        results_list.sort(key=lambda x: x['agent_name'])
        
        print(f"\n✓ Successfully loaded {len(results_list)} agents")
        print(f"Generating comparison plots...")
        
        print(f"\nGenerating comparison plots...")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(20, 11))
        fig.suptitle('Agent Performance Comparison - Single Episode', fontsize=16, fontweight='bold')
        
        # Extract data
        agent_names = [r['agent_name'] for r in results_list]
        # Use standard highlighting colors: blue, green, red, orange, purple, etc.
        standard_colors = ['#1f77b4', '#2ca02c', '#d62728', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        colors = standard_colors[:len(agent_names)]
        
        # 1. Step Rewards
        ax = axes[0, 0]
        for result, color in zip(results_list, colors):
            steps = result['metrics']['steps']
            rewards = result['metrics']['rewards']
            # Smooth with moving average
            window = min(20, len(rewards) // 10) if len(rewards) > 20 else 1
            if window > 1:
                smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
                ax.plot(steps[:len(smoothed)], smoothed, label=result['agent_name'], 
                       color=color, linewidth=2.5)
                # Plot raw data with transparency
                ax.plot(steps, rewards, color=color, alpha=0.2, linewidth=1)
            else:
                ax.plot(steps, rewards, label=result['agent_name'], 
                       color=color, linewidth=2.5)
        ax.set_title('Step Rewards', fontsize=12, fontweight='bold')
        ax.set_xlabel('Step')
        ax.set_ylabel('Reward')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # 2. Cumulative Rewards
        ax = axes[0, 1]
        for result, color in zip(results_list, colors):
            steps = result['metrics']['steps']
            cumulative_rewards = result['metrics']['cumulative_rewards']
            ax.plot(steps, cumulative_rewards, label=result['agent_name'], 
                   color=color, linewidth=2.5)
        ax.set_title('Cumulative Rewards', fontsize=12, fontweight='bold')
        ax.set_xlabel('Step')
        ax.set_ylabel('Cumulative Reward')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # 3. QoS Satisfaction
        ax = axes[0, 2]
        for result, color in zip(results_list, colors):
            steps = result['metrics']['steps']
            qos = result['metrics']['qos']
            # Smooth
            window = min(20, len(qos) // 10) if len(qos) > 20 else 1
            if window > 1:
                smoothed = np.convolve(qos, np.ones(window)/window, mode='valid')
                ax.plot(steps[:len(smoothed)], smoothed, label=result['agent_name'], 
                       color=color, linewidth=2.5)
                ax.plot(steps, qos, color=color, alpha=0.2, linewidth=1)
            else:
                ax.plot(steps, qos, label=result['agent_name'], 
                       color=color, linewidth=2.5)
        ax.set_title('QoS Satisfaction', fontsize=12, fontweight='bold')
        ax.set_xlabel('Step')
        ax.set_ylabel('QoS Satisfaction')
        ax.set_ylim([0, 1.05])
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # 4. Energy Efficiency
        ax = axes[1, 0]
        for result, color in zip(results_list, colors):
            steps = result['metrics']['steps']
            energy = result['metrics']['energy']
            # Smooth
            window = min(20, len(energy) // 10) if len(energy) > 20 else 1
            if window > 1:
                smoothed = np.convolve(energy, np.ones(window)/window, mode='valid')
                ax.plot(steps[:len(smoothed)], smoothed, label=result['agent_name'], 
                       color=color, linewidth=2.5)
                ax.plot(steps, energy, color=color, alpha=0.2, linewidth=1)
            else:
                ax.plot(steps, energy, label=result['agent_name'], 
                       color=color, linewidth=2.5)
        ax.set_title('Energy Consumption Rate', fontsize=12, fontweight='bold')
        ax.set_xlabel('Step')
        ax.set_ylabel('Energy Consumption (Lower is Better)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # 5. Fairness Index
        ax = axes[1, 1]
        for result, color in zip(results_list, colors):
            steps = result['metrics']['steps']
            fairness = result['metrics']['fairness']
            # Smooth
            window = min(20, len(fairness) // 10) if len(fairness) > 20 else 1
            if window > 1:
                smoothed = np.convolve(fairness, np.ones(window)/window, mode='valid')
                ax.plot(steps[:len(smoothed)], smoothed, label=result['agent_name'], 
                       color=color, linewidth=2.5)
                ax.plot(steps, fairness, color=color, alpha=0.2, linewidth=1)
            else:
                ax.plot(steps, fairness, label=result['agent_name'], 
                       color=color, linewidth=2.5)
        ax.set_title('Fairness Index', fontsize=12, fontweight='bold')
        ax.set_xlabel('Step')
        ax.set_ylabel('Fairness')
        ax.set_ylim([0, 1.05])
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # 6. Summary Bar Chart
        ax = axes[1, 2]
        metrics_names = ['Total\nReward', 'Mean\nQoS', 'Mean\nEnergy', 'Mean\nFairness']
        x = np.arange(len(metrics_names))
        width = 0.8 / len(results_list)
        
        for i, (result, color) in enumerate(zip(results_list, colors)):
            # Normalize metrics for comparison
            total_rewards = [r['summary']['total_reward'] for r in results_list]
            max_reward = max(total_rewards) if max(total_rewards) > 0 else 1
            
            energies = [r['summary']['mean_energy'] for r in results_list]
            
            values = [
                result['summary']['total_reward'] / max_reward,
                result['summary']['mean_qos'],
                result['summary']['mean_energy'],  # Invert energy
                result['summary']['mean_fairness']
            ]
            
            offset = (i - len(results_list)/2 + 0.5) * width
            ax.bar(x + offset, values, width, label=result['agent_name'], 
                  color=color, alpha=0.8)
        
        ax.set_ylabel('Normalized Performance')
        ax.set_title('Performance Summary (Normalized)', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_names)
        ax.set_ylim([0, 1.1])
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3, axis='y')

        # Movement Energy


        
        plt.tight_layout()
        
        # Save figure
        plot_path = os.path.join(self.run_dir, 'comparison.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison plot to {plot_path}")
        
        return fig
    
    def save_results(self, results_list: List[Dict]):
        """Save results to JSON file"""
        # Convert results for JSON serialization (exclude large arrays)
        serializable_results = []
        for result in results_list:
            serializable = {
                'agent_name': result['agent_name'],
                'summary': result['summary'],
                'num_steps': len(result['metrics']['steps'])
            }
            serializable_results.append(serializable)
        
        results_path = os.path.join(self.run_dir, 'results_summary.json')
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Saved results summary to {results_path}")
        
        # Save detailed metrics as CSV for each agent
        for result in results_list:
            agent_name = result['agent_name'].replace(' ', '_').replace('(', '').replace(')', '')
            csv_path = os.path.join(self.run_dir, f'{agent_name}.csv')
            title_list = ['step', 'reward', 'cumulative_reward', 'qos', 'energy', 'fairness', 'active_ues']
            for u in range(len(self.env.uavs.values())):
                title_list.append(f'uav_{u}_pos_x')
                title_list.append(f'uav_{u}_pos_y')
                title_list.append(f'uav_{u}_pos_z')
            
            import csv
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(title_list)
                
                metrics = result['metrics']
                for i in range(len(metrics['steps'])):
                    n_uavs = len(list(self.env.uavs.values()))
                    row = [
                        metrics['steps'][i],
                        metrics['rewards'][i],
                        metrics['cumulative_rewards'][i],
                        metrics['qos'][i],
                        metrics['energy'][i],
                        metrics['fairness'][i],
                        metrics['active_ues'][i],
                    ]
                    for u in range(n_uavs):
                        row.extend([
                            metrics[f'uav_{u}_pos_x'][i],
                            metrics[f'uav_{u}_pos_y'][i],
                            metrics[f'uav_{u}_pos_z'][i],
                        ])
                    writer.writerow(row)
            
            print(f"Saved detailed metrics to {csv_path}")

    def plot_trajectory(self, run_dir: str):
        """
        Plot UAV trajectories from saved CSV files with color gradients showing temporal progression.
        Each subplot is saved as a separate image file.
        
        Args:
            run_dir: Directory containing the CSV files with trajectory data
        """
        import pandas as pd
        import matplotlib.pyplot as plt
        import numpy as np
        import os
        import glob
        
        # Find all CSV files (excluding trajectory-specific files if they exist)
        csv_files = glob.glob(os.path.join(run_dir, '*.csv'))
        csv_files = [f for f in csv_files if 'trajectory' not in os.path.basename(f)]
        
        if not csv_files:
            print(f"No CSV files found in {run_dir}")
            return
        
        print(f"Found {len(csv_files)} baseline result files")
        
        # Base colors for different UAVs
        base_colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
        
        # Create figure with subplots for each baseline
        num_baselines = len(csv_files)
        fig, axes = plt.subplots(1, num_baselines, figsize=(7*num_baselines, 6))
        if num_baselines == 1:
            axes = [axes]
        
        fig.suptitle('UAV Trajectories - Color Intensity Shows Temporal Progression', 
                    fontsize=16, fontweight='bold')
        
        # Store distance information for summary
        distance_summary = {}
        
        for idx, csv_path in enumerate(csv_files):
            # Extract baseline name from filename
            baseline_name = os.path.basename(csv_path).replace('.csv', '').replace('_', ' ')
            print(f"\nProcessing: {baseline_name}")
            
            # Read CSV
            df = pd.read_csv(csv_path)
            # Extract 500 steps from df
            df = df.head(500)
            
            # Find UAV columns (columns ending with _pos_x, _pos_y, _pos_z)
            uav_ids = set()
            for col in df.columns:
                if '_pos_x' in col:
                    uav_id = col.replace('_pos_x', '')
                    uav_ids.add(uav_id)
            
            uav_ids = sorted(list(uav_ids))
            print(f"  Found {len(uav_ids)} UAVs: {uav_ids}")
            
            ax = axes[idx]
            
            # Track total distance for this baseline
            total_distance = 0.0
            uav_distances = {}
            
            # Plot each UAV's trajectory with color gradient
            for uav_idx, uav_id in enumerate(uav_ids):
                # Extract positions
                x_col = f'{uav_id}_pos_x'
                y_col = f'{uav_id}_pos_y'
                
                x = df[x_col].values
                y = df[y_col].values
                
                # Calculate distance traveled by this UAV
                distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
                uav_distance = np.sum(distances)
                uav_distances[uav_id] = uav_distance
                total_distance += uav_distance
                
                # Create line segments for gradient coloring
                points = np.array([x, y]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                
                # Create color gradient (from pale to strong)
                n_points = len(x)
                base_color = base_colors[uav_idx % len(base_colors)]
                
                # Create LineCollection with varying alpha/color intensity
                alphas = np.linspace(0.1, 1.0, n_points - 1)  # From pale to strong
                
                # Plot using segments with varying alpha
                for i in range(len(segments)):
                    alpha = alphas[i]
                    ax.plot([segments[i][0][0], segments[i][1][0]], 
                        [segments[i][0][1], segments[i][1][1]], 
                        color=base_color, alpha=alpha, linewidth=2)
                
                # Mark start position (pale)
                ax.scatter(x[0], y[0], color=base_color, marker='o', s=150, 
                        alpha=0.4, edgecolors='black', linewidths=2, zorder=5)
                
                # Mark end position (strong)
                ax.scatter(x[-1], y[-1], color=base_color, marker='s', s=150, 
                        alpha=1.0, edgecolors='black', linewidths=2, zorder=5,
                        label=f'{uav_id}')
                
                print(f"    ✓ Plotted trajectory for {uav_id}: {len(x)} points")
            
            # Configure subplot
            ax.set_xlabel('X (m)', fontsize=11)
            ax.set_ylabel('Y (m)', fontsize=11)
            ax.set_title(baseline_name, fontsize=12, fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
            ax.set_xlim(0, 2000)
            ax.set_ylim(0, 2000)
            
            # Store distance information
            distance_summary[baseline_name] = {
                'total': total_distance,
                'per_uav': uav_distances
            }
            
            # Print distance information for this baseline
            print(f"  Total distance traveled by all UAVs: {total_distance:.2f} m")
            for uav_id, dist in uav_distances.items():
                print(f"    - {uav_id}: {dist:.2f} m")
        
        plt.tight_layout()
        
        # Save each subplot separately
        for idx, ax in enumerate(axes):
            # Create a new figure for each subplot
            fig_single = plt.figure(figsize=(7, 6))
            ax_single = fig_single.add_subplot(111)
            
            # Re-read and plot data for this subplot
            csv_path = csv_files[idx]
            baseline_name = os.path.basename(csv_path).replace('.csv', '').replace('_', ' ')
            
            df = pd.read_csv(csv_path)
            df = df.head(500)
            
            # Find UAV columns
            uav_ids = set()
            for col in df.columns:
                if '_pos_x' in col:
                    uav_id = col.replace('_pos_x', '')
                    uav_ids.add(uav_id)
            uav_ids = sorted(list(uav_ids))
            
            # Plot trajectories
            for uav_idx, uav_id in enumerate(uav_ids):
                x_col = f'{uav_id}_pos_x'
                y_col = f'{uav_id}_pos_y'
                
                x = df[x_col].values
                y = df[y_col].values
                
                points = np.array([x, y]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                
                n_points = len(x)
                base_color = base_colors[uav_idx % len(base_colors)]
                alphas = np.linspace(0.1, 1.0, n_points - 1)
                
                for i in range(len(segments)):
                    alpha = alphas[i]
                    ax_single.plot([segments[i][0][0], segments[i][1][0]], 
                        [segments[i][0][1], segments[i][1][1]], 
                        color=base_color, alpha=alpha, linewidth=2)
                
                ax_single.scatter(x[0], y[0], color=base_color, marker='o', s=150, 
                        alpha=0.4, edgecolors='black', linewidths=2, zorder=5)
                
                ax_single.scatter(x[-1], y[-1], color=base_color, marker='s', s=150, 
                        alpha=1.0, edgecolors='black', linewidths=2, zorder=5,
                        label=f'{uav_id}')
            
            ax_single.set_xlabel('X (m)', fontsize=11)
            ax_single.set_ylabel('Y (m)', fontsize=11)
            ax_single.set_title(baseline_name, fontsize=12, fontweight='bold')
            ax_single.legend(loc='best')
            ax_single.grid(True, alpha=0.3)
            ax_single.set_aspect('equal')
            ax_single.set_xlim(0, 2000)
            ax_single.set_ylim(0, 2000)
            
            # Save individual subplot
            output_path = os.path.join(run_dir, f'trajectory_{idx+1}.pdf')
            plt.tight_layout()
            fig_single.savefig(output_path, format='pdf', bbox_inches='tight')
            print(f"✓ Saved subplot {idx+1} to {output_path}")
            plt.close(fig_single)
        
        # Save combined figure
        output_path_combined = os.path.join(run_dir, 'trajectories_colored.pdf')
        fig.savefig(output_path_combined, format='pdf', bbox_inches='tight')
        print(f"\n✓ Saved combined trajectory plot to {output_path_combined}")
        
        # Print summary comparison
        print("\n" + "="*60)
        print("DISTANCE SUMMARY - COMPARISON ACROSS ALL METHODS")
        print("="*60)
        for method_name, distances in distance_summary.items():
            print(f"\n{method_name}:")
            print(f"  Total distance: {distances['total']:.2f} m")
            print(f"  Average per UAV: {distances['total']/len(distances['per_uav']):.2f} m")
        print("\n" + "="*60)
        
        plt.show()



def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare baseline agents')
    parser.add_argument('--env_config', type=str, default='config/environment/default.yaml',
                       help='Path to environment config')
    parser.add_argument('--checkpoint_maddpg', type=str, default='saved_models_maddpg/model13/checkpoints/checkpoint_step_1000000.pth',
                       help='Path to trained MADDPG model checkpoint')
    parser.add_argument('--checkpoint_mappo', type=str, default='saved_models/mappo/model4/checkpoints/checkpoint_step_1000000.pth',
                       help='Path to trained MAPPO model checkpoint')
    parser.add_argument('--steps', type=int, default=2000,
                       help='Number of steps for evaluation')
    parser.add_argument('--seed', type=int, default=43,
                       help='Random seed for reproducibility')
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    
    # Load environment config to get dimensions
    env_config = Configuration(args.env_config)
    num_agents = env_config.system.num_uavs
    
    # Create environment (will be reset with same seed for each agent)
    env = NetworkSlicingEnv(config_path=args.env_config)
    
    # Get observation/action dimensions
    obs_sample = env.reset()
    obs_dim = len(list(obs_sample.values())[0])
    action_dim = 4 + env_config.system.num_das_per_slice * 3
    
    print(f"\n{'='*60}")
    print(f"Environment Setup (Seed: {args.seed})")
    print(f"{'='*60}")
    print(f"  Num Agents: {num_agents}")
    print(f"  Observation Dim: {obs_dim}")
    print(f"  Action Dim: {action_dim}")
    print(f"  Evaluation Steps: {args.steps}")
    print(f"{'='*60}\n")
    
    # Initialize evaluation framework
    evaluator = EvaluationFramework(env_config_path=args.env_config)
    
    # Initialize agents
    agents = []
    
    random_agent = RandomAgent(num_agents, obs_dim, action_dim)
    agents.append((random_agent, "Random"))

    qos_greedy_agent = QoSAwareHeightGreedyAgent(num_agents, obs_dim, action_dim, env)
    agents.append((qos_greedy_agent, "QoS-Aware Greedy"))

    coverage_greedy_agent = CoverageMaximizationGreedyAgent(num_agents, obs_dim, action_dim, env)
    agents.append((coverage_greedy_agent, "Coverage Greedy"))


    # MADDPG agent (if checkpoint provided)
    if args.checkpoint_maddpg:
        maddpg_agent = MADDPGAgent(
            num_agents=num_agents,
            obs_dim=obs_dim,
            action_dim=action_dim,
            training=False
        )
        maddpg_agent.load_models(args.checkpoint_maddpg)
        agents.append((maddpg_agent, "MADDPG"))
        print(f"✓ Loaded MADDPG model from {args.checkpoint_maddpg}\n")

    if args.checkpoint_mappo:
        mappo_agent = MAPPOAgent(
            num_agents=num_agents,
            obs_dim=obs_dim,
            action_dim=action_dim,
            training=False
        )
        mappo_agent.load_models(args.checkpoint_mappo)
        mappo_agent_wrapper = MAPPOWrapper(mappo_agent)
        agents.append((mappo_agent_wrapper, "MAPPO"))
        print(f"✓ Loaded MAPPO model from {args.checkpoint_mappo}\n")

    # Evaluate all agents on the same environment
    results_list = []
    for agent, agent_name in agents:
        # Reset environment with same seed for fair comparison
        np.random.seed(args.seed)
        
        results = evaluator.evaluate_agent(
            agent, 
            agent_name,
            env=env,
            max_steps=args.steps
        )
        results_list.append(results)
    
    # # Save results
    evaluator.save_results(results_list)

    # Generate comparison plots
    evaluator.compare_agents(evaluator.run_dir)
    # evaluator.compare_agents("baseline_results/run_20251025_171841")
    
    print(f"\n{'='*60}")
    print("Evaluation Complete!")
    print(f"Results saved to: {evaluator.run_dir}")
    print(f"{'='*60}")

def plot(path: str):

    evaluator = EvaluationFramework(env_config_path='config/environment/default.yaml')
    evaluator.compare_agents(path)

def trajectory_plot(path: str):

    evaluator = EvaluationFramework(env_config_path='config/environment/default.yaml')
    evaluator.plot_trajectory(path)

if __name__ == "__main__":
    main()
    # plot('baseline_results/run_20251028_121654')
    # trajectory_plot('baseline_results/run_20251208_174702')