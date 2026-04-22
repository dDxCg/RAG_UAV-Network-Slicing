import numpy as np
from environment import NetworkSlicingEnv
from agents_mappo import MAPPOAgent  
import os
from tqdm import tqdm
import re
from utils import Configuration

class TrainingManagerMAPPO:
    """Manages the MAPPO training process with periodic visualizer toggle checks"""

    def __init__(self, env_config_path: str = None, train_config_path: str = None, 
                 checkpoint_path: str = None, enable_visualization: bool = False,
                 check_interval: int = 10):
        """
        Control files (create in project directory):
          - 'viz_on.flag'  : Turn visualization ON
          - 'viz_off.flag' : Turn visualization OFF
          - 'quit.flag'    : Quit training gracefully
        
        Args:
            check_interval: Check for control files every N steps
        """
        # Load configurations using your Configuration class
        env_config = Configuration(env_config_path)
        train_config = Configuration(train_config_path)

        self.config = {
            # Environment parameters
            'num_uavs': env_config.system.num_uavs,
            'num_ues': env_config.system.num_ues,
            'service_area': tuple(env_config.system.service_area),
            'uav_fly_range_h': tuple(env_config.system.uav_fly_range_h),
            'num_das_per_slice': env_config.system.num_das_per_slice,

            # Step-based training parameters
            'total_training_steps': train_config.total_training_steps,

            # Step-based intervals
            'log_interval': train_config.log_interval,
            'save_interval': train_config.save_interval,
            'evaluation_interval': train_config.evaluation_interval,
            'plot_interval': train_config.plot_interval,

            # Paths
            'save_dir': train_config.save_dir,
            
            # Config paths
            'env_config_path': env_config_path,
            'train_config_path': train_config_path,

            # MAPPO-specific: Rollout collection parameters
            'n_steps': getattr(train_config, 'n_steps', 2048),  # Rollout length
            'min_buffer_size': getattr(train_config, 'min_buffer_size', 64),  # Min for training
        }

        # Initialize environment
        self.env = NetworkSlicingEnv(config_path=self.config['env_config_path'])

        # Create model saving directory
        self.model_dir = self._get_next_model_dir(self.config['save_dir'])
        self._create_model_folder()

        # Save config files
        self._save_config_files()

        # Initialize metrics storage
        self._initialize_metrics()
        
        # Get observation and action dimensions
        obs_sample = self.env.reset()
        obs_dim = len(list(obs_sample.values())[0])
        act_dim = 4 + self.config['num_das_per_slice'] * 3  # pos + power + bandwidth per DA
        print(f"Observation dimension: {obs_dim}")
        print(f"Action dimension: {act_dim}")

        # ============================================
        # Initialize MAPPO Agent (Changed!)
        # ============================================
        self.MAPPOagent = MAPPOAgent(
            num_agents=self.config['num_uavs'],
            obs_dim=obs_dim,
            action_dim=act_dim,
            training=True,
            config_path="./config/agents/mappo_default.yaml"  # ← MAPPO config
        )

        # Load checkpoint if provided
        if checkpoint_path:
            self.MAPPOagent.load_models(checkpoint_path)
            print(f"Loaded checkpoint from {checkpoint_path}")

        # Visualization        
        self.enable_visualization = enable_visualization
        self.visualizer = None
        self.check_interval = check_interval

        # Control file paths
        self.viz_on_flag = 'viz_on.flag'
        self.viz_off_flag = 'viz_off.flag'
        self.quit_flag = 'quit.flag'
        
        # Clean up any existing control files
        self._cleanup_control_files()

        self._print_instructions()
        
    def _print_instructions(self):
        """Print control instructions"""
        print("\n" + "="*70)
        print("  FILE-BASED CONTROLS:")
        print("  To toggle visualization ON:  touch viz_on.flag")
        print("  To toggle visualization OFF: touch viz_off.flag")
        print("  To quit training:            touch quit.flag")
        print("")
        print("  Example from another terminal:")
        print("    cd /path/to/your/project")
        print("    touch viz_on.flag    # Turn on visualization")
        print("    touch viz_off.flag   # Turn off visualization")
        print("    touch quit.flag      # Stop training")
        print(f"\n  Files are checked every {self.check_interval} steps")
        print("="*70 + "\n")
    
    def _cleanup_control_files(self):
        """Remove any existing control files"""
        for flag in [self.viz_on_flag, self.viz_off_flag, self.quit_flag]:
            if os.path.exists(flag):
                os.remove(flag)

    def _check_control_files(self):
        """Check for control file commands"""
        # Check for quit command
        if os.path.exists(self.quit_flag):
            print("\nQuit flag detected, stopping training...")
            os.remove(self.quit_flag)
            return 'quit'
        
        # Check for visualization toggle
        if os.path.exists(self.viz_on_flag):
            if not self.enable_visualization:
                print("\nVisualization ON flag detected, enabling...")
                self._initialize_visualizer()
            os.remove(self.viz_on_flag)
        
        if os.path.exists(self.viz_off_flag):
            if self.enable_visualization:
                print("\n📴 Visualization OFF flag detected, disabling...")
                self._cleanup_visualizer()
            os.remove(self.viz_off_flag)
        
        return None

    def _create_model_folder(self):
        os.makedirs(self.config['save_dir'], exist_ok=True)

        model_dir = self.model_dir
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(os.path.join(model_dir, 'training_progress'), exist_ok=True)
        os.makedirs(os.path.join(model_dir, 'checkpoints'), exist_ok=True)

    def _initialize_metrics(self):  
        # Initialize tracking variables for step-based training
        self.step_metrics = {
            'steps': [],
            'rewards': [],
            'qos_satisfaction': [],
            'energy_usage_level': [],
            'fairness_level': [],
            'active_ues': [],
            'avg_throughput_sat': [],
            'avg_delay_sat': [],
            'avg_reliability_sat': [],
            'handovers': []
        }

        # Training metrics (adjusted for MAPPO)
        self.training_metrics = {
            'steps': [],              # Steps when training occurred
            'actor_losses': [],       # Average actor loss per training session
            'critic_losses': [],      # Average critic loss per training session
            'policy_losses': [],      # PPO policy loss
            'value_losses': [],       # Value function loss
            'entropy': [],            # Policy entropy
            'approx_kl': [],          # Approximate KL divergence
            'clip_fraction': [],      # PPO clip fraction
            'buffer_size': []         # Rollout buffer size
        }

        # Initialize per-agent action metrics
        for i in range(self.config['num_uavs']):
            self.step_metrics[f'agent_{i}_pos_x'] = []
            self.step_metrics[f'agent_{i}_pos_y'] = []
            self.step_metrics[f'agent_{i}_pos_z'] = []
            self.step_metrics[f'agent_{i}_power'] = []
            for j in range(int(len(self.env.demand_areas) / self.config['num_uavs'])):
                self.step_metrics[f'agent_{i}_da_{j}'] = []

    def _get_next_model_dir(self, save_dir):
        """Find the next available model directory name (model1, model2, ...)"""
        os.makedirs(save_dir, exist_ok=True)
        existing = [d for d in os.listdir(save_dir) if os.path.isdir(os.path.join(save_dir, d))]
        numbers = []
        for name in existing:
            m = re.match(r'model(\d+)', name)
            if m:
                numbers.append(int(m.group(1)))
        next_num = 1
        while f"model{next_num}" in existing:
            next_num += 1
        return os.path.join(save_dir, f"model{next_num}")

    def _save_config_files(self):
        """Save copies of the config files used for this training run"""
        import shutil
        shutil.copy(self.config['env_config_path'], os.path.join(self.model_dir, 'env_config.yaml'))
        shutil.copy(self.config['train_config_path'], os.path.join(self.model_dir, 'train_config.yaml'))
        shutil.copy('./config/agents/mappo_default.yaml', os.path.join(self.model_dir, 'agents_config.yaml'))
        print(f"Saved config files to {self.model_dir}")

    def train(self):
        """
        Main MAPPO training loop - ROLLOUT-BASED
        """
        print(f"\nStarting MAPPO training for {self.config['total_training_steps']} steps")
        print(f"Saving to: {self.model_dir}")
        print(f"Rollout length: {self.MAPPOagent.n_steps} steps\n")
        
        observations = self.env.reset()
        global_step = 0
        episode_reward = 0
        episode_length = 0
        
        progress_bar = tqdm(total=self.config['total_training_steps'], desc="Training Steps")
        
        last_save_step = 0
        last_plot_step = 0

        #Local list to track rewards

        while global_step < self.config['total_training_steps']:
            # ============================================
            # PHASE 1: COLLECT ROLLOUT (n_steps)
            # ============================================
            rollout_steps = 0
            
            while rollout_steps < self.MAPPOagent.n_steps and global_step < self.config['total_training_steps']:
                # Periodically check control files
                if global_step % self.check_interval == 0:
                    if self._check_control_files() == 'quit':
                        progress_bar.close()
                        return
                
                # ============================================
                # Select actions (MAPPO returns 3 values!)
                # ============================================
                actions, log_probs, value = self.MAPPOagent.select_actions(
                    observations, 
                    deterministic=False  # Stochastic for training
                )

                # Environment step
                next_observations, reward, done, info = self.env.step(actions)
                
                # ============================================
                # Store transition (different signature!)
                # ============================================
                self.MAPPOagent.store_transition(
                    observations=observations,
                    actions=actions,
                    log_probs=log_probs,  # ← Not greedy_actions!
                    reward=reward,
                    done=done,           # ← No next_observations here!
                    value=value          # ← Need value estimate!
                )
                
                # Store environment metrics
                self._store_step_metrics(global_step, actions, reward, info)
                
                # Update visualization if enabled
                # if self.enable_visualization and self.visualizer is not None:
                #     self._update_visualization()

                # Update tracking
                episode_reward += reward
                episode_length += 1
                rollout_steps += 1
                global_step += 1
                progress_bar.update(1)
                
                observations = next_observations
                
                # Reset if episode ends during rollout
                if done:
                    observations = self.env.reset()
                    episode_reward = 0
                    episode_length = 0
            
            # ============================================
            # PHASE 2: TRAIN ON COLLECTED ROLLOUT
            # ============================================
            if len(self.MAPPOagent.buffer) >= self.config['min_buffer_size']:
                # Train (pass next_observations for bootstrapping!)
                metrics = self.MAPPOagent.train(next_observations=observations)
                
                if metrics:
                    # Store training metrics (different format!)
                    self.training_metrics['steps'].append(global_step)
                    self.training_metrics['actor_losses'].append(np.mean(metrics['actor_losses']))
                    self.training_metrics['critic_losses'].append(metrics['critic_loss'])
                    self.training_metrics['policy_losses'].append(metrics.get('policy_loss', 0))
                    self.training_metrics['value_losses'].append(metrics.get('value_loss', 0))
                    self.training_metrics['entropy'].append(metrics.get('entropy', 0))
                    self.training_metrics['approx_kl'].append(metrics.get('approx_kl', 0))
                    self.training_metrics['clip_fraction'].append(metrics.get('clip_fraction', 0))
                    self.training_metrics['buffer_size'].append(len(self.MAPPOagent.buffer))
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        'Reward': f'{reward:.3f}',
                        'Buffer': f'{len(self.MAPPOagent.buffer)}',
                        'Actor Loss': f'{np.mean(metrics["actor_losses"]):.4f}',
                        'Policy Loss': f'{metrics.get("policy_loss", 0):.4f}',
                        'Entropy': f'{metrics.get("entropy", 0):.4f}',
                        'KL': f'{metrics.get("approx_kl", 0):.4f}',
                    })
            
            # ============================================
            # PERIODIC OPERATIONS
            # ============================================
            print(global_step)
            if global_step - last_save_step > self.config['save_interval'] and global_step > 0:
                self._save_checkpoint(global_step)
                self._save_metrics_to_csv()
                last_save_step = global_step
                print(f"Saved checkpoint at step {global_step}")

            
            if global_step - last_plot_step > self.config['plot_interval'] and global_step > 0:
                self._plot_training_progress(global_step)
                last_plot_step = global_step
                print(f"Generated plots at step {global_step}")

        progress_bar.close()
        
        # Final save
        self._save_checkpoint(global_step)
        self._save_metrics_to_csv()
        self._plot_training_progress(global_step)

        return self._final_report()

    def _final_report(self):
        import pandas as pd
        import os
        step_metrics_path = f'{self.model_dir}/step_metrics.csv'
        if not os.path.exists(step_metrics_path):
            print(f"Warning: {step_metrics_path} not found")
            return {
                "reward": -1e6,
                "qos": -1e6,
                "energy_usage": -1e6,
                "fairness": -1e6
            }

        step_df = pd.read_csv(step_metrics_path)
        steps = step_df['steps'].values

        reward = step_df['rewards'].tail(1000).mean()
        qos = step_df['qos_satisfaction'].tail(1000).mean()
        energy = step_df['energy_usage_level'].tail(1000).mean()
        fairness = step_df['fairness_level'].tail(1000).mean()

        return {
            "reward": reward,
            "qos": qos,
            "energy_usage": energy,
            "fairness": fairness
        }
        

    def _store_step_metrics(self, step, actions, reward, info):
        """Store metrics for this step"""
        self.step_metrics['steps'].append(step)
        self.step_metrics['rewards'].append(reward)
        self.step_metrics['qos_satisfaction'].append(info['qos_satisfaction'])
        self.step_metrics['energy_usage_level'].append(info['energy_usage_level'])
        self.step_metrics['fairness_level'].append(info['fairness_level'])
        self.step_metrics['active_ues'].append(info['active_ues'])
        self.step_metrics['avg_throughput_sat'].append(self.env.stats.get('avg_throughput_sat', 0))
        self.step_metrics['avg_delay_sat'].append(self.env.stats.get('avg_delay_sat', np.inf))
        self.step_metrics['avg_reliability_sat'].append(self.env.stats.get('avg_reliability_sat', 0))
        self.step_metrics['handovers'].append(self.env.stats.get('handovers', 0))

        if f'movement_energy' not in self.step_metrics:
            self.step_metrics[f'movement_energy'] = []
            self.step_metrics[f'transmission_energy'] = []
        movement_energy = np.mean([uav.energy_used['movement'] for uav in self.env.uavs.values()])
        transmission_energy = np.mean([uav.energy_used['transmission'] for uav in self.env.uavs.values()])
        self.step_metrics[f'movement_energy'].append(movement_energy)
        self.step_metrics[f'transmission_energy'].append(transmission_energy)

        # Store actions of each agent for analysis
        for i, action in actions.items():
            self.step_metrics[f'agent_{i}_pos_x'].append(action[0])
            self.step_metrics[f'agent_{i}_pos_y'].append(action[1])
            self.step_metrics[f'agent_{i}_pos_z'].append(action[2])
            self.step_metrics[f'agent_{i}_power'].append(action[3])
            for j in range(int(len(self.env.demand_areas) / self.config['num_uavs'])):
                self.step_metrics[f'agent_{i}_da_{j}'].append(action[4 + j])

    def _plot_training_progress(self, current_step):
        """Plot comprehensive training progress (adapted for MAPPO)"""
        import matplotlib.pyplot as plt
        import pandas as pd
        import os
        
        fig, axes = plt.subplots(4, 5, figsize=(20, 12))
        fig.suptitle(f'MAPPO Training Progress - Step {current_step}', fontsize=16)
        
        # Read data from CSV files
        step_metrics_path = f'{self.model_dir}/step_metrics.csv'
        training_metrics_path = f'{self.model_dir}/training_metrics.csv'
        
        if not os.path.exists(step_metrics_path):
            print(f"Warning: {step_metrics_path} not found")
            plt.close()
            return
        
        # Load step metrics
        step_df = pd.read_csv(step_metrics_path)
        steps = step_df['steps'].values
        
        # Load training metrics if exists
        training_df = None
        if os.path.exists(training_metrics_path):
            training_df = pd.read_csv(training_metrics_path)
        
        # Plot 1: Reward progression
        rewards = step_df['rewards'].values
        axes[0, 0].plot(steps, rewards, alpha=0.3, color='blue')
        if len(steps) > 100:
            smoothed_rewards = self._moving_average(rewards, 1000)
            axes[0, 0].plot(steps[-len(smoothed_rewards):], smoothed_rewards, 'b-', linewidth=2)
        axes[0, 0].set_title('Training Reward')
        axes[0, 0].set_xlabel('Steps')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True)
        
        # Plot 2: QoS satisfaction
        qos_satisfaction = step_df['qos_satisfaction'].values
        axes[0, 1].plot(steps, qos_satisfaction, 'g-', alpha=0.6)
        if len(steps) > 100:
            smoothed_qos = self._moving_average(qos_satisfaction, 1000)
            axes[0, 1].plot(steps[-len(smoothed_qos):], smoothed_qos, 'darkgreen', linewidth=2)
        axes[0, 1].set_title('QoS Satisfaction')
        axes[0, 1].set_xlabel('Steps')
        axes[0, 1].set_ylabel('QoS Satisfaction')
        axes[0, 1].grid(True)
        
        # Plot 3: Energy efficiency
        energy_usage_level = step_df['energy_usage_level'].values
        axes[0, 2].plot(steps, energy_usage_level, 'orange', alpha=0.6)
        if len(steps) > 100:
            smoothed_energy = self._moving_average(energy_usage_level, 1000)
            axes[0, 2].plot(steps[-len(smoothed_energy):], smoothed_energy, 'darkorange', linewidth=2)
        axes[0, 2].set_title('Energy Consumption Rate')
        axes[0, 2].set_xlabel('Steps')
        axes[0, 2].set_ylabel('Energy Consumption Rate')
        axes[0, 2].grid(True)
        
        # Plot 4: Active UEs
        active_ues = step_df['active_ues'].values
        axes[0, 3].plot(steps, active_ues, 'purple', alpha=0.6)
        if len(steps) > 100:
            smoothed_ues = self._moving_average(active_ues, 1000)
            axes[0, 3].plot(steps[-len(smoothed_ues):], smoothed_ues, 'darkviolet', linewidth=2)
        axes[0, 3].set_title('Active UEs')
        axes[0, 3].set_xlabel('Steps')
        axes[0, 3].set_ylabel('Number of UEs')
        axes[0, 3].grid(True)

        # Plot 5: Actor losses
        if training_df is not None and 'actor_losses' in training_df.columns:
            training_steps = training_df['steps'].values
            actor_losses = training_df['actor_losses'].values
            axes[0, 4].plot(training_steps, actor_losses, 
                        'red', alpha=0.6, marker='o', markersize=3)
            axes[0, 4].set_title('Actor Losses')
            axes[0, 4].set_xlabel('Steps')
            axes[0, 4].set_ylabel('Loss')
            axes[0, 4].grid(True)
        
        # Plot 6: Critic losses
        if training_df is not None and 'critic_losses' in training_df.columns:
            training_steps = training_df['steps'].values
            critic_losses = training_df['critic_losses'].values
            axes[1, 0].plot(training_steps, critic_losses, 
                        'blue', alpha=0.6, marker='o', markersize=3)
            axes[1, 0].set_title('Critic Losses (Value Function)')
            axes[1, 0].set_xlabel('Steps')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].grid(True)

        # Plot 7: Fairness Level
        fairness_level = step_df['fairness_level'].values
        axes[1, 1].plot(steps, fairness_level, 'brown', alpha=0.6)
        if len(steps) > 100:
            smoothed_fairness = self._moving_average(fairness_level, 1000)
            axes[1, 1].plot(steps[-len(smoothed_fairness):], smoothed_fairness, 'saddlebrown', linewidth=2)
        axes[1, 1].set_title('Average Fairness Level')
        axes[1, 1].set_xlabel('Steps')
        axes[1, 1].set_ylabel('Average Fairness Level')
        axes[1, 1].grid(True)

        # Plot 8: Policy Entropy (MAPPO-specific!)
        if training_df is not None and 'entropy' in training_df.columns:
            training_steps = training_df['steps'].values
            entropy = training_df['entropy'].values
            axes[1, 2].plot(training_steps, entropy, 'cyan', alpha=0.6, marker='o', markersize=3)
            axes[1, 2].set_title('Policy Entropy (Exploration)')
            axes[1, 2].set_xlabel('Steps')
            axes[1, 2].set_ylabel('Entropy')
            axes[1, 2].grid(True)

        # Plot 9: Power usage of agents
        for i in range(self.config['num_uavs']):
            col_name = f'agent_{i}_power'
            if col_name in step_df.columns:
                agent_power = step_df[col_name].values
                axes[1, 3].plot(steps, agent_power, alpha=0.6, label=f'Agent {i}')
        axes[1, 3].set_title('Power Usage of Agents')
        axes[1, 3].set_xlabel('Steps')
        axes[1, 3].set_ylabel('Power Usage (Watts)')
        axes[1, 3].legend(loc='lower left')
        axes[1, 3].grid(True)

        # Plot 10: PPO Clip Fraction (MAPPO-specific!)
        if training_df is not None and 'clip_fraction' in training_df.columns:
            training_steps = training_df['steps'].values
            clip_fraction = training_df['clip_fraction'].values
            axes[1, 4].plot(training_steps, clip_fraction, 'purple', alpha=0.6, marker='o', markersize=3)
            axes[1, 4].set_title('PPO Clip Fraction')
            axes[1, 4].set_xlabel('Steps')
            axes[1, 4].set_ylabel('Clip Fraction')
            axes[1, 4].axhline(y=0.1, color='r', linestyle='--', alpha=0.3, label='Healthy Range')
            axes[1, 4].axhline(y=0.3, color='r', linestyle='--', alpha=0.3)
            axes[1, 4].legend()
            axes[1, 4].grid(True)

        # Plot 11: Policy Loss vs Value Loss (MAPPO-specific!)
        if training_df is not None and 'policy_losses' in training_df.columns:
            training_steps = training_df['steps'].values
            policy_losses = training_df['policy_losses'].values
            value_losses = training_df['value_losses'].values
            
            ax_twin = axes[2, 0].twinx()
            line1 = axes[2, 0].plot(training_steps, policy_losses, 'red', alpha=0.6, label='Policy Loss')
            line2 = ax_twin.plot(training_steps, value_losses, 'blue', alpha=0.6, label='Value Loss')
            
            axes[2, 0].set_title('Policy Loss vs Value Loss')
            axes[2, 0].set_xlabel('Steps')
            axes[2, 0].set_ylabel('Policy Loss', color='red')
            ax_twin.set_ylabel('Value Loss', color='blue')
            axes[2, 0].tick_params(axis='y', labelcolor='red')
            ax_twin.tick_params(axis='y', labelcolor='blue')
            
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            axes[2, 0].legend(lines, labels, loc='upper right')
            axes[2, 0].grid(True)

        # Plot 12: Average Throughput Satisfaction
        if 'avg_throughput_sat' in step_df.columns:
            avg_throughput_sat = step_df['avg_throughput_sat'].values
            axes[2, 1].plot(steps, avg_throughput_sat, 'magenta', alpha=0.6)
            if len(steps) > 100:
                smoothed_throughput = self._moving_average(avg_throughput_sat, 1000)
                axes[2, 1].plot(steps[-len(smoothed_throughput):], smoothed_throughput, 'darkmagenta', linewidth=2)
            axes[2, 1].set_title('Average Throughput Satisfaction')
            axes[2, 1].set_xlabel('Steps')
            axes[2, 1].set_ylabel('Throughput Satisfaction')
            axes[2, 1].grid(True)

        # Plot 13: Average Delay Satisfaction
        if 'avg_delay_sat' in step_df.columns:
            avg_delay_sat = step_df['avg_delay_sat'].values
            axes[2, 2].plot(steps, avg_delay_sat, 'teal', alpha=0.6)
            if len(steps) > 100:
                smoothed_delay = self._moving_average(avg_delay_sat, 1000)
                axes[2, 2].plot(steps[-len(smoothed_delay):], smoothed_delay, 'darkcyan', linewidth=2)
            axes[2, 2].set_title('Average Delay Satisfaction')
            axes[2, 2].set_xlabel('Steps')
            axes[2, 2].set_ylabel('Delay Satisfaction')
            axes[2, 2].grid(True)
        
        # Plot 14: Average Reliability Satisfaction
        if 'avg_reliability_sat' in step_df.columns:
            avg_reliability_sat = step_df['avg_reliability_sat'].values
            axes[2, 3].plot(steps, avg_reliability_sat, 'olive', alpha=0.6)
            if len(steps) > 100:
                smoothed_reliability = self._moving_average(avg_reliability_sat, 1000)
                axes[2, 3].plot(steps[-len(smoothed_reliability):], smoothed_reliability, 'darkolivegreen', linewidth=2)
            axes[2, 3].set_title('Average Reliability Satisfaction')
            axes[2, 3].set_xlabel('Steps')
            axes[2, 3].set_ylabel('Reliability Satisfaction')
            axes[2, 3].grid(True)

        # Plot 15: Approximate KL Divergence (MAPPO-specific!)
        if training_df is not None and 'approx_kl' in training_df.columns:
            training_steps = training_df['steps'].values
            approx_kl = training_df['approx_kl'].values
            axes[2, 4].plot(training_steps, approx_kl, 'sienna', alpha=0.6, marker='o', markersize=3)
            axes[2, 4].set_title('Approximate KL Divergence')
            axes[2, 4].set_xlabel('Steps')
            axes[2, 4].set_ylabel('KL Divergence')
            axes[2, 4].axhline(y=0.01, color='r', linestyle='--', alpha=0.3, label='Target KL')
            axes[2, 4].legend()
            axes[2, 4].grid(True)

        # Plot 16: Movement Energy
        if 'movement_energy' in step_df.columns:
            movement_energy = step_df['movement_energy'].values
            axes[3, 0].plot(steps, movement_energy, 'navy', alpha=0.6)
            if len(steps) > 100:
                smoothed_movement_energy = self._moving_average(movement_energy, 10)
                axes[3, 0].plot(steps[-len(smoothed_movement_energy):], smoothed_movement_energy, 'midnightblue', linewidth=2)
            axes[3, 0].set_title('Total Movement Energy')
            axes[3, 0].set_xlabel('Steps')
            axes[3, 0].set_ylabel('Energy (Joules)')
            axes[3, 0].grid(True)

        # Plot 17: Transmission Energy
        if 'transmission_energy' in step_df.columns:
            transmission_energy = step_df['transmission_energy'].values
            axes[3, 1].plot(steps, transmission_energy, 'darkred', alpha=0.6)
            if len(steps) > 100:
                smoothed_transmission_energy = self._moving_average(transmission_energy, 10)
                axes[3, 1].plot(steps[-len(smoothed_transmission_energy):], smoothed_transmission_energy, 'firebrick', linewidth=2)
            axes[3, 1].set_title('Total Transmission Energy')
            axes[3, 1].set_xlabel('Steps')
            axes[3, 1].set_ylabel('Energy (Joules)')
            axes[3, 1].grid(True)

        # Plot 18: Handovers
        if 'handovers' in step_df.columns:
            handovers = step_df['handovers'].values
            axes[3, 2].plot(steps, handovers, 'forestgreen', alpha=0.6)
            if len(steps) > 100:
                smoothed_handovers = self._moving_average(handovers, 1000)
                axes[3, 2].plot(steps[-len(smoothed_handovers):], smoothed_handovers, 'darkgreen', linewidth=2)
            axes[3, 2].set_title('Number of Handovers')
            axes[3, 2].set_xlabel('Steps')
            axes[3, 2].set_ylabel('Handovers')
            axes[3, 2].grid(True)

        plt.tight_layout()
        
        # Ensure directory exists
        progress_dir = f'{self.model_dir}/training_progress'
        os.makedirs(progress_dir, exist_ok=True)
        
        plt.savefig(f'{progress_dir}/mappo_training_progress_step_{current_step}.png', dpi=150, bbox_inches='tight')
        plt.close()

    def _moving_average(self, data, window_size):
        """Calculate moving average"""
        if len(data) < window_size:
            return data
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

    def _save_checkpoint(self, step):
        """Save model checkpoint"""
        checkpoint_path = os.path.join(self.model_dir, 'checkpoints', f'checkpoint_step_{step}.pth')
        self.MAPPOagent.save_models(checkpoint_path)
        print(f"Checkpoint saved at step {step}")

    def _save_metrics_to_csv(self):
        """Save metrics to CSV files"""
        import csv
        import os
        
        def save_dict_of_lists_to_csv(metrics_dict, csv_path):
            """Helper function to save dict of lists structure to CSV"""
            if not metrics_dict or not metrics_dict.get('steps'):
                return False
            
            try:
                # Validate all lists have same length
                first_key = next(iter(metrics_dict))
                expected_length = len(metrics_dict[first_key])
                
                for key, values in metrics_dict.items():
                    if len(values) != expected_length:
                        if len(values) < expected_length:
                            values.extend([None] * (expected_length - len(values)))
                        else:
                            metrics_dict[key] = values[:expected_length]
                
                # Convert to list of dicts
                records = []
                for i in range(expected_length):
                    record = {key: values[i] for key, values in metrics_dict.items()}
                    records.append(record)
                
                # Write to CSV
                mode = 'a' if os.path.exists(csv_path) else 'w'
                with open(csv_path, mode, newline='') as csvfile:
                    fieldnames = list(metrics_dict.keys())
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    
                    if mode == 'w':
                        writer.writeheader()
                    
                    writer.writerows(records)
                
                return True
                
            except Exception as e:
                print(f"Error saving metrics to {csv_path}: {e}")
                return False
        
        # Save step metrics
        if save_dict_of_lists_to_csv(self.step_metrics, f'{self.model_dir}/step_metrics.csv'):
            for key in self.step_metrics:
                self.step_metrics[key].clear()
        
        # Save training metrics
        if save_dict_of_lists_to_csv(self.training_metrics, f'{self.model_dir}/training_metrics.csv'):
            for key in self.training_metrics:
                self.training_metrics[key].clear()

def main():
    env_config_file = "./config/environment/adaptive.yaml"
    train_config_file = './config/train/mappo_default.yaml'
    checkpoint_file = None
    # checkpoint_file = "saved_models/mappo_model1/checkpoints/checkpoint_step_100000.pth"
    
    trainer = TrainingManagerMAPPO(
        env_config_path=env_config_file,
        train_config_path=train_config_file,
        checkpoint_path=checkpoint_file
    )

    trainer.train()

if __name__ == "__main__":
    main()