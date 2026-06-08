import json
import yaml
from pathlib import Path


from train_mappo import TrainingManagerMAPPO

from type import EnvConfig
from env_sampling import update_env_config_file

class ConfigTracker:
    def __init__(self, env_config_file):
        self.env_config_file = env_config_file
        self.env_config = self._load_env(env_config_file)
        
    def _load_env(self, env_config_file):
        with open(env_config_file, 'r', encoding='utf-8') as f:
            env_config = yaml.safe_load(f)
        return env_config

    def _check_env_changes(self):
        """Check for changes in the environment config file."""
        new_env_config = self._load_env(self.env_config_file)
        return self.env_config != new_env_config

    def _extract_env_obj(self):
        env_obj = EnvConfig(
            embb_prob=self.env_config['slicing']['slice_probabilities']['embb'],
            urllc_prob=self.env_config['slicing']['slice_probabilities']['urllc'],
            mmtc_prob=self.env_config['slicing']['slice_probabilities']['mmtc'],
            walk_prob=self.env_config['ue_dynamics']['patterns']['walk']['probability'],
            vehicle_prob=self.env_config['ue_dynamics']['patterns']['vehicle']['probability'],
            stationary_prob=self.env_config['ue_dynamics']['patterns']['stationary']['probability'],
            arrival_rate=self.env_config['ue_dynamics']['arrival_rate'],
            departure_rate=self.env_config['ue_dynamics']['departure_rate']
        )
        return env_obj

    def _update_reward_weights(self, results: dict):
        self.env_config['reward_weights']['qos'] = results['w_qos']
        self.env_config['reward_weights']['energy'] = results['w_energy']
        self.env_config['reward_weights']['fairness'] = results['w_fairness']
        with open(self.env_config_file, 'w', encoding='utf-8') as f:
            yaml.dump(self.env_config, f)

    def _get_weights(self):
        return {
            "w_qos": self.env_config['reward_weights']['qos'],
            "w_energy": self.env_config['reward_weights']['energy'],
            "w_fairness": self.env_config['reward_weights']['fairness']
        }
    
def _print_section(title):
    print("\n" + f" {title} ".center(100, "=") + "\n")

def _save_log(log: dict, filepath):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(json.dumps(log, ensure_ascii=False) + "\n")

def main():

    # Configs
    train_config_file = 'config/train/mappo_default.yaml'  
    env_config_file = "config/environment/adaptive.yaml"
    config_tracker = ConfigTracker(env_config_file)

    import pandas as pd
    df = pd.read_csv('logs/v2_ml_data_predictions.csv')
    eval_envs = df.to_dict(orient='records')
    print(eval_envs)

    # Main loop 
    while True:
        env_row = None
        env_config = None
        if eval_envs:
            env_row = eval_envs.pop(0)
            env_config = EnvConfig(
                embb_prob=env_row['embb_prob'],
                urllc_prob=env_row['urllc_prob'],
                mmtc_prob=env_row['mmtc_prob'],
                walk_prob=env_row['walk_prob'],
                vehicle_prob=env_row['vehicle_prob'],
                stationary_prob=env_row['stationary_prob'],
                arrival_rate=env_row['arrival_rate'],
                departure_rate=env_row['departure_rate']
            )
        else:
            break
        
        if env_config:
            update_env_config_file(env_config, env_config_file)
        else:
            raise ValueError("env: None Value")

        if config_tracker._check_env_changes():
            _print_section("Environment Change Detected")
            config_tracker.env_config = config_tracker._load_env(env_config_file)
            env_obj = config_tracker._extract_env_obj()
            env_list = env_obj.__dict__()
            print(f"Current Environment: {env_list}")

            reward_weights = {
                "w_qos": env_row["w_qos"],
                "w_energy": env_row["w_energy"],
                "w_fairness": env_row["w_fairness"]
            }
            
            config_tracker._update_reward_weights(reward_weights)
            config_tracker.env_config = config_tracker._load_env(env_config_file)

            _print_section("Training MAPPO")
            trainer = TrainingManagerMAPPO(
                env_config_path=env_config_file, 
                train_config_path=train_config_file,
                checkpoint_path=None
            )
            report = trainer.train()
            reward = report["reward"]

            weights = config_tracker._get_weights()
            score = report["qos"] - report["energy_usage"] + report["fairness"]
            
            log = {
                **env_row,
                **report,
                "score": score
            }
            print(f"{log}")

            # Save log
            _print_section("Saving Log")
            _save_log(log, "logs/ml_logs")

            
            _print_section(f"Cycle Complete. Waiting for next environment change...")  
            
        
if __name__ == "__main__":
    main()