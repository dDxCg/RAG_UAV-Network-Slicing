from train_mappo import TrainingManagerMAPPO
import yaml
from type import EnvConfig
import kb_handling

class ConfigTracker:
    def __init__(self, env_config_file):
        self.env_config_file = env_config_file
        self.env_config = None
        
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
            embbb_prob=self.env_config['slicing']['slice_probabilities']['embb'],
            urllc_prob=self.env_config['slicing']['slice_probabilities']['urllc'],
            mmtc_prob=self.env_config['slicing']['slice_probabilities']['mmtc'],
            arrival_rate=self.env_config['ue_dynamics']['arrival_rate'],
            departure_rate=self.env_config['ue_dynamics']['departure_rate'],
            w_qos=self.env_config['reward_weights']['qos'],
            w_energy=self.env_config['reward_weights']['energy'],
            w_fairness=self.env_config['reward_weights']['fairness']
        )
        return env_obj

def main():
    env_config_file = "config/environment/adaptive.yaml"
    train_config_file = 'config/train/mappo_default.yaml'  
    checkpoint_file = None

    config_tracker = ConfigTracker(env_config_file)

    while True:
        # TODO: sampling environment configs
        if config_tracker._check_env_changes():
            print("Environment updated. Retraining with new configuration...")
            config_tracker.env_config = config_tracker._load_env(env_config_file)
            env_obj = config_tracker._extract_env_obj()

            # TODO: LLM-Guided reward weights
            

            # TODO: Update reward weights

            trainer = TrainingManagerMAPPO(
                env_config_path=env_config_file, 
                train_config_path=train_config_file,
                checkpoint_path=checkpoint_file
            )

            reward = trainer.train()

            # TODO: semantic analyze 

            # Update knowledge base
            reward = round(float(reward), 2)
            kb_handling.add_to_kb(env_obj, reward)
            print (f"Added to KB: {env_obj} with reward {reward}")
        

if __name__ == "__main__":
    main()