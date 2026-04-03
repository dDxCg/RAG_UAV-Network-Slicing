from train_mappo import TrainingManagerMAPPO
import yaml
from type import EnvConfig, KBEntry
from kb_handling import add_to_kb, retrieve_candidates
from llm import suggest_reward_weights

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
            embbb_prob=self.env_config['slicing']['slice_probabilities']['embb'],
            urllc_prob=self.env_config['slicing']['slice_probabilities']['urllc'],
            mmtc_prob=self.env_config['slicing']['slice_probabilities']['mmtc'],
            arrival_rate=self.env_config['ue_dynamics']['arrival_rate'],
            departure_rate=self.env_config['ue_dynamics']['departure_rate'],
        )
        return env_obj

    def _update_reward_weights(self, weights: dict):
        self.env_config['reward_weights']['qos'] = weights['w_qos']
        self.env_config['reward_weights']['energy'] = weights['w_energy']
        self.env_config['reward_weights']['fairness'] = weights['w_fairness']
        with open(self.env_config_file, 'w', encoding='utf-8') as f:
            yaml.dump(self.env_config, f)
    
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
            selected_exps = retrieve_candidates(env_obj)
            suggested_weights = suggest_reward_weights(env_obj, selected_exps)
            
            # TODO: Update reward weights
            config_tracker._update_reward_weights(suggested_weights)
            config_tracker.env_config = config_tracker._load_env(env_config_file)

            trainer = TrainingManagerMAPPO(
                env_config_path=env_config_file, 
                train_config_path=train_config_file,
                checkpoint_path=checkpoint_file
            )

            reward = trainer.train()

            # TODO: semantic analyze 
            semantic_thinking = "Placeholder"

            # Update knowledge base
            kb_entry = KBEntry(
                env_config = env_obj,
                w_qos = config_tracker.env_config['reward_weights']['qos'],
                w_energy = config_tracker.env_config['reward_weights']['energy'],
                w_fairness = config_tracker.env_config['reward_weights']['fairness'],
                reward = round(float(reward), 2) if reward is not None else -1e9,
                thinking = semantic_thinking  
            )
            add_to_kb(kb_entry)
            
        
if __name__ == "__main__":
    main()