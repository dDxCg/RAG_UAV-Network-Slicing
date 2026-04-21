from train_mappo import TrainingManagerMAPPO
import yaml
from type import EnvConfig, KBEntry, EnvVector, EvalVector
from kb_handler import append_entry, query_similar_entries
from llm import generate_response
from env_sampling import sample_env_config, update_env_config_file
from prompt_builder import build_prompt_fs, build_prompt_kb, build_prompt_zs
from semantic_analyzer import numerical_analysis
import time
from db_config import connect_db

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
    
def _print_section(title):
    print("\n" + f" {title} ".center(100, "=") + "\n")

def main():
    conn = connect_db()
    env_config_file = "config/environment/adaptive.yaml"
    train_config_file = 'config/train/mappo_default.yaml'  
    checkpoint_file = None

    config_tracker = ConfigTracker(env_config_file)

    while True:
        # Sampling environment configs
        env = sample_env_config()
        update_env_config_file(env, env_config_file)
        if config_tracker._check_env_changes():
            _print_section("Environment Change Detected")
            config_tracker.env_config = config_tracker._load_env(env_config_file)
            env_obj = config_tracker._extract_env_obj()
            env_list = env_obj.__to_dict__()
            print(f"Current Environment: {env_list}")

            _print_section("Semantic Analysis & KB Retrieval")
            env = numerical_analysis(env_list)

            selected_exps = query_similar_entries(conn, EnvVector(env), EvalVector(env), top_k=3)
            print(f"Retrieved {len(selected_exps)} similar experiences from KB for LLM analysis:\n{selected_exps}")
            
            prompt = build_prompt_kb(env, selected_exps)
            
            _print_section("LLM Analysis")
            start_time = time.time()
            results = generate_response(prompt)
            print(f"LLM finished in {time.time() - start_time:.2f} seconds.\n{results}")
            
            config_tracker._update_reward_weights(results)
            config_tracker.env_config = config_tracker._load_env(env_config_file)

            _print_section("Training MAPPO")
            # trainer = TrainingManagerMAPPO(
            #     env_config_path=env_config_file, 
            #     train_config_path=train_config_file,
            #     checkpoint_path=checkpoint_file
            # )
            # reward = trainer.train()
            
            log = {
                **env,
                **results,
            }
            # Update knowledge base
            _print_section("Updating Knowledge Base")
            entry = KBEntry(log)
            print(f"Updating knowledge base: {entry}")
            # append_entry(conn, entry)
            
            _print_section("Cycle Complete. Waiting for next environment change...")
            time.sleep(60)  
            
        
if __name__ == "__main__":
    main()