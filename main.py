import json
import yaml
import time
import argparse
from pathlib import Path

from db_config import connect_db

from train_mappo import TrainingManagerMAPPO

from type import EnvConfig, KBEntry, EnvVector, EvalVector
from kb_handler import append_entry, query_similar_entries
from llm import generate_response
from env_sampling import sample_env_config, update_env_config_file, sample_env_eval
from prompt_builder import build_prompt_fs, build_prompt_kb, build_prompt_zs
from semantic_analyzer import _get_env_pref

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

def _build_save_path(mode, eval_samples, seed):
    stamp = time.strftime("%Y%m%d")
    if eval_samples > 0:
        return f"logs/{mode}_{eval_samples}_{seed}.jsonl"
    else:
        return f"logs/{mode}_{seed}_{stamp}.jsonl"

# Report selected mode
def _report_args(mode, eval_samples, seed):
    mode_desc = {
        "zs": "Zero-Shot Prompting",
        "fs": "Few-Shot Prompting with Static Examples",
        "kb": "Knowledge Base Retrieval",
        "none": "No LLM Analysis (Static Reward Weights)"
    }
    print(f"Selected Mode: {mode_desc.get(mode, 'Unknown Mode')}")
    if eval_samples > 0:
        print(f"Evaluation mode enabled: Running {eval_samples} samples")
    else:
        print("Evaluation mode disabled: Running indefinitely until manually stopped")
    print(f"Random Seed: {seed}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["zs", "fs", "kb", "none"],
        default="kb",
        help="zs: zero-shot, fs: few-shot static examples, kb: knowledge-base retrieval, none: no LLM analysis, static reward weights"
    )
    parser.add_argument(
        "--eval",
        type=int,
        default=0,
        help="Run only N samples for evaluation"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    args = parser.parse_args()
    _report_args(args.mode, args.eval, args.seed)

    # Configs
    train_config_file = 'config/train/mappo_default.yaml'  
    env_config_file = "config/environment/adaptive.yaml"
    if args.mode == 'none':
        env_config_file = "config/environment/default.yaml"
    config_tracker = ConfigTracker(env_config_file)

    # Sample evaluation environments if eval mode is enabled
    eval_envs = []
    if args.eval > 0:
        _print_section(f"Sampling {args.eval} samples")
        eval_envs = sample_env_eval(seed=args.seed, n_samples=args.eval)
        print(f"Sampled {len(eval_envs)} evaluation environments:\n{eval_envs}")

    # Database connection for KB mode

    # Main loop 
    while True:
        env_config = None
        if args.eval > 0:
            if eval_envs:
                env_config = eval_envs.pop(0)
            else:
                break
        else: 
            env_config = sample_env_config()
        
        if env_config:
            update_env_config_file(env_config, env_config_file)
        else:
            ValueError("env: None Value")

        if config_tracker._check_env_changes():
            _print_section("Environment Change Detected")
            config_tracker.env_config = config_tracker._load_env(env_config_file)
            env_obj = config_tracker._extract_env_obj()
            env_list = env_obj.__dict__()
            print(f"Current Environment: {env_list}")

            env = env_list
            if args.mode != "none":
                _print_section("Semantic Analysis & KB Retrieval")
                env = _get_env_pref(env_list)
                
                prompt = ""
                if args.mode == "zs":
                    prompt = build_prompt_zs(env)
                elif args.mode == "fs":
                    prompt = build_prompt_fs(env)
                elif args.mode == "kb":
                    conn = connect_db()
                    selected_exps = query_similar_entries(conn, EnvVector(env), EvalVector(env), top_k=3)
                    print(f"Retrieved {len(selected_exps)} similar experiences from KB for LLM analysis:\n{selected_exps}")
                    prompt = build_prompt_kb(env, selected_exps)
                
                _print_section("LLM Analysis")
                start_time = time.time()
                llm_res = generate_response(prompt)
                print(f"LLM finished in {time.time() - start_time:.2f} seconds.\n{llm_res}")
                
                config_tracker._update_reward_weights(llm_res)
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
                **env,
                **report,
                "score": score
            }
            if args.mode != "none":
                log = {
                    **env,
                    **llm_res,
                    **report,
                    "score": score
                }
            print(f"{log}")

            # Save log
            _print_section("Saving Log")
            save_path = _build_save_path(args.mode, args.eval, args.seed)
            _save_log(log, save_path)

            # Update knowledge base
            if args.mode == "kb":
                _print_section("Updating Knowledge Base")
                entry = KBEntry(log)
                print(f"Updating knowledge base: {entry}")
                conn = connect_db()
                append_entry(conn, entry)

            
            _print_section(f"Cycle Complete. Waiting for next environment change...")
            # time.sleep(10)  
            
        
if __name__ == "__main__":
    main()