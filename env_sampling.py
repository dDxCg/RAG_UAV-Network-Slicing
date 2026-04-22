import numpy as np
import yaml

from type import EnvConfig


def sample_env_config(rng=None):
    rng = rng or np.random.default_rng()
    p = rng.dirichlet([1, 1, 1]).tolist()
    q = rng.dirichlet([1, 1, 1]).tolist()
    a = float(rng.uniform(0.1, 5.0))
    d = float(rng.uniform(0.1, a))

    return EnvConfig(
        embb_prob=q[0],
        urllc_prob=q[1],
        mmtc_prob=q[2],
        walk_prob=p[0],
        vehicle_prob=p[1],
        stationary_prob=p[2],
        arrival_rate=a,
        departure_rate=d
    )

def update_env_config_file(env_config: EnvConfig, file_path: str):
    with open(file_path, 'r', encoding='utf-8') as f:
        env_dict = yaml.safe_load(f)

    env_dict['slicing']['slice_probabilities']['embb'] = env_config.embb_prob
    env_dict['slicing']['slice_probabilities']['urllc'] = env_config.urllc_prob
    env_dict['slicing']['slice_probabilities']['mmtc'] = env_config.mmtc_prob
    env_dict['ue_dynamics']['arrival_rate'] = env_config.arrival_rate
    env_dict['ue_dynamics']['departure_rate'] = env_config.departure_rate
    env_dict['ue_dynamics']['patterns']['walk']['probability'] = env_config.walk_prob
    env_dict['ue_dynamics']['patterns']['vehicle']['probability'] = env_config.vehicle_prob
    env_dict['ue_dynamics']['patterns']['stationary']['probability'] = env_config.stationary_prob

    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.dump(env_dict, f)
    return 

def sample_env_eval(seed, n_samples=10):
    rng = np.random.default_rng(seed)
    eval_envs = []
    for _ in range(n_samples):
        env_config = sample_env_config(rng)
        eval_envs.append(env_config)
    return eval_envs


if __name__ == "__main__":
    run_1 = sample_env_eval(42)
    run_2 = sample_env_eval(42)
    print (run_1.pop()) 
