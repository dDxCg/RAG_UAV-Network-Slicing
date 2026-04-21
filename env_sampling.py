import numpy as np
import yaml

from type import EnvConfig


def sample_env_config():
    p = np.random.dirichlet([1, 1, 1]).tolist()
    q = np.random.dirichlet([1, 1, 1]).tolist()
    a = float(np.random.uniform(0.1, 5.0))
    d = float(np.random.uniform(0.1, a))

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

if __name__ == "__main__":
    for _ in range(5):
        config = sample_env_config()
        update_env_config_file(config, "config/environment/adaptive.yaml")
        print(config)
