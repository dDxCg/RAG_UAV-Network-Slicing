import yaml

from type import EnvConfig
import random
from decimal import Decimal

def sample_env_config():
    points = sorted(random.sample(range(101), 2))
    segments = [
        points[0],               
        points[1] - points[0],   
        100 - points[1]          
    ]
    random.shuffle(segments)
    embb_prob = float(Decimal(segments[0]) / Decimal(100))
    urllc_prob = float(Decimal(segments[1]) / Decimal(100))
    mmtc_prob = float(Decimal(segments[2]) / Decimal(100))
    arrival_rate = round(random.uniform(0.1, 3.0), 2)
    departure_rate = round(random.uniform(0.1, 2.0), 2)

    return EnvConfig(
        embb_prob=embb_prob,
        urllc_prob=urllc_prob,
        mmtc_prob=mmtc_prob,
        arrival_rate=arrival_rate,
        departure_rate=departure_rate
    )

def update_env_config_file(env_config: EnvConfig, file_path: str):
    with open(file_path, 'r', encoding='utf-8') as f:
        env_dict = yaml.safe_load(f)

    env_dict['slicing']['slice_probabilities']['embb'] = env_config.embb_prob
    env_dict['slicing']['slice_probabilities']['urllc'] = env_config.urllc_prob
    env_dict['slicing']['slice_probabilities']['mmtc'] = env_config.mmtc_prob
    env_dict['ue_dynamics']['arrival_rate'] = env_config.arrival_rate
    env_dict['ue_dynamics']['departure_rate'] = env_config.departure_rate

    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.dump(env_dict, f)

if __name__ == "__main__":
    for _ in range(5):
        config = sample_env_config()
        print(config)
