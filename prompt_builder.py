from jinja2 import Environment, FileSystemLoader

def build_prompt_zs(env: dict):
    jinja_env = Environment(loader=FileSystemLoader('.'))
    template = jinja_env.get_template('prompt_zs.md')
    return template.render(env)

def build_prompt_fs(env: dict):
    jinja_env = Environment(loader=FileSystemLoader('.'))
    template = jinja_env.get_template('prompt_fs.md')
    return template.render(env)

def build_prompt_kb(env: dict, examples: list):
    jinja_env = Environment(loader=FileSystemLoader('.'))
    template = jinja_env.get_template('prompt_kb.md')
    return template.render(env, examples=examples)

if __name__ == "__main__":
    env = {"arrival_rate": 2.5, "departure_rate": 2.5, 
           "walk_prob": 0.33, "vehicle_prob": 0.33, "staionary_prob": 0.34, 
           "embb_prob": 0.33, "urllc_prob": 0.33, "mmtc_prob": 0.34, 
           "qos_demand": 0.332, 
           "dynamic_ratio": 1.0, "dynamic_stress": 0.1655,  
           "m_stress": 0.331, 
           "raw_pressure": 0.4975, "pressure": 0.0002, 
           "raw_jain": 0.7599, "jain": 0.9998, "jain_m": 0.6203, "jain_s": 0.9308}
    
    test_examples = [
    {
        "embb_prob": 0.33, "urllc_prob": 0.33, "mmtc_prob": 0.34,
        "walk_prob": 0.33, "vehicle_prob": 0.33, "stationary_prob": 0.34,
        "arrival_rate": 2.5, "departure_rate": 2.5,
        "pressure": 0.0002, "slice_stress": 0.332, "mobility_stress": 0.331, "dynamic_stress": 0.1655,
        "jain": 0.9998, "jain_s": 0.9308, "jain_m": 0.6203,
        "w_qos": 6.0, "w_energy": 3.0, "w_fairness": 1.0,
        "reward": 3.05, "energy_penalty": 1.9996, "fairness_penalty": 0.0002, "score": 1.05,
        "thinking": "System is idle with low pressure. Prioritized QoS and Energy conservation since fairness is already near perfect."
    },
    {
        "embb_prob": 0.50, "urllc_prob": 0.25, "mmtc_prob": 0.25,
        "walk_prob": 0.10, "vehicle_prob": 0.80, "stationary_prob": 0.10,
        "arrival_rate": 4.8, "departure_rate": 1.2,
        "pressure": 0.7521, "slice_stress": 0.821, "mobility_stress": 0.912, "dynamic_stress": 0.654,
        "jain": 0.4215, "jain_s": 0.512, "jain_m": 0.389,
        "w_qos": 2.0, "w_energy": 2.0, "w_fairness": 6.0,
        "reward": -1.42, "energy_penalty": 0.85, "fairness_penalty": 4.52, "score": -5.92,
        "thinking": "High pressure and high mobility stress detected due to vehicle density. Fairness is critically low; shifting weight to w_fairness to stabilize the network."
    },
    {
        "embb_prob": 0.10, "urllc_prob": 0.80, "mmtc_prob": 0.10,
        "walk_prob": 0.45, "vehicle_prob": 0.10, "stationary_prob": 0.45,
        "arrival_rate": 3.0, "departure_rate": 3.0,
        "pressure": 0.1500, "slice_stress": 0.210, "mobility_stress": 0.150, "dynamic_stress": 0.120,
        "jain": 0.8850, "jain_s": 0.850, "jain_m": 0.820,
        "w_qos": 8.0, "w_energy": 1.0, "w_fairness": 1.0,
        "reward": 4.85, "energy_penalty": 0.25, "fairness_penalty": 0.12, "score": 4.48,
        "thinking": "URLLC dominant slice requires maximum QoS. Minimal stress and high fairness allow for aggressive w_qos weighting with low energy/fairness overhead."
    }
]
    prompt = build_prompt_kb(env, test_examples)
    print(prompt)