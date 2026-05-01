from jinja2 import Environment, FileSystemLoader

fs_examples = [
    {"walk_prob": 0.15, "vehicle_prob": 0.6, "stationary_prob": 0.25, "embb_prob": 0.8, "urllc_prob": 0.1, "mmtc_prob": 0.1, "arrival_rate": 3.5, "departure_rate": 2.0, "thinking": "High Stress", "qos_demand": 0.38, "dynamic_ratio": 1.75, "m_stress": 0.475, "dynamic_stress": 0.3023, "raw_pressure": 0.6823, "pressure": 0.9447, "w_qos": 6.0, "w_energy": 3.0, "w_fairness": 1.0, "reward": 2.219740560962472, "qos": 0.46751184684747055, "energy_usage": 0.21696219459053206, "fairness": 0.06555606364924456, "score": 0.32},
    {"walk_prob": 0.25, "vehicle_prob": 0.3, "stationary_prob": 0.45, "embb_prob": 0.33, "urllc_prob": 0.34, "mmtc_prob": 0.33, "arrival_rate": 2.5, "departure_rate": 2.5, "thinking": "Mean Stress", "qos_demand": 0.334, "dynamic_ratio": 1.0, "m_stress": 0.305, "dynamic_stress": 0.1525, "raw_pressure": 0.4865, "pressure": 0.4526, "w_qos": 5.0, "w_energy": 4.0, "w_fairness": 1.0, "reward": 2.3667880224541764, "qos": 0.5962432563330076, "energy_usage": 0.1996275674389244, "fairness": 0.18408201054483653, "score": 0.58},
    {"walk_prob": 0.15, "vehicle_prob": 0.02, "stationary_prob": 0.83, "embb_prob": 0.05, "urllc_prob": 0.05, "mmtc_prob": 0.9, "arrival_rate": 0.5, "departure_rate": 4.5, "thinking": "Idle", "qos_demand": 0.22, "dynamic_ratio": 0.1111, "m_stress": 0.127, "dynamic_stress": 0.0127, "raw_pressure": 0.2327, "pressure": 0.0096, "w_qos": 2.5, "w_energy": 6.5, "w_fairness": 1.0, "reward": 1.2871758269529237, "qos": 0.7712617436734646, "energy_usage": 0.1688380467109776, "fairness": 0.45646877139061687, "score": 1.06}
]

def build_prompt_zs(env: dict):
    jinja_env = Environment(loader=FileSystemLoader('.'))
    template = jinja_env.get_template('prompt_zs.md')
    return template.render(env)

def build_prompt_fs(env: dict, examples=fs_examples):
    jinja_env = Environment(loader=FileSystemLoader('.'))
    template = jinja_env.get_template('prompt_fs.md')
    return template.render(env, examples=examples)

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

    prompt = build_prompt_fs(env)
    print(prompt)