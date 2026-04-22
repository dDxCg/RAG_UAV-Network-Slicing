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

    
    test_examples = [{'jain': 0.0, 'score': 1.3, 'w_qos': 6.5, 'jain_m': 0.6665, 'jain_s': 0.4859, 'reward': 4.9, 'm_stress': 0.325, 'pressure': 0.0005, 'raw_jain': 0.5691, 'trial_no': 1, 'w_energy': 2.5, 'embb_prob': 0.05, 'mmtc_prob': 0.9, 'walk_prob': 0.45, 'qos_demand': 0.22, 'urllc_prob': 0.05, 'w_fairness': 1.0, 'env_penalty': 3.5981, 'arrival_rate': 4.5, 'raw_pressure': 0.5125, 'vehicle_prob': 0.3, 'dynamic_ratio': 9.0, 'departure_rate': 0.5, 'dynamic_stress': 0.2925, 'energy_penalty': 2.5986, 'stationary_prob': 0.25, 'fairness_penalty': 0.9995}, {'jain': 0.0, 'score': 1.27, 'w_qos': 6.5, 'jain_m': 0.6665, 'jain_s': 0.4859, 'reward': 5.19, 'm_stress': 0.325, 'pressure': 0.0005, 'raw_jain': 0.5691, 'trial_no': 5, 'w_energy': 2.0, 'embb_prob': 0.05, 'mmtc_prob': 0.9, 'walk_prob': 0.45, 'qos_demand': 0.22, 'urllc_prob': 0.05, 'w_fairness': 1.5, 'env_penalty': 3.9146, 'arrival_rate': 4.5, 'raw_pressure': 0.5125, 'vehicle_prob': 0.3, 'dynamic_ratio': 9.0, 'departure_rate': 0.5, 'dynamic_stress': 0.2925, 'energy_penalty': 3.2483, 'stationary_prob': 0.25, 'fairness_penalty': 0.6663}, {'jain': 0.0, 'score': 1.08, 'w_qos': 6.0, 'jain_m': 0.6665, 'jain_s': 0.4859, 'reward': 5.47, 'm_stress': 0.325, 'pressure': 0.0005, 'raw_jain': 0.5691, 'trial_no': 6, 'w_energy': 1.5, 'embb_prob': 0.05, 'mmtc_prob': 0.9, 'walk_prob': 0.45, 'qos_demand': 0.22, 'urllc_prob': 0.05, 'w_fairness': 2.5, 'env_penalty': 4.3977, 'arrival_rate': 4.5, 'raw_pressure': 0.5125, 'vehicle_prob': 0.3, 'dynamic_ratio': 9.0, 'departure_rate': 0.5, 'dynamic_stress': 0.2925, 'energy_penalty': 3.9979, 'stationary_prob': 0.25, 'fairness_penalty': 0.3998}]
    prompt = build_prompt_kb(env, test_examples)
    print(prompt)