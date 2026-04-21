from jinja2 import Environment, FileSystemLoader

def build_prompt_zs(env: dict):
    jinja_env = Environment(loader=FileSystemLoader('.'))
    template = jinja_env.get_template('prompt_zs.md')
    return template.render(env)

def build_prompt_fs(env: dict):
    jinja_env = Environment(loader=FileSystemLoader('.'))
    template = jinja_env.get_template('prompt_fs.md')
    return template.render(env)

if __name__ == "__main__":
    env = {"arrival_rate": 2.5, "departure_rate": 2.5, 
           "walk_prob": 0.33, "vehicle_prob": 0.33, "staionary_prob": 0.34, 
           "embb_prob": 0.33, "urllc_prob": 0.33, "mmtc_prob": 0.34, 
           "qos_demand": 0.332, 
           "dynamic_ratio": 1.0, "dynamic_stress": 0.1655,  
           "m_stress": 0.331, 
           "raw_pressure": 0.4975, "pressure": 0.0002, 
           "raw_jain": 0.7599, "jain": 0.9998, "jain_m": 0.6203, "jain_s": 0.9308}
    prompt = build_prompt_zs(env)
    print(prompt)