

def _build_prompt(env_config, selected_exps):
    prompt = """Placeholder"""
    prompt += f"\nCurrent Environment Config:\n{env_config}\n"
    prompt += "Relevant Past Experiences:\n"
    for exp in selected_exps:
        prompt += f"{exp}\n"
    prompt += "Based on the above, suggest new reward weights (w_qos, w_energy, w_fairness) for the current environment configuration."
    return prompt

def suggest_reward_weights(env_config, selected_exps):
    prompt = _build_prompt(env_config, selected_exps)
    # TODO: Call LLM API with the prompt and parse the response to extract reward weights
    return {
        "w_qos": 7,
        "w_energy": 2,
        "w_fairness": 1
    }