import numpy as np
from statistics import NormalDist
import math

def _calculate_preference(env_config: dict):
    mobility_weights = np.array([0.2, 0.7, 0.1])
    mobility_probs = np.array([
        env_config["walk_prob"], 
        env_config["vehicle_prob"], 
        env_config["stationary_prob"]
    ])

    slice_weights = np.array([0.4, 0.4, 0.2])
    slice_probs = np.array([
        env_config["embb_prob"], 
        env_config["urllc_prob"], 
        env_config["mmtc_prob"]
    ])

    m_stress = np.dot(mobility_probs, mobility_weights)

    dynamic_ratio = env_config["arrival_rate"] / (env_config["departure_rate"] + 1e-6)
    norm_dynamic_ratio = dynamic_ratio / (dynamic_ratio + 1)
    
    dynamic_stress = norm_dynamic_ratio * m_stress
    qos_demand = np.dot(slice_probs, slice_weights)
    
    pressure = qos_demand + dynamic_stress
    return {
        "qos_demand": round(qos_demand, 4),
        "dynamic_ratio": round(dynamic_ratio, 4),
        "m_stress": round(m_stress, 4),
        "dynamic_stress": round(dynamic_stress, 4),
        "raw_pressure": round(pressure, 4)
    }

def _calculate_jain_demand(env_config: dict):
    # 1. Service Demand (Slice)
    # weights: eMBB=0.4, URLLC=0.4, mMTC=0.2
    s_demands = np.array([
        env_config["embb_prob"] * 0.4,
        env_config["urllc_prob"] * 0.4,
        env_config["mmtc_prob"] * 0.2
    ])
    jain_s = (np.sum(s_demands)**2) / (3 * np.sum(s_demands**2) + 1e-9)

    # 2. Mobility Demand
    # weights: Walk=0.2, Vehicle=0.7, Stationary=0.1
    m_demands = np.array([
        env_config["walk_prob"] * 0.2,
        env_config["vehicle_prob"] * 0.7,
        env_config["stationary_prob"] * 0.1
    ])
    jain_m = (np.sum(m_demands)**2) / (3 * np.sum(m_demands**2) + 1e-9)

    jain_total = math.sqrt(jain_s * jain_m)
    return {
        "raw_jain": round(jain_total, 4),
        "jain_m": round(jain_m, 4),
        "jain_s": round(jain_s, 4)
    }

# cdf map z (position of current value) in gauss dist plot to [0,1]
# cdf describe area -inf to z: x,z,cdf co_variable + cdf(z) also show the relate between x and mean 
def _calculate_pressure_calibrated(raw_pressure):
    mean = 0.5679
    std = 0.2429

    z_score = (raw_pressure - mean)/std

    # Aware energy around mean value
    # --> mean + std is the expected value to allow greedy use energy to push qos
    pressure = NormalDist(mu=mean, sigma=std).cdf(z_score)
    return pressure

def _calculate_jain_calibrated(raw_jain):
    mean = 0.6372
    std = 0.1172
    z_score = (raw_jain - mean)/std

    jain = NormalDist(mu=mean, sigma=std).cdf(z_score)
    return jain 

def numerical_analysis(env_config: dict):
    pref = _calculate_preference(env_config)
    jain = _calculate_jain_demand(env_config)

    pressure_calibrated = _calculate_pressure_calibrated(pref["raw_pressure"])
    jain_calibrated = _calculate_jain_calibrated(jain["raw_jain"])

    return {
        **env_config,
        **pref,
        **jain,
        "pressure": round(pressure_calibrated, 4),
        "jain": round(jain_calibrated, 4)
    }

def _get_env_penalty(env_config, w_qos, w_energy, w_fairness):
    pref = _calculate_preference(env_config)
    raw_pressure = pref['raw_pressure']
    pressure = _calculate_pressure_calibrated(raw_pressure)

    jain_pref = _calculate_jain_demand(env_config)
    raw_jain = jain_pref["raw_jain"]
    jain = _calculate_jain_calibrated(raw_jain)

    ratio_factor = w_qos / (w_energy + 1e-6)
    # Higher pressure, lower energy penalty ~ Harder env, allow more energy usage
    energy_penalty = ratio_factor * (1 - pressure)

    # Raw Jain [0,1]: 1 is totally fair
    # --> (1 - jain) Less jain = More penalty
    # --> (1 - pressure) More stable - more aware to fairness
    fairness_penalty = (1 / (w_fairness + 1e-6)) * (1 - pressure) * (1 - jain)

    penalty = fairness_penalty + energy_penalty
    return {
        "env_penalty": round(penalty, 4),
        "energy_penalty": round(energy_penalty, 4),
        "fairness_penalty": round(fairness_penalty, 4)
    }