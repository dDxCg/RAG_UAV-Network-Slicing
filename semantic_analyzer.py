import numpy as np
from statistics import NormalDist
import math

def _calculate_preference(env_config):
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
        "m_stress": round(m_stress, 4),
        "dynamic_stress": round(dynamic_stress, 4),
        "raw_pressure": round(pressure, 4)
    }

def _calculate_pressure_calibrated(raw_pressure):
    mean = 0.5001
    std = 0.1142

    z = (raw_pressure - mean)/std

    pressure = NormalDist(mu=0, sigma=1).cdf(z)
    return pressure

def _get_env_pref(env_config):
    pref = _calculate_preference(env_config)
    raw_pressure = pref['raw_pressure']
    pressure = _calculate_pressure_calibrated(raw_pressure)

    return {
        **env_config,
        **pref,
        "pressure": round(pressure, 4)
    }