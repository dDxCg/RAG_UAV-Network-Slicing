from dataclasses import dataclass
from dataclasses import asdict
from psycopg.types.json import Json
@dataclass
class EnvConfig:
    embb_prob: float
    urllc_prob: float
    mmtc_prob: float
    walk_prob: float
    vehicle_prob: float
    stationary_prob: float
    arrival_rate: float
    departure_rate: float
    
    def __to_dict__(self):
        return {
            'embb_prob': self.embb_prob,
            'urllc_prob': self.urllc_prob,
            'mmtc_prob': self.mmtc_prob,
            'walk_prob': self.walk_prob,
            'vehicle_prob': self.vehicle_prob,
            'stationary_prob': self.stationary_prob,
            'arrival_rate': self.arrival_rate,
            'departure_rate': self.departure_rate
        }
    
@dataclass
class EnvVector:
    embb_prob: float
    urllc_prob: float
    mmtc_prob: float
    walk_prob: float
    vehicle_prob: float
    stationary_prob: float
    arrival_rate: float
    departure_rate: float

    def __init__(self, env: dict):
        self.embb_prob = env['embb_prob']
        self.urllc_prob = env['urllc_prob']
        self.mmtc_prob = env['mmtc_prob']
        self.walk_prob = env['walk_prob']
        self.vehicle_prob = env['vehicle_prob']
        self.stationary_prob = env['stationary_prob']
        self.arrival_rate = env['arrival_rate']
        self.departure_rate = env['departure_rate']

    def __to_pg_vector__(self):
        vec = list(asdict(self).values())
        return '[' + ','.join(str(float(x)) for x in vec) + ']'
@dataclass
class EvalVector:
    pressure: float
    slice_pressure: float
    mobility_pressure: float
    dynamic_pressure: float
    jain: float
    slice_jain: float
    mobility_jain: float

    def __init__(self, eval: dict):
        self.pressure = eval['pressure']
        self.slice_pressure = eval['qos_demand']
        self.mobility_pressure = eval['m_stress']
        self.dynamic_pressure = eval['dynamic_stress']
        self.jain = eval['jain']
        self.slice_jain = eval['jain_s']
        self.mobility_jain = eval['jain_m']

    def __to_pg_vector__(self):
        vec = list(asdict(self).values())
        return '[' + ','.join(str(float(x)) for x in vec) + ']'

@dataclass
class WeightVector:
    w_qos: float
    w_energy: float
    w_fairness: float

    def __init__(self, log: dict):
        self.w_qos = log['w_qos']
        self.w_energy = log['w_energy']
        self.w_fairness = log['w_fairness']

    def __to_pg_vector__(self):
        vec = list(asdict(self).values())
        return '[' + ','.join(str(float(x)) for x in vec) + ']'

@dataclass
class KBEntry:
    env_vector: EnvVector
    eval_vector: EvalVector
    weight_vector: WeightVector
    metadata: Json

    def __init__(self, log: dict):
        self.env_vector = EnvVector(log)
        self.eval_vector = EvalVector(log)
        self.weight_vector = WeightVector(log)
        self.metadata = Json(log)

    def __to_dict__(self):
        return {
            'env_vector': self.env_vector.__to_dict__(),
            'eval_vector': self.eval_vector.__to_dict__(),
            'weight_vector': self.weight_vector.__to_dict__(),
            'metadata': self.metadata
        }

if __name__ == "__main__":
    log = {
        "embb_prob": 0.10, "urllc_prob": 0.80, "mmtc_prob": 0.10,
        "walk_prob": 0.45, "vehicle_prob": 0.10, "stationary_prob": 0.45,
        "arrival_rate": 3.0, "departure_rate": 3.0,
        "pressure": 0.1500, "qos_demand": 0.210, "m_stress": 0.150, "dynamic_stress": 0.120,
        "jain": 0.8850, "jain_s": 0.850, "jain_m": 0.820,
        "w_qos": 8.0, "w_energy": 1.0, "w_fairness": 1.0,
        "reward": 4.85, "energy_penalty": 0.25, "fairness_penalty": 0.12, "score": 4.48,
        "thinking": "URLLC dominant slice requires maximum QoS. Minimal stress and high fairness allow for aggressive w_qos weighting with low energy/fairness overhead."
    }
    entry = KBEntry(log)
    print(entry)

