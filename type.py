from dataclasses import dataclass

@dataclass
class EnvConfig:
    embbb_prob: float
    urllc_prob: float
    mmtc_prob: float
    arrival_rate: float
    departure_rate: float
    w_qos: float
    w_energy: float
    w_fairness: float

    def __to_dict__(self):
        return {
            'embbb_prob': self.embbb_prob,
            'urllc_prob': self.urllc_prob,
            'mmtc_prob': self.mmtc_prob,
            'arrival_rate': self.arrival_rate,
            'departure_rate': self.departure_rate,
            'w_qos': self.w_qos,
            'w_energy': self.w_energy,  
            'w_fairness': self.w_fairness
        }