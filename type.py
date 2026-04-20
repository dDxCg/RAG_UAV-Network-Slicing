from dataclasses import dataclass

@dataclass
class EnvConfig:
    embb_prob: float
    urllc_prob: float
    mmtc_prob: float
    arrival_rate: float
    departure_rate: float
    
    def __to_dict__(self):
        return {
            'embb_prob': self.embb_prob,
            'urllc_prob': self.urllc_prob,
            'mmtc_prob': self.mmtc_prob,
            'arrival_rate': self.arrival_rate,
            'departure_rate': self.departure_rate
        }
    
@dataclass
class KBEntry:
    env_config: EnvConfig
    w_qos: float
    w_energy: float
    w_fairness: float
    reward: float
    thinking: str

    def __to_dict__(self):
        return {
            **self.env_config.__to_dict__(),
            'w_qos': self.w_qos,
            'w_energy': self.w_energy,
            'w_fairness': self.w_fairness,
            'reward': self.reward,
            'thinking': self.thinking
        }