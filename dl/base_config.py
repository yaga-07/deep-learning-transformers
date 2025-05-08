from dataclasses import dataclass

@dataclass
class BaseConfig:
    model_name: str = "base_model"
    model_type: str = "base"

    def to_dict(self):
        return self.__dict__

    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)
