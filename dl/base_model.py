import torch
import torch.nn as nn
from dl.base_config import BaseConfig

class BaseModel(nn.Module):
    def __init__(self, config: BaseConfig):
        super().__init__()
        self.config = config

    @classmethod
    def from_pretrained(cls, pretrained_path: str):
        # Load model from a pretrained checkpoint
        state_dict = torch.load(pretrained_path)
        config = BaseConfig.from_dict(state_dict['config'])
        model = cls(config)
        model.load_state_dict(state_dict['model'])
        return model
