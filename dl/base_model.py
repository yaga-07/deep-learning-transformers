import torch
import torch.nn as nn
from dl.base_config import BaseConfig

class BaseModel(nn.Module):
    """
    BaseModel is a foundational class for deep learning models. It includes methods
    for saving and loading pretrained models along with their configurations.
    """

    def __init__(self, config: BaseConfig):
        """
        Initialize the BaseModel with a configuration.

        Args:
            config (BaseConfig): The configuration object for the model.
        """
        super().__init__()
        self.config = config

    @classmethod
    def from_pretrained(cls, pretrained_path: str):
        """
        Load a model from a pretrained checkpoint.

        Args:
            pretrained_path (str): Path to the pretrained model checkpoint.

        Returns:
            BaseModel: An instance of the model loaded with pretrained weights and configuration.
        """
        # Load model from a pretrained checkpoint
        state_dict = torch.load(pretrained_path)
        config = BaseConfig.from_dict(state_dict['config'])
        model = cls(config)
        model.load_state_dict(state_dict['model'])
        return model

    def save_pretrained(self, save_path: str):
        """
        Save the model's state dict along with its configuration to the specified path.

        Args:
            save_path (str): Path to save the model checkpoint.
        """
        state_dict = {
            'config': self.config.to_dict(),
            'model': self.state_dict()
        }
        torch.save(state_dict, save_path)
