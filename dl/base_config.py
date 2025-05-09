from dataclasses import dataclass

@dataclass
class BaseConfig:
    """
    BaseConfig is a configuration class for defining model-specific parameters.
    It provides methods to convert the configuration to and from a dictionary format.
    """
    model_name: str = "base_model"
    """The name of the model."""
    
    model_type: str = "base"
    """The type of the model."""

    def to_dict(self):
        """
        Convert the configuration object to a dictionary.

        Returns:
            dict: A dictionary representation of the configuration.
        """
        return self.__dict__

    @classmethod
    def from_dict(cls, config_dict):
        """
        Create a configuration object from a dictionary.

        Args:
            config_dict (dict): A dictionary containing configuration parameters.

        Returns:
            BaseConfig: An instance of the configuration class.
        """
        return cls(**config_dict)
