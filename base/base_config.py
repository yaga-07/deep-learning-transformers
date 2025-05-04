from dataclasses import dataclass

@dataclass
class BaseConfig:
    model_type: str = "base"
    vocab_size: int = 30522
    hidden_size: int = 768
    num_attention_heads: int = 12
    num_hidden_layers: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    layer_norm_eps: float = 1e-12
    initializer_range: float = 0.02

    def to_dict(self):
        return self.__dict__

    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)
