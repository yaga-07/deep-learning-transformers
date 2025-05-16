from dl.base_config import BaseConfig

class BertConfig(BaseConfig):
    def __init__(
        self,
        model_type="bert",
        vocab_size=30522,
        hidden_size=768,
        num_attention_heads=12,
        num_hidden_layers=12,
        intermediate_size=3072,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        layer_norm_eps=1e-12,
        initializer_range=0.02,
        **kwargs
    ):
        super().__init__()
        self.model_type = model_type
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        # Set any extra attributes from kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)
