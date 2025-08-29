import torch

class Config:
    def __init__(
            self,
            vocab_size=10000,
            pad_idx=0,
            seq_length=512,
            d_model = 512,
            d_ff=2048,
            num_query_heads=8,
            num_layers=12,
            norm_eps = 1e-6,
            bias=False,
            gamma_init=1.0,
            attention_probs_dropout_ratio=0.1,
            dropout_ratio=0.1,
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ):
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.seq_length = seq_length
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.num_query_heads = num_query_heads
        self.norm_eps = norm_eps
        self.bias = bias
        self.gamma_init = gamma_init
        self.attention_probs_dropout_ratio = attention_probs_dropout_ratio
        self.dropout_ratio = dropout_ratio
        self.device = device
