import torch
import torch.nn as nn

from config import Config
from attention import ALiBiGroupedQueryAttention

class FeedForwardNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense_layer1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.gelu = nn.GELU()
        self.dense_layer2 = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_ratio)

    def forward(self, hidden_states):
        hidden_states = self.dense_layer2(self.dropout(self.gelu(self.dense_layer1(hidden_states))))
        return hidden_states

class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm_1 = nn.LayerNorm(config.d_model, config.norm_eps)
        self.decoder_self_attention = ALiBiGroupedQueryAttention(config, num_kv_heads=4) # GQA
        self.dropout_1 = nn.Dropout(p=config.dropout_ratio)

        self.norm_2 = nn.LayerNorm(config.d_model, config.norm_eps)
        self.feed_forward = FeedForwardNetwork(config)
        self.dropout_2 = nn.Dropout(p=config.dropout_ratio)

    def forward(self, trg):
        _hidden_states = trg
        norm_1 = self.norm_1(trg)
        gqa_output_1 = self.decoder_self_attention(norm_1, norm_1, norm_1)
        hidden_states = _hidden_states + self.dropout_1(gqa_output_1)

        _hidden_states = hidden_states
        norm_2 = self.norm_2(hidden_states)
        hidden_states = self.feed_forward(norm_2)
        hidden_states = _hidden_states + self.dropout_2(hidden_states)

        return hidden_states

class DecoderOnlyTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pad_idx = config.pad_idx
        self.embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_idx)

        self.decoder_layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_layers)])

        self.output_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.output_head.weight = self.embedding.weight

    def forward(self, trg):
        hidden_states = self.embedding(trg)

        for decoder_layer in self.decoder_layers:
            hidden_states = decoder_layer(hidden_states)

        logits = self.output_head(hidden_states)

        return logits
        
if __name__ == '__main__':
    config = Config()
    decoder_layer = DecoderLayer(config)
    print(f'num query heads: {config.num_query_heads} | num key/value heads: {decoder_layer.decoder_self_attention.num_kv_heads}')
    print(decoder_layer.decoder_self_attention.num_kv_heads != config.num_query_heads)
