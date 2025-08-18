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

class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm_1 = nn.LayerNorm(config.d_model, config.norm_eps)
        self.encoder_self_attention = ALiBiGroupedQueryAttention(config, num_kv_heads=None) # MHA
        self.dropout_1 = nn.Dropout(p=config.dropout_ratio)

        self.norm_2 = nn.LayerNorm(config.d_model, config.norm_eps)
        self.feed_forward = FeedForwardNetwork(config)
        self.dropout_2 = nn.Dropout(p=config.dropout_ratio)

    def forward(self, src, src_mask):
        _hidden_states = src
        norm_1 = self.norm_1(src)
        mha_output = self.encoder_self_attention(norm_1, norm_1, norm_1, src_mask, is_cross=False)
        hidden_states = _hidden_states + self.dropout_1(mha_output)

        _hidden_states = hidden_states
        norm_2 = self.norm_2(hidden_states)
        hidden_states = self.feed_forward(norm_2)
        hidden_states = _hidden_states + self.dropout_2(hidden_states)

        return hidden_states

class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm_1 = nn.LayerNorm(config.d_model, config.norm_eps)
        self.decoder_self_attention = ALiBiGroupedQueryAttention(config, num_kv_heads=4) # GQA
        self.dropout_1 = nn.Dropout(p=config.dropout_ratio)

        self.norm_2 = nn.LayerNorm(config.d_model, config.norm_eps)
        self.decoder_cross_attention = ALiBiGroupedQueryAttention(config, num_kv_heads=4) # GQA
        self.dropout_2 = nn.Dropout(p=config.dropout_ratio)

        self.norm_3 = nn.LayerNorm(config.d_model, config.norm_eps)
        self.feed_forward = FeedForwardNetwork(config)
        self.dropout_3 = nn.Dropout(p=config.dropout_ratio)

    def forward(self, trg, encoder_output, trg_mask, src_mask):
        _hidden_states = trg
        norm_1 = self.norm_1(trg)
        gqa_output_1 = self.decoder_self_attention(norm_1, norm_1, norm_1, trg_mask, is_cross=False)
        hidden_states = _hidden_states + self.dropout_1(gqa_output_1)

        _hidden_states = hidden_states
        norm_2 = self.norm_2(hidden_states)
        gqa_output_2 = self.decoder_cross_attention(norm_2, encoder_output, encoder_output, src_mask, is_cross=True)
        hidden_states = _hidden_states + self.dropout_2(gqa_output_2)

        _hidden_states = hidden_states
        norm_3 = self.norm_3(hidden_states)
        hidden_states = self.feed_forward(norm_3)
        hidden_states = _hidden_states + self.dropout_3(hidden_states)

        return hidden_states

class TransformerEncoder(nn.Module):
    def __init__(self, config, shared_embedding):
        super().__init__()
        self.embedding = shared_embedding
        self.encoder_layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.num_layers)])

    def forward(self, src, src_mask):
        hidden_states = self.embedding(src)

        for encoder_layer in self.encoder_layers:
            hidden_states = encoder_layer(hidden_states, src_mask)

        return hidden_states

class TransformerDecoder(nn.Module):
    def __init__(self, config, shared_embedding):
        super().__init__()
        self.embedding = shared_embedding
        self.decoder_layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_layers)])

    def forward(self, trg, encoder_output, trg_mask, src_mask):
        hidden_states = self.embedding(trg)

        for decoder_layer in self.decoder_layers:
            hidden_states = decoder_layer(hidden_states, encoder_output, trg_mask, src_mask)

        return hidden_states

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pad_idx = config.pad_idx
        self.shared_embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_idx)

        self.Encoder = TransformerEncoder(config, shared_embedding=self.shared_embedding)
        self.Decoder = TransformerDecoder(config, shared_embedding=self.shared_embedding)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.shared_embedding.weight


    def forward(self, src, trg):
        # src.shape: [batcha_size, src_length]
        # trg.shape: [batch_size, trg_length]
        src_mask = (src != self.pad_idx).to(src.device)
        trg_pad_mask = (trg != self.pad_idx).unsqueeze(1) # [batch_size, 1, trg_length]
        trg_len = trg.shape[-1]
        trg_causal_mask = torch.ones((trg_len, trg_len), dtype=torch.bool).tril()
        trg_mask = (trg_pad_mask & trg_causal_mask).to(src.device) # [batch_size, trg_length, trg_length]

        encoder_output = self.Encoder(src, src_mask)
        decoder_output = self.Decoder(trg, encoder_output, trg_mask, src_mask)

        logits = self.lm_head(decoder_output)

        return logits

if __name__ == '__main__':
    config = Config()
    encoder_layer = EncoderLayer(config)
    print(encoder_layer.encoder_self_attention.num_kv_heads == config.num_query_heads)

    decoder_layer = DecoderLayer(config)
    print(f'num query heads: {config.num_query_heads} | num key/value heads: {decoder_layer.decoder_self_attention.num_kv_heads}')

    print(decoder_layer.decoder_self_attention.num_kv_heads != config.num_query_heads)
    print(decoder_layer.decoder_cross_attention.num_kv_heads != config.num_query_heads)

