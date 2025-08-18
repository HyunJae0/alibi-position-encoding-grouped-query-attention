# encoder self attention -> MHA
# decoder self attention and cross attention -> GQA

import torch
import torch.nn as nn
import torch.nn.init as init
import math

from config import Config


def get_slopes(num_heads, device):
    # each head has a different m(slope)
    n = 2 ** math.floor(math.log2(num_heads))
    m_0 = 2.0 ** (-8.0 / n) # 2^{-8/n}
    m = torch.pow(m_0, torch.arange(1, 1+n))
    
    return m.unsqueeze(-1).unsqueeze(-1).to(device) # m.shape: [num_heads, 1, 1]

def get_relative_positions(seq_length, device):
    x = torch.arange(seq_length, device=device)[None, :]
    y = torch.arange(seq_length, device=device)[:, None]
    
    return x-y

class ALiBiGroupedQueryAttention(nn.Module):
    def __init__(self, config, num_kv_heads):
        super().__init__()
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else config.num_query_heads
        """
        num_kv_heads is None -> Multi-head Attention
        because "num_kv_heads == num_heads" -> Multi-head Attention
        
        num_kv_heads !=0 -> Grouped-query Attention
        num_kv_heads == 1 -> Multi-query Attention
        """
        self.num_query_heads = config.num_query_heads


        if self.num_kv_heads == 1:
            raise ValueError(
                f'the number of key/value heads must be greater than 1'
            )
        elif self.num_query_heads % self.num_kv_heads != 0:
            raise ValueError(
                f'query heads {self.num_query_heads} must be divisible by {self.num_kv_heads}'
            )


        self.query_head_dim = config.d_model // config.num_query_heads

        self.W_q = nn.Linear(config.d_model, self.num_query_heads * self.query_head_dim, bias=config.bias)
        self.W_k = nn.Linear(config.d_model, self.num_kv_heads * self.query_head_dim, bias=config.bias)
        self.W_v = nn.Linear(config.d_model, self.num_kv_heads * self.query_head_dim, bias=config.bias)
        self.W_o = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        """
        num_query_heads == number of query heads
        num_kv_heads == number of key/value heads
        -> queries_per_group == (num_query_heads / num_kv_heads) <=> num_kv_heads == (num_query_heads / queries_per_group)

        W_q.shape: [d_model, num_query_heads * query_head_dim] = [d_model, d_model]
        W_k & W_v shape: [d_model, num_kv_heads * query_head_dim] = [d_model, d_model // queries_per_group]
        """
        self.softmax = nn.Softmax(dim=-1)
        self.attn_weights_dropout = nn.Dropout(p=config.attention_probs_dropout_ratio)

        self.register_buffer('m', get_slopes(config.num_query_heads, config.device))

        init.xavier_normal_(self.W_q.weight)
        init.xavier_normal_(self.W_k.weight)
        init.xavier_normal_(self.W_v.weight, gain=config.gamma_init)
        init.xavier_normal_(self.W_o.weight, gain=config.gamma_init)
        if self.W_q.bias is not None: init.constant_(self.W_q.bias, 0)
        if self.W_k.bias is not None: init.constant_(self.W_k.bias, 0)
        if self.W_v.bias is not None: init.constant_(self.W_v.bias, 0)
        if self.W_o.bias is not None: init.constant_(self.W_o.bias, 0)

    def forward(self, query, key, value, mask, is_cross=False):
        b, q_len, _ = query.shape # batch size, query length
        kv_len = key.shape[1] # key/value length

        if not is_cross:
            """
            This code does not consider generation phase
            """
            assert q_len == kv_len # self-attention -> num_tokens_query == num_tokens_key

        q_proj = self.W_q(query)
        k_proj = self.W_k(key)
        v_proj = self.W_v(value)
        # split heads
        Q = q_proj.view(b, q_len, self.num_query_heads, self.query_head_dim) # [batch_size, query_length, num_query_heads, query_head_dim]
        K = k_proj.view(b, kv_len, self.num_kv_heads, self.query_head_dim) # [batch_size, kv_length, num_kv_heads, query_head_dim]
        V = v_proj.view(b, kv_len, self.num_kv_heads, self.query_head_dim) # [batch_size, kv_length, num_kv_heads, query_head_dim]

        num_groups = self.num_kv_heads
        query_heads_per_group = self.num_query_heads // num_groups

        Q = Q.view(b, q_len, num_groups, query_heads_per_group, self.query_head_dim).permute(0, 2, 3, 1, 4)
        K = K.transpose(1, 2).unsqueeze(2)
        V = V.transpose(1, 2).unsqueeze(2)
        # Q.shpae: [batch_size, num_groups, query_heads_per_group, query_length, query_head_dim]
        # K.shpae: [batch_size, num_kv_heads(=num_groups), 1, kv_length, query_head_dim]
        # V.shpae: [batch_size, num_kv_heads(=num_groups), 1, kv_length, query_head_dim]

        ## attention scores
        Q = Q / math.sqrt(self.query_head_dim)
        gqa_scores = Q @ K.transpose(3, 4)
        # gqa_scores.shape: [batch_size, num_groups, query_heads_per_group, query_length, kv_length]

        if not is_cross: # encoder/decoder self-attention
            alibi_bias = (self.m * get_relative_positions(q_len, query.device)).unsqueeze(0)
            # alibi_bias.shape: [1, num_query_heads, query_length, query_length]

            alibi_bias = alibi_bias.view(1, num_groups, query_heads_per_group, q_len, q_len)
            # alibi_bias.shape: [1, num_groups, query_heads_per_group, query_length, query_length]

            gqa_scores = gqa_scores + alibi_bias

        if mask is not None:
            if mask.ndim == 2: # padding mask
                mask = mask.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                # mask.shape: [b, 1, 1, 1, len]
            elif mask.ndim == 3: # padding mask + causal mask
                mask = mask.unsqueeze(1).unsqueeze(2)
                # mask.shape: [b, 1, 1, len, len]
            gqa_scores = gqa_scores.masked_fill(mask == False, torch.finfo(gqa_scores.dtype).min)

        ## attention weights
        gqa_weights = self.attn_weights_dropout(self.softmax(gqa_scores))

        ## attention value(context)
        gqa_output = gqa_weights @ V
        # [batch_size, num_groups, query_heads_per_group, query_length, kv_length]
        # @
        # [batch_size, kv_heads(=num_groups), 1, kv_length, query_head_dim]
        # -> [batch_size, num_groups, query_heads_per_group, query_length, query_head_dim]

        gqa_output = gqa_output.permute(0, 3, 1, 2, 4).contiguous().view(b,q_len, -1)
        # [batch_size, query_length, num_groups, query_heads_per_group, query_head_dim]
        # -> [batch_size, query_length, d_model]
        # d_model = num_groups * query_heads_per_group * query_head_dim
        output = self.W_o(gqa_output)

        return output




if __name__ == '__main__':
    config = Config()
    q = torch.randn(16, 512, 512).to(config.device) # [batch_size, seq_len, embed_dim]
    k = torch.randn(16, 510, 512).to(config.device)
    v = torch.randn(16, 510, 512).to(config.device)
    a = ALiBiGroupedQueryAttention(config, 4).to(config.device)

    attn_output = a.forward(q, k, v, None,True)
    print(attn_output.shape)
















