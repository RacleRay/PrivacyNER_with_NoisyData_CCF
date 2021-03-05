import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super(TransformerEncoderLayer, self).__init__()
        self.att_layer = MultiHeadAttention(config.hidden_size,
                                            config.num_heads,
                                            scaled=config.scaled,
                                            attn_dropout=config.attn_dropout)
        if config.use_ff_output:
            self.ff = FeedForward(config.hidden_size,
                                config.hidden_dropout,
                                config.layer_norm_eps)

    def forward(self, inputs, pos_embedding, seq_mask):
        output = self.att_layer(inputs, inputs, inputs, pos_embedding, seq_mask)
        if self.config.use_ff_output:
            output = self.ff(output, inputs)
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, scaled=True, attn_dropout=None):
        super(MultiHeadAttention, self).__init__()

        self.hidden_size = hidden_size

        self.num_heads = num_heads
        self.per_head_size = self.hidden_size // self.num_heads
        self.scaled = scaled
        assert (self.per_head_size * self.num_heads == self.hidden_size)

        # 正常 attention 的 q,k,v 变换矩阵
        self.w_k = nn.Linear(self.hidden_size, self.hidden_size)
        self.w_q = nn.Linear(self.hidden_size, self.hidden_size)
        self.w_v = nn.Linear(self.hidden_size, self.hidden_size)

        # 计算 Rij 的权重
        self.w_r = nn.Linear(self.hidden_size, self.hidden_size)

        # 计算 A* 的权重
        self.u = nn.Parameter(torch.randn(self.num_heads, self.per_head_size), requires_grad=True)
        self.v = nn.Parameter(torch.randn(self.num_heads, self.per_head_size), requires_grad=True)

        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, key, query, value, pos, flat_mask):
        batch = key.size(0)

        # 输入线性变换
        key = self.w_k(key)
        query = self.w_q(query)
        value = self.w_v(value)
        rel_pos_embedding = self.w_r(pos)

        ####### 计算 A* 矩阵的方法 和 论文不是完全一致
        # batch, seq_len, n_head, d_head
        key = torch.reshape(key, [batch, -1, self.num_heads, self.per_head_size])
        query = torch.reshape(query, [batch, -1, self.num_heads, self.per_head_size])
        value = torch.reshape(value, [batch, -1, self.num_heads, self.per_head_size])
        # batch, seq_len, seq_len, n_head, d_head
        rel_pos_embedding = torch.reshape(rel_pos_embedding,
                                          list(rel_pos_embedding.size()[:3]) + [self.num_heads, self.per_head_size])

        # batch, n_head, seq_len, d_head
        key = key.transpose(1, 2)
        query = query.transpose(1, 2)
        value = value.transpose(1, 2)

        # batch, n_head, d_head, seq_len
        key = key.transpose(-1, -2)

        # 1, num_heads, 1, d_head
        u_for_c = self.u.unsqueeze(0).unsqueeze(-2)

        # batch, n_head, seq_len, d_head
        query_and_u_for_c = query + u_for_c

        # batch, n_head, seq_len, seq_len
        A_C = torch.matmul(query_and_u_for_c, key)

        # batch, n_head, seq_len, d_head, seq_len
        rel_pos_embedding_for_b = rel_pos_embedding.permute(0, 3, 1, 4, 2)
        # batch, n_head, seq_len, seq_len, 1, d_head
        query_for_b = query.view([batch, self.num_heads, query.size(2), 1, self.per_head_size])
        # batch, n_head, seq_len, seq_len, 1, d_head
        query_for_b_and_v_for_d = query_for_b + self.v.view(1, self.num_heads, 1, 1, self.per_head_size)

        # batch, n_head, seq_len, seq_len
        B_D = torch.matmul(query_for_b_and_v_for_d, rel_pos_embedding_for_b).squeeze(-2)

        # batch, n_head, seq_len, seq_len
        attn_score_raw = A_C + B_D

        # 计算 score
        if self.scaled:
            attn_score_raw = attn_score_raw / math.sqrt(self.per_head_size)

        mask = 1 - flat_mask.long().unsqueeze(1).unsqueeze(1)
        attn_score_raw_masked = attn_score_raw.masked_fill(mask.bool(), -1e15)

        # batch, n_head, seq_len, seq_len
        attn_score = F.softmax(attn_score_raw_masked, dim=-1)
        attn_score = self.dropout(attn_score)

        # batch, n_head, seq_len, d_head
        value_weighted_sum = torch.matmul(attn_score, value)
        # batch, seq_len, n_head, d_head -> batch, seq_len, hidden_size
        result = value_weighted_sum.transpose(1, 2).contiguous().reshape(batch, -1, self.hidden_size)

        return result


class FeedForward(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob, layer_norm_eps):
        super(FeedForward, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states




