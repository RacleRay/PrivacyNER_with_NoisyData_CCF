import torch
import torch.nn as nn
import math
import torch.nn.functional as F


def get_pos_embedding(max_seq_len,
                      embedding_dim,
                      padding_idx=None,
                      rel_pos_init=0):
    """
    rel pos init:
        如果是0，那么从-max_len到max_len的相对位置编码矩阵就按0-2*max_len来初始化，
        如果是1，那么就按-max_len,max_len来初始化
    """
    num_embeddings = 2 * max_seq_len + 1
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
    if rel_pos_init == 0:
        emb = torch.arange(num_embeddings,
                           dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
    else:
        emb = torch.arange(-max_seq_len, max_seq_len + 1,
                           dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)],
                    dim=1).view(num_embeddings, -1)
    if embedding_dim % 2 == 1:
        # zero pad
        emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
    if padding_idx is not None:
        emb[padding_idx, :] = 0
    return emb


class FourPosFusionEmbedding(nn.Module):
    "FLAT 位置编码"
    def __init__(self, fusion_method, pe_ss, pe_se, pe_es, pe_ee, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.pe_ss = pe_ss
        self.pe_se = pe_se
        self.pe_es = pe_es
        self.pe_ee = pe_ee

        self.fusion_method = fusion_method
        if self.fusion_method == 'ff':
            self.pos_fusion_forward = nn.Sequential(
                nn.Linear(self.hidden_size * 4, self.hidden_size),
                nn.ReLU(inplace=True)
            )

        if self.fusion_method == 'ff_linear':
            self.pos_fusion_forward = nn.Linear(self.hidden_size * 4,
                                                self.hidden_size)

        elif self.fusion_method == 'ff_two':
            self.pos_fusion_forward = nn.Sequential(
                nn.Linear(self.hidden_size * 2, self.hidden_size),
                nn.ReLU(inplace=True)
            )

        elif self.fusion_method == 'attn':
            self.w_r = nn.Linear(self.hidden_size, self.hidden_size)
            self.pos_attn_score = nn.Sequential(
                nn.Linear(self.hidden_size * 4, self.hidden_size * 4),
                nn.ReLU(),
                nn.Linear(self.hidden_size * 4, 4),
                nn.Softmax(dim=-1)
            )

        elif self.fusion_method == 'gate':
            self.w_r = nn.Linear(self.hidden_size, self.hidden_size)
            self.pos_gate_score = nn.Sequential(
                nn.Linear(self.hidden_size * 4, self.hidden_size * 2),
                nn.ReLU(),
                nn.Linear(self.hidden_size * 2, 4 * self.hidden_size)
            )

    def forward(self, pos_s, pos_e):
        batch = pos_s.size(0)
        max_seq_len = pos_s.size(1)

        pos_ss = pos_s.unsqueeze(-1) - pos_s.unsqueeze(-2)
        pos_se = pos_s.unsqueeze(-1) - pos_e.unsqueeze(-2)
        pos_es = pos_e.unsqueeze(-1) - pos_s.unsqueeze(-2)
        pos_ee = pos_e.unsqueeze(-1) - pos_e.unsqueeze(-2)

        # prepare relative position encoding
        pe_ss = self.pe_ss[(pos_ss).view(-1) + max_seq_len].view(
            size=[batch, max_seq_len, max_seq_len, -1])
        pe_se = self.pe_se[(pos_se).view(-1) + max_seq_len].view(
            size=[batch, max_seq_len, max_seq_len, -1])
        pe_es = self.pe_es[(pos_es).view(-1) + max_seq_len].view(
            size=[batch, max_seq_len, max_seq_len, -1])
        pe_ee = self.pe_ee[(pos_ee).view(-1) + max_seq_len].view(
            size=[batch, max_seq_len, max_seq_len, -1])

        if self.fusion_method == 'ff':
            pe_4 = torch.cat([pe_ss, pe_se, pe_es, pe_ee], dim=-1)
            rel_pos_embedding = self.pos_fusion_forward(pe_4)

        elif self.fusion_method == 'ff_linear':
            pe_4 = torch.cat([pe_ss, pe_se, pe_es, pe_ee], dim=-1)
            rel_pos_embedding = self.pos_fusion_forward(pe_4)

        elif self.fusion_method == 'ff_two':
            pe_2 = torch.cat([pe_ss, pe_ee], dim=-1)
            rel_pos_embedding = self.pos_fusion_forward(pe_2)

        elif self.fusion_method == 'attn':
            pe_4 = torch.cat([pe_ss, pe_se, pe_es, pe_ee], dim=-1)
            attn_score = self.pos_attn_score(pe_4)
            pe_4_unflat = self.w_r(pe_4.view(
                batch, max_seq_len, max_seq_len, 4, self.hidden_size))
            pe_4_fusion = (attn_score.unsqueeze(-1) * pe_4_unflat).sum(dim=-2)
            rel_pos_embedding = pe_4_fusion

        elif self.fusion_method == 'gate':
            pe_4 = torch.cat([pe_ss, pe_se, pe_es, pe_ee], dim=-1)
            gate_score = self.pos_gate_score(pe_4).view(
                batch, max_seq_len, max_seq_len, 4, self.hidden_size)
            gate_score = F.softmax(gate_score, dim=-2)
            pe_4_unflat = self.w_r(pe_4.view(
                batch, max_seq_len, max_seq_len, 4, self.hidden_size))
            pe_4_fusion = (gate_score * pe_4_unflat).sum(dim=-2)
            rel_pos_embedding = pe_4_fusion

        return rel_pos_embedding
