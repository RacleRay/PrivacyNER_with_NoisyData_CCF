import copy
import torch
import torch.nn as nn

from model.layers.attention import TransformerEncoderLayer
from model.layers.position import FourPosFusionEmbedding, get_pos_embedding
from config import Config



class FLAT(nn.Module):
    def __init__(self):
        super(FLAT, self).__init__()
        self.params = {'other': []}

        # flat position
        pe = get_pos_embedding(Config.max_len, Config.dim_pos)

        if Config.pos_norm:
            pe_sum = pe.sum(dim=-1, keepdim=True)
            with torch.no_grad():
                pe = pe/pe_sum

        self.pe = nn.Parameter(pe, requires_grad=Config.learnable_position)
        if Config.four_pos_shared:
            self.pe_ss = self.pe
            self.pe_se = self.pe
            self.pe_es = self.pe
            self.pe_ee = self.pe
        else:
            self.pe_ss = nn.Parameter(copy.deepcopy(pe), requires_grad=Config.learnable_position)
            self.pe_se = nn.Parameter(copy.deepcopy(pe), requires_grad=Config.learnable_position)
            self.pe_es = nn.Parameter(copy.deepcopy(pe), requires_grad=Config.learnable_position)
            self.pe_ee = nn.Parameter(copy.deepcopy(pe), requires_grad=Config.learnable_position)

        self.pos_layer = FourPosFusionEmbedding('ff', self.pe_ss, self.pe_se, self.pe_es, self.pe_ee,
                                                Config.hidden_size)

        # shape 调整
        if Config.bert_out_size != Config.flat_in_feat_size:
            self.adapter = nn.Linear(Config.bert_out_size, Config.flat_in_feat_size)
            self.params['other'].extend([p for p in self.adapter.parameters()])

        # flat layers
        # 此处作为一种融合BERT特征和词向量特征的方法，不需要设计的太复杂，毕竟已经用了BERT
        self.encoder_layers = []
        for _ in range(Config.num_flat_layers):
            encoder_layer = TransformerEncoderLayer(Config)
            self.encoder_layers.append(encoder_layer)
            self.params['other'].extend([p for p in encoder_layer.parameters()])
        self.encoder_layers = nn.ModuleList(self.encoder_layers)

    def get_params(self):
        return self.params

    def forward(self, inputs):
        char_word_vec = inputs['char_word_vec']
        char_word_mask = inputs['char_word_mask']
        char_word_s = inputs['char_word_s']
        char_word_e = inputs['char_word_e']
        char_len = inputs['char_len']

        pos_embedding = self.pos_layer(char_word_s, char_word_e)

        if Config.bert_out_size != Config.flat_in_feat_size:
            hidden = self.adapter(char_word_vec)
        else:
            hidden = char_word_vec

        for layer in self.encoder_layers:
            hidden = layer(hidden, pos_embedding, char_word_mask)

        # 只取char token的输出
        char_vec = hidden[:, : char_len, :]

        return {'text_vec': char_vec}
