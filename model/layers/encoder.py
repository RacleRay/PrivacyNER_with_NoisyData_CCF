import pickle
import torch
import torch.nn as nn
from transformers import BertModel

from config import Config


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.params = {'ptm': [], 'other': []}
        self.detach_ptm_flag = False

        self.bert = BertModel.from_pretrained(Config.ptm_model, output_hidden_states=True)
        self.params['ptm'].extend([p for p in self.bert.parameters()])

        if Config.use_w2v:
            # 这里的线性变换输出size变为 756，但是 FLAT 模型输入指定为 160 (此处),后续还学会转化到 160
            # 所以也可以将 BERT encode output size 转化为 160，w2v也转化为 160，再concat，输入 FLAT 模型
            self.w2v_linear = nn.Linear(Config.w2v_feat_size, Config.ptm_feat_size)
            self.layer_norm = nn.LayerNorm(Config.ptm_feat_size, eps=Config.layer_norm_eps)
            self.dropout = nn.Dropout(Config.dropout)

            self.params['other'].extend([p for p in self.w2v_linear.parameters()])
            self.params['other'].extend([p for p in self.layer_norm.parameters()])

            self.w2v_array = pickle.load(open(Config.W2V_PATH + 'w2v_vector.pkl', 'rb'))

    def forward(self, inputs):
        "处理得到 FLAT 模型的输入"
        token_id_padded, mask = inputs['text'], inputs['mask']

        # get BERT Encoder out
        text_vecs = self.get_bert_vec(token_id_padded, mask)
        char_vec = text_vecs[Config.num_ptm_layers]

        if Config.use_w2v:
            # get word vec
            word_idx, word_mask = inputs['word_idx'], inputs['word_mask']
            char_lens = inputs['char_len']

            # 当使用预训练 word vec 时，载入静态参数。  另外可以尝试输入 id，构建embedding重新训练参数
            # 构建 FLAT 的输入格式，这部分时组织输入，不进入计算图。
            char_word_vec = []
            for i, bchar_vec in enumerate(char_vec):
                bert_vec = []
                word_vec = []
                pad_vec = []
                for idx, vec in enumerate(bchar_vec):
                    if idx < char_lens[i]:
                        bert_vec.append(vec)
                    else:
                        if int(word_idx[i][idx]) != 0:
                            vec = torch.tensor(self.w2v_array[int(word_idx[i][idx])]).float().cuda()
                            word_vec.append(vec)
                        else:
                            pad_vec.append(torch.zeros(Config.ptm_feat_size).cuda())

                while idx < word_mask.size(1) - 1:
                    pad_vec.append(torch.zeros(Config.ptm_feat_size).cuda())
                    idx += 1

                bert_vec = torch.stack(bert_vec, dim=0).cuda()  # 2维

                if len(word_vec) > 0:
                    word_vec = torch.stack(word_vec, dim=0).cuda()  # 2维
                    word_vec = self.w2v_linear(word_vec)
                    new_vec = torch.cat((bert_vec, word_vec), dim=0)  # 2维
                else:
                    new_vec = bert_vec  # 2维
                new_vec = self.layer_norm(new_vec)
                new_vec = self.dropout(new_vec)

                if len(pad_vec) > 1:
                    pad_vec = torch.stack(pad_vec, dim=0).cuda() # 2维
                    new_vec = torch.cat((new_vec, pad_vec), dim=0)
                elif len(pad_vec) == 1:
                    pad_vec = pad_vec[0].unsqueeze(0)
                    new_vec = torch.cat((new_vec, pad_vec), dim=0)

                char_word_vec.append(new_vec)
            char_word_vec = torch.stack(char_word_vec, dim=0).float().cuda()

            # 位置编码
            pos = torch.arange(0, token_id_padded.size(1)).long().unsqueeze(dim=0).cuda()
            pos = pos * mask.long()

            pad = torch.tensor([0 for _ in range(word_mask.size(1) - mask.size(1))]).unsqueeze(0).repeat(mask.size(0), 1).cuda()
            pos = torch.cat((pos, pad), dim=1)

            # char bert 部分的 head tail 输入
            char_s = pos
            char_e = pos

            # lattice 部分的 head tail 输入
            word_s, word_e = inputs['word_pos_b'], inputs['word_pos_e']

            char_word_head = char_s + word_s
            char_word_tail = char_e + word_e

            # 下面操作都统一到，加上 lattice 的最大长度
            # word_mask 已经在 生成数据集时，pad 到这个batch最大的长度（包括lattice）
            pad = torch.tensor([0 for _ in range(len(word_mask[0]) - len(mask[0]))]).unsqueeze(0).repeat(mask.size(0), 1).cuda()
            char_mask = torch.cat((mask, pad), dim=1).bool()
            char_word_mask = char_mask | word_mask.bool()

            return {'char_word_vec': char_word_vec,
                    'char_word_mask': char_word_mask,
                    'char_word_s': char_word_head,
                    'char_word_e': char_word_tail,
                    'char_len': mask.size(1)}
        else:
            return {'text_vec': char_vec}

    def get_params(self):
        return self.params

    def detach_ptm(self, flag):
        self.detach_ptm_flag = flag

    def get_bert_vec(self, token_id_padded, text_mask, text_pos=None):
        _, _, text_vecs = self.bert(token_id_padded, text_mask, position_ids=text_pos)

        # 不进入训练图
        if self.detach_ptm_flag:
            for vec in text_vecs:
                vec.detach()

        return text_vecs