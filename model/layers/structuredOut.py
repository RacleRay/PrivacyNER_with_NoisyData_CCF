import torch
import torch.nn as nn
from pytorchcrf import CRF
from config import Config


class StructuredOut(nn.Module):
    def __init__(self):
        super(StructuredOut, self).__init__()
        self.params = {'other': [], 'crf': []}

        # flat_out_feat_size: 160
        # Config.num_types * Config.num_tag_each_type: 56
        self.emission_linear = nn.Linear(Config.flat_out_feat_size,
                                         Config.num_types * Config.num_tag_each_type)

        self.crf = CRF(Config.num_tag_each_type , batch_first=True)

        self.params['other'].extend([p for p in self.emission_linear.parameters()])
        self.params['crf'].extend([p for p in self.crf.parameters()])

    def get_params(self):
        return self.params

    def forward(self, inputs, en_pred=True):
        text_vec, mask = inputs['text_vec'], inputs['mask']

        # # mask 掉special token的位置
        # NOTE：这是错误的方式，CRF输入要求起始为 1，作为开始符号
        # first_special_token = torch.tensor([[0] for _ in range(mask.size(0))]).cuda()
        # mask.scatter_(1, first_special_token, 0)
        # last_special_token = mask.sum(dim=1, keepdim=True)
        # mask.scatter_(1, last_special_token.long().cuda(), 0)

        emission = self.cal_emission(text_vec)

        if en_pred:
            pred = self.decode(emission, mask)
        else:
            pred = None
        return {'emission': emission,
                'pred': pred}

    def decode(self, emission, mask):
        "emission: batch, num_types, seq_len, num_tag_each_type"
        emission_shape = emission.size()

        # batch, 1, seq_len
        mask = mask.unsqueeze(dim=1)
        # batch, num_types, seq_len   # 每种 type 的mask都是一致的，同一个输入嘛
        mask = mask.repeat(1, emission_shape[1], 1)
        # batch * num_types, seq_len
        mask = mask.reshape([-1, mask.size(2)])

        # batch * num_types, seq_len, num_tag_each_type
        emission = emission.reshape([-1, emission_shape[2], emission_shape[3]])

        # batch * num_types, seq_len
        result = self.crf.decode(emission, mask)

        # batch, num_types, seq_len
        result = result.reshape([-1, emission_shape[1], emission_shape[2]])
        result = result.tolist()
        return result

    def cal_emission(self, text_vec):
        # batch, seq_len, num_types * num_tag_each_type
        emission = self.emission_linear(text_vec)
        # batch, seq_len, num_types, num_tag_each_type
        emission = emission.reshape(list(emission.size()[:2]) + [Config.num_types, Config.num_tag_each_type])
        # batch, num_types, seq_len, num_tag_each_type
        emission = emission.permute([0, 2, 1, 3])
        return emission

    def cal_loss(self, preds, targets, mask):
        if 'loss_mask' in targets:
            mask = targets['loss_mask']

        # # mask 掉special token的位置
        # NOTE：这是错误的方式，CRF输入要求起始为 1，作为开始符号
        # first_special_token = torch.tensor([[0] for _ in range(mask.size(0))])
        # mask.scatter_(1, first_special_token, 0)
        # last_special_token = mask.sum(dim=1) - 1
        # mask.scatter_(1, last_special_token, 0)

        emission = preds['emission']  # batch, num_types, seq_len, num_tag_each_type
        y_true = targets['y_true']    # batch, num_types, seq_len
        mask = mask.unsqueeze(dim=1)
        mask = mask.repeat(1, emission.size(1), 1)
        mask = mask.reshape([-1, mask.size(2)])         # batch*num_types, seq_len
        emission = emission.reshape([-1, emission.size(2), emission.size(3)])  # batch*num_types, seq_len, num_tag_each_type
        y_true = y_true.reshape([-1, y_true.size(2)])   # batch*num_types, seq_len
        _loss = -self.crf(emission, y_true, mask, reduction='token_mean')
        return _loss

    def find_entity(self, pred, mask):
        "pred: batch, num_types, seq_len.  输出：[category, start_idx, end_idx]"
        entity = []
        for pre, mask in zip(pred, mask):
            entity_line = []

            for type_index in range(Config.num_types):
                cur_type = pre[type_index]

                start_idx = -1
                for pos, label in enumerate(cur_type):
                    if int(mask[pos]) == 1:
                        if label == 0:  # O
                            if start_idx != -1:
                                entity_line.append([type_index, start_idx, pos - 1])
                                start_idx = -1
                        elif label == 1:  # B
                            if start_idx != -1:
                                entity_line.append([type_index, start_idx, pos - 1])
                            start_idx = pos
                        elif label == 3:  # E
                            if start_idx != -1:
                                entity_line.append([type_index, start_idx, pos])
                                start_idx = -1
                    else:
                        break

            entity.append(entity_line)

        return entity