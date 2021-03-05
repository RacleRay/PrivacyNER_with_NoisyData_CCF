import torch.nn as nn
from config import Config

from model.layers.encoder import Encoder
from model.layers.flat import FLAT
from model.layers.structuredOut import StructuredOut


class CascadeFLAT(nn.Module):
    """
    整体模型为： BERT encoder + Word vec 计算第一阶段表示，再将结果输入 FLAT model，进行第二阶段融合计算。
                 输出为 CRF 层。
                 训练先 固定 BERT encoder，一定阶段后，开放 encoder 进行 finetuning。
                 Word vec 没有进行微调，也可以设为embedding进行微调，但是显存要求较高。
    """
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()

        self.flat = FLAT()

        self.out_layer = StructuredOut()

        self.layer_list = []
        self.layer_list.append(self.encoder)
        self.layer_list.append(self.flat)
        self.layer_list.append(self.out_layer)

        self.params = {}

    def forward(self, inputs, en_decode=True):
        # encoder
        encoder_inputs = {'text': inputs['text'], 'mask': inputs['mask']}
        if Config.use_w2v:
            encoder_inputs['word_idx'] = inputs['word_idx']
            encoder_inputs['word_mask'] = inputs['word_mask']
            encoder_inputs['word_pos_b'] = inputs['word_pos_b']
            encoder_inputs['word_pos_e'] = inputs['word_pos_e']
            encoder_inputs['char_len'] = inputs['char_len']

        encoder_outputs = self.encoder(encoder_inputs)

        # FLAT fusion
        fusion_outputs = self.flat(encoder_outputs)

        # output
        fusion_outputs['mask'] = encoder_inputs['mask']

        result = self.out_layer(fusion_outputs, en_pred=en_decode)

        return result

    def cal_loss(self, preds, targets, mask):
        loss = self.out_layer.cal_loss(preds, targets, mask)
        return loss

    def find_entity(self, pred, mask):
        "mask: char mask"
        return self.out_layer.find_entity(pred, mask)

    def get_params(self):
        if not self.params:
            for layer in self.layer_list:
                for key, value in layer.get_params().items():
                    if key not in self.params:
                        self.params[key] = []
                    self.params[key].extend(value)
        return self.params

    def detach_ptm(self, flag):
        self.encoder.detach_ptm(flag)
