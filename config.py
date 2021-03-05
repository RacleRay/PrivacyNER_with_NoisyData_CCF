import os
import random
import torch
import numpy as np
import pkuseg
from utils.preprocess.tokenizer import MyTokenizer


################################################################
# device config
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def init_seeds(seed=0, deterministic=True):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True  # 固定使用默认卷积算法
        torch.backends.cudnn.benchmark = False  # 关闭动态卷积算法优化
    else:
        # 当网络不会动态变化时，且输入shape固定，优化卷积计算
        torch.backends.cudnn.benchmark = True

# random seed config
init_seeds(2020)


################################################################
class Config:
    max_len = 150
    num_types = 14  # 标签种类
    num_tag_each_type = 4
    tokenizer = 'hfl/chinese-roberta-wwm-ext'


    ###################
    use_w2v = True
    w2v_dropout = 0.1

    # ptm：encoder选择的模型
    ptm_model = 'hfl/chinese-roberta-wwm-ext'  # 'hfl/chinese-roberta-wwm-ext-large'
    ptm_feat_size = 768
    num_ptm_layers = 12  # 提取特征所在的层级

    bert_out_size = ptm_feat_size

    # w2v
    w2v_feat_size = 300


    ###################
    # model
    num_flat_layers = 1
    use_ff_output = True

    # commmon
    flat_in_feat_size = 160
    hidden_size = 160  # out_size
    flat_out_feat_size = hidden_size

    # position embedding 的种类数
    num_pos = 4
    dim_pos = flat_in_feat_size
    pos_norm = False
    learnable_position = False  # position embedding 是否可学习
    four_pos_shared = True  # 是否共享每种 position embedding 的参数

    num_heads = 8
    scaled = False
    attn_dropout = 0.1
    hidden_dropout = 0.1

    layer_norm_eps = 1e-12
    dropout = 0.1


    ##################
    # train
    en_eval = True

    reshuffle = False

    # 交叉验证
    en_cross = True
    fold_num = 5

    # 验证集比例
    dev_rate = 0.2

    # 对抗训练
    en_fgm = True
    # 平均权重
    en_swa = True

    epochs = 10
    end_epoch = 10  # scheduler 参数
    batch_size = 16
    lr = {'ptm': 3e-5,
          'other': 3e-3,
          'crf': 3e-2}

    verbose = True
    cuda = True


    ##################
    # path
    # path config
    ROOT = os.path.abspath(__file__)[:-len("config.py")]

    DATA_PATH = ROOT + "data/origin/"
    ADDITIONAL_DATA_PATH = ROOT + "data/addition/"

    UTIL_PATH = ROOT + "utils/"

    MODEL_PATH = ROOT + "model/"
    MODULE_PATH = ROOT + "model/layers/"

    W2V_PATH = ROOT + 'data/w2v/'

    RESULT_PATH = ROOT + "result/"
    WEIGHT_PATH = ROOT + "weight/"
    LOG_PATH = ROOT + "log/"


################################################################
# 全局分词器
pku_seg = pkuseg.pkuseg()
tokenizer = MyTokenizer.from_pretrained(Config.tokenizer)


# 类别
type_list = ['position', 'name', 'movie', 'organization', 'company', 'game', 'book', 'address', 'scene',
            'government', 'email', 'mobile', 'QQ', 'vx']


type2id = {pred: i for i, pred in enumerate(type_list)}
id2type = {v: k for k, v in type2id.items()}


if __name__ == '__main__':
    path_list = [
        Config.LOG_PATH, Config.DATA_PATH, Config.ADDITIONAL_DATA_PATH, Config.UTIL_PATH, Config.MODEL_PATH,
        Config.MODULE_PATH, Config.RESULT_PATH, Config.WEIGHT_PATH, Config.W2V_PATH
    ]
    for path in path_list:
        if not os.path.exists(path):
            os.mkdir(path)