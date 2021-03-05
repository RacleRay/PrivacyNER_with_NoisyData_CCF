import time
import os
import json
import copy
import pickle
import numpy
import re
from config import Config, pku_seg, tokenizer, type_list, type2id, id2type
from rich.progress import track
from utils.tools import console


# word2vec for lattice model
# 先处理 word2vec 文件，得到w2v_vocab.pkl数据
if os.path.exists(os.path.join(Config.W2V_PATH, 'w2v_vocab.pkl')):
    word_dict = pickle.load(open(Config.W2V_PATH + 'w2v_vocab.pkl', 'rb'))
    word_dict = {word: idx for idx, word in enumerate(word_dict)}


def format_data(data_type):
    """
    data_type: "train" or "test" or 'train_loss_mask'

    处理输出文件：
        map: 原text 与 tokenizer处理后字符index对应原text中的index 的映射
        token: 模型输入
        seg：输入数据，切分后的结果
      若为 'train_loss_mask' type，输出为 token 文件，其他数据可以使用 "train" 阶段生成的数据
        见 dataset.py DatasetFactory 中 token 的载入逻辑
    """
    out_data = {'seg': [],
                'token': [],
                'map': []}

    # read clud data, get len_map
    # 这属于比赛数据的信息泄露，知道它加入数据噪声的来源是 clue 数据集，正常情况下，不会这么操作。
    clue_data = []
    len_map = [[] for _ in range(55)]

    fp = open(Config.ADDITIONAL_DATA_PATH + 'train_dev_data.json','r', encoding='utf-8')
    for d in fp.readlines():
        d = json.loads(d)
        clue_data.append(d)
        len_map[len(d['text'])].append(d)
    fp.close()

    ############ 处理基本数据
    if data_type == 'train':
        in_file = 'train_data.json'
    elif data_type == 'train_loss_mask':
        in_file = 'seg_train_data_loss_mask.json'
        out_data = {'token': []}
    else:
        in_file = 'test_data.json'
    file_path = os.path.join(Config.DATA_PATH, in_file)
    fp = open(file_path, 'r', encoding='utf-8')

    for line in track(fp.readlines(), description="Processing original data..."):
        item = json.loads(line)

        if data_type in ('train', 'test'):
            samples = split_text(item, Config.max_len - 2, len_map)  # max_len - 2 不计入 special token
            for sample in samples:
                sample['lattice'] = get_lattice_word(sample)
                # text_token：模型输入
                # text_map：原text 与 tokenizer处理后字符index对应原text中的index 的映射
                text_token, text_map = convert_to_token(sample, Config.max_len)
                out_data['token'].append(text_token)
                out_data['map'].append(text_map)

            # seg： 输入数据，切分后的结果
            out_data['seg'].extend(samples)

        elif data_type == 'train_loss_mask':
            # 只取 text_token， text_token中进行了 mask，将 'loss_mask' 位置 mask
            # 因为这是在 训练多轮之后，再更改训练数据，所以，只需要取得更新的数据即可
            for sample in item:
                text_token, text_map = convert_to_token(sample, Config.max_len)
                out_data['token'].append(text_token)

    fp.close()

    ############ 处理外部数据
    if data_type == 'train':
        ccf_data = out_data['seg']

        ### process
        clue_out_ccf_data = []
        ccf_in_clue_data = []

        for d in ccf_data:
            if d['clue_id'] < 12082 and d['clue_id'] > 0:
                ccf_in_clue_data.append(d['clue_id'])

        # 加入，在训练数据中没有出现过的 外部数据
        index = ccf_data[-1]['id'] + 1
        for d in clue_data:
            if d['id'] not in ccf_in_clue_data:
                new_d = {'id': index,
                        'sub_id': 0,
                        'clue_id': d['id'],
                        'text': d['text'],
                        'entities': d['entities']}
                index += 1
                clue_out_ccf_data.append(new_d)

        for d in track(clue_out_ccf_data, description="Processing additional data..."):
            samples = split_text(d, Config.max_len - 2, len_map)
            for sample in samples:
                sample['lattice'] = get_lattice_word(sample)
                # text_token：模型输入
                # text_map：原text 与 tokenizer处理后字符index对应原text中的index 的映射
                text_token, text_map = convert_to_token(sample, Config.max_len)
                out_data['token'].append(text_token)
                out_data['map'].append(text_map)

            # seg： 输入数据，切分后的结果
            out_data['seg'].extend(samples)

    ############ 保存中间数据
    for key in out_data.keys():
        file_path = os.path.join(Config.DATA_PATH, '{}_'.format(key) + in_file)
        fp = open(file_path, 'w+', encoding='utf-8')
        for d in track(out_data[key], description=f"Saving {key} data..."):
            string = json.dumps(d, ensure_ascii=False) + '\n'
            fp.write(string)
        fp.flush()
        time.sleep(2)
        fp.close()
        console.log(f"Saved file {file_path}")


##############################################################################

def split_text(sample, max_len, len_map):
    new_samples = []

    # split_by_clue：数据泄露，提取分离噪声数据
    for news in split_by_clue(sample, len_map):
        new_samples.extend(split_by_len(news, max_len))

    new_samples.sort(key=lambda x: x['sub_id'])   # sub_id: 切分后的新数据，位于原text中的起始位置

    for index, sample in enumerate(new_samples):  # 切分后，统一重排 index
        sample['sub_id'] = index

    return new_samples


def split_by_clue(sample_clue, len_map):
    """将噪声数据，提取分离

    这属于比赛数据的信息泄露，知道它加入数据噪声的来源是 clue 数据集，正常情况下，不会这么操作。函数内容也没什么意思"""
    out_samples = []

    def split_(sample_):
        "递归处理"
        nonlocal out_samples

        a = len_map[:len(sample_['text']) + 1]
        a.reverse() # 排序长文本先遍历

        for a_len_map in a:
            for clue_d in a_len_map:
                # 在 比赛数据 中 找 clue 数据
                start_index = sample_['text'].find(clue_d['text'])

                # 替换为 中文符号再查询
                if start_index == -1 and ',' in clue_d['text']:
                    if 'text_2' not in clue_d.keys():
                        clue_d['text_2'] = re.sub(r',', '，', clue_d['text'])
                    start_index = sample_['text'].find(clue_d['text_2'])

                if start_index != -1:  # 抽取出 随机插入的 clue数据，这是比赛数据加噪声的方式泄露了
                    end_index = start_index + len(clue_d['text']) - 1

                    # sub_id 记录 重新生成的样本 在原串中的 起始位置
                    new_sample = {'id': sample_['id'],
                                  'sub_id': sample_['sub_id'] + start_index,
                                  'clue_id': clue_d['id'],
                                  'text': sample_['text'][start_index:end_index + 1],
                                  'entities': []}

                    # 修正 新 数据 标注
                    for e in sample_['entities']:
                        if e['pos_b'] >= start_index and e['pos_e'] <= end_index:
                            new_e = copy.deepcopy(e)
                            new_e['pos_b'] = e['pos_b'] - start_index
                            new_e['pos_e'] = e['pos_e'] - start_index
                            new_sample['entities'].append(new_e)
                    out_samples.append(new_sample)

                    # 重新生成 前面没有重复部分
                    if start_index > 0:
                        rest_sample_0 = {'id': sample_['id'], 'sub_id': sample_['sub_id'],
                                         'text': sample_['text'][:start_index], 'entities': []}
                        for e in sample_['entities']:
                            if e['pos_e'] < start_index:
                                new_e = copy.deepcopy(e)
                                rest_sample_0['entities'].append(new_e)
                        split_(rest_sample_0)

                    # 重新生成 后面没有重复部分
                    if end_index + 1 < len(sample_['text']):
                        rest_sample_1 = {'id': sample_['id'], 'sub_id': sample_['sub_id'] + end_index + 1,
                                         'text': sample_['text'][end_index + 1:], 'entities': []}
                        for e in sample_['entities']:
                            if e['pos_b'] > end_index:
                                new_e = copy.deepcopy(e)
                                new_e['pos_b'] = e['pos_b'] - end_index - 1
                                new_e['pos_e'] = e['pos_e'] - end_index - 1
                                rest_sample_1['entities'].append(new_e)
                        # 继续向后找
                        split_(rest_sample_1)
                    return

        new_sample = {'id': sample_['id'], 'sub_id': sample_['sub_id'], 'clue_id': -1,
                      'text': sample_['text'], 'entities': copy.deepcopy(sample_['entities'])}
        out_samples.append(new_sample)

    sample_clue['sub_id'] = 0
    split_(sample_clue)
    return out_samples


def split_by_len(sample_in, max_len):
    "将输入按最大长度进行切分"
    if len(sample_in['text']) <= max_len:
        return [sample_in]

    out_samples = []
    right_limit = 0
    rest_text = sample_in['text']
    while len(rest_text) > max_len:

        new_sample = copy.deepcopy(sample_in)
        new_sample['entities'] = []
        for char_index in range(max_len - 1, -1, -1):

            # 最右边开始，按照 '，', '。', '!', '?' 分割
            if (rest_text[char_index] in ('，', '。', '!', '?')) or char_index == 0:
                if char_index == 0:
                    char_index = max_len - 1
                left_limit = right_limit
                right_limit += char_index + 1
                new_sample['text'] = rest_text[:char_index + 1]
                new_sample['sub_id'] = sample_in['sub_id'] + left_limit

                for entity in sample_in['entities']:
                    if entity['pos_b'] >= left_limit and entity['pos_e'] < right_limit:
                        new_entity = copy.deepcopy(entity)
                        new_entity['pos_b'] = entity['pos_b'] - left_limit
                        new_entity['pos_e'] = entity['pos_e'] - left_limit
                        new_sample['entities'].append(new_entity)

                rest_text = rest_text[char_index + 1:]
                out_samples.append(new_sample)
                break

    # 剩余部分
    left_limit = right_limit
    new_sample = copy.deepcopy(sample_in)
    new_sample['text'] = rest_text
    new_sample['entities'] = []
    new_sample['sub_id'] = sample_in['sub_id'] + left_limit

    for entity in sample_in['entities']:
        if entity['pos_b'] >= left_limit:
            new_entity = copy.deepcopy(entity)
            new_entity['pos_b'] = entity['pos_b'] - left_limit
            new_entity['pos_e'] = entity['pos_e'] - left_limit
            new_sample['entities'].append(new_entity)

    out_samples.append(new_sample)
    return out_samples


##############################################################################

def get_lattice_word(sample):
    lattice_word = pkuseg_cut(sample['text'])
    # 按 词长度 升序   按 结束位置 升序
    lattice_word.sort(key=lambda x: len(x[0]))
    lattice_word.sort(key=lambda x: x[2])
    return lattice_word


def pkuseg_cut(text):
    "获取 lattice 所需的数据:  [(word, start idx, end idx), ...]"
    index = 0
    word_list = []
    for word in pku_seg.cut(text):
        word_len = len(word)
        if word_len > 1 and is_all_chinese(word):
            word_list.append((word, index, index + word_len - 1))
        index += word_len
    return word_list


##############################################################################

def convert_to_token(sample, max_length):
    # add_special_tokens: [cls]  [sep] ...
    text = tokenizer.my_encode(sample['text'],
                               max_length=max_length,
                               add_special_tokens=True,
                               truncation=True)

    # text_map: 处理后字符index对应原text中的index 的映射
    # raw2decode: 原text中字符的index对应处理后字符的index 的映射
    text_map, raw2decode = tokenizer.get_token_map(sample['text'])

    lattice = []  # [start idx，end idx, 词在 word2vec中的index]
    for lword in sample['lattice']:
        if lword[0] in word_dict:
            lword_index = word_dict[lword[0]]
            lattice.append([raw2decode[lword[1]], raw2decode[lword[2]], lword_index])

    entities = []
    if 'entities' in sample.keys():
        for entity in sample['entities']:
            entities.append({"category": type2id[entity['category']],
                                "pos_b": raw2decode[entity['pos_b']],
                                "pos_e": raw2decode[entity['pos_e']]})  # 在 decode 上训练预测模型

    mask_seq = None
    if 'loss_mask' in sample.keys():  # 通过 在 selfsupervise.py 文件中的 generate_loss_mask 产生该数据
        mask_seq = numpy.ones(len(text))
        # mask 位置 置为 0
        for e in sample['loss_mask']:
            e_s = raw2decode[e[0]]
            e_e = raw2decode[e[1]]
            mask_seq[e_s: e_e + 1] = 0
        mask_seq = mask_seq.tolist()

    # sub_id: 切分后切分部分排序id
    return {
                'id': sample['id'],
                'sub_id': sample['sub_id'],
                'text': text,  # encode之后的结果 token id
                'entities': entities,
                'lattice': lattice,
                'loss_mask': mask_seq
            }, \
            {
                'id': sample['id'],
                'sub_id': sample['sub_id'],
                'text': sample['text'],  # 原 text
                'text_map': text_map
            }


#########################################################################

def get_clue_data():
    # 根据文本长度分组
    clue_data = []
    len_map = [[] for _ in range(55)]

    fp = open(Config.ADDITIONAL_DATA_PATH + 'train_dev_data.json',
              'r', encoding='utf-8')
    for d in fp.readlines():
        d = json.loads(d)
        clue_data.append(d)
        len_map[len(d['text'])].append(d)

    fp.close()

    return len_map



def is_all_chinese(word_str):
    for c in word_str:
        if not '\u4e00' <= c <= '\u9fa5':
            return False
    return True