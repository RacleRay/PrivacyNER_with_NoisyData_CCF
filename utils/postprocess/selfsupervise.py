import copy
from utils.tools import json_load_by_line, json_dump_by_line



def generate_loss_mask(dev_res_file, origin_train_file, outfile):
    """根据在验证集的预测结果，将预测结果概率低的 entities 全部mask，生成新的训练集。
    依据来自交叉验证。

    Args:
        dev_res_file (str): 交叉验证的预测结果
        origin_train_file (str): 原训练数据, 'seg_train_data.json'
        outfile (str): 保存结果文件
    """
    dev_data = json_load_by_line(dev_res_file)

    # 预测统计
    record = []
    for sample in dev_data:
        new_sample = copy.deepcopy(sample)

        entity_count = {}
        for e in sample['entities']:
            key = e['category'] + e['privacy'] + str(e['pos_b']) + str(e['pos_e'])
            if key not in entity_count:
                entity_count[key] = [e, 0]
            entity_count[key] += 1

        new_sample['entities'] = entity_count
        record.append(new_sample)

    del dev_data


    # 添加 mask
    train_data = json_load_by_line(origin_train_file)
    check_count_list = ('name', 'movie', 'organization', 'company', 'game', 'book',
                         'government', 'position', 'address', 'scene')

    for idx in range(len(train_data)):
        if idx >= len(record): break

        d = train_data[idx] # origin
        pd = record[idx]    # pred

        assert d['id'] == pd['id']
        assert d['sub_id'] == pd['sub_id']

        # loss mask 位置在后续处理中会 标记为 0
        if 'loss_mask' not in d.keys():
            d['loss_mask'] = []

        for e in d['entities']:
            ekey = e['category'] + e['privacy'] + str(e['pos_b']) + str(e['pos_e'])
            if e['category'] in check_count_list:
                # mask 掉 低概率 预测
                if ekey not in pd['entities'] or pd['entities'][ekey][1] < 2: # count
                    pair = [e['pos_b'], e['pos_e']]
                    if pair not in d['loss_mask']:
                        d['loss_mask'].append(pair)

        for ekey in pd['entities'].keys():
            e = pd['entities'][ekey][0]
            # 在预测中出现，而在 origin data 中没有的 预测。
            if e not in d['entities'] and pd['entities'][ekey][1] > 1 and e['category'] in ('name', 'book', 'game'):
                pair = [e['pos_b'], e['pos_e']]
                if pair not in d['loss_mask']:
                    d['loss_mask'].append(pair)

    json_dump_by_line(train_data, outfile)