import json
import copy
import pandas as pd
from model.trainer import Trainer
from utils.tools import json_load_by_line, json_dump_by_line, console
from config import Config, type_list


def split_by_category(entities):
    result = dict((key, set()) for key in type_list)
    for e in entities:
        result[e['category']].add(e)
    return result


def analyze_dev_results(dev_res_dir, target_file, total_dev_res_file=None, en_cross=True, verbose=False):
    """
    dev_res_dir: 验证集预测结果路径
    target_file：数据的标签文件，此处是切分后的文件，'seg_train_data.json'
    total_dev_res_file: 输出保存整合的验证集预测结果，当生成 loss mask 时使用。
    """
    pred_data = []

    if en_cross:
        for i in range(Config.fold_num):
            pd = json_load_by_line(dev_res_dir + 'dev_result_{}.json'.format(i))
            pred_data.extend(pd)
    else:
        pred_data = json_load_by_line(dev_res_dir + 'dev_result.json')

    pred_data.sort(key=lambda x: x['sub_id'])
    pred_data.sort(key=lambda x: x['id'])

    if total_dev_res_file:
        json_dump_by_line(pred_data, total_dev_res_file)

    # 按照 id 与 sub id 建立两级字典
    target_dict = {}
    fp = open(target_file, 'r', encoding='utf-8')
    for d in fp.readlines():
        d = json.loads(d)
        if d['id'] not in target_dict:
            target_dict[d['id']] = {}
        if d['sub_id'] not in target_dict[d['id']]:
            target_dict[d['id']][d['sub_id']] = d

    # 统计计算结果
    f1_result = dict((key, {'correct_n': 0, 'pred_posi_n': 0, 'orig_posi_n': 0}) for key in type_list + ['all'])

    for p in pred_data:
        target = target_dict[p['id']][p['sub_id']]

        pred_c = split_by_category(p['entities'])
        target_c = split_by_category(target['entities'])

        precise_list = []
        recall_list = []

        for key in type_list:
            for e in pred_c[key]:
                if e in target_c[key]:    # true positive
                    f1_result[key]['correct_n'] += 1
                    f1_result['all']['correct_n'] += 1
                else:                     # false positive
                    precise_list.append(e)
            for e in target_c[key]:
                if e not in pred_c[key]:  # false negative
                    recall_list.append(e)

            f1_result[key]['pred_posi_n'] += len(pred_c[key])      # sum of positive
            f1_result[key]['orig_posi_n'] += len(target_c[key])    # sum of true
            f1_result['all']['pred_posi_n'] += len(pred_c[key])    # sum of positive
            f1_result['all']['orig_posi_n'] += len(target_c[key])  # sum of true

        if verbose and (precise_list or recall_list):
            console.log('\n false positive: ')
            console.log(precise_list)   # false positive
            console.log('false negative: ')
            console.log(recall_list)    # false negative

    for key in f1_result.keys():
        p, r, f1 = Trainer.calculate_f1(f1_result[key]['correct_n'],
                                        f1_result[key]['pred_posi_n'],
                                        f1_result[key]['orig_posi_n'],
                                        verbose=True)
        console.log('{:<12s}: precise {:0.6f} - recall {:0.6f} - f1 {:0.6f}'.format(key, p, r, f1))


def generate_submit(test_pred_file, file_out, cheat_with_clue=True, clue_file=""):
    """生成结果 submit. 由于数据集信息泄露的问题，验证数据中，可能存在 clue 中已经标记的数据. 说实话，这实际并没有什么意义

    Args:
        test_pred_file (str or list): model预测输出文件路径
        file_out (str): 结果保存路径
        cheat_with_clue (bool, optional): 是否直接将clue中标记的数据，在test数据中查找并标记. Defaults to True.
        clue_file (str, optional): clue 标记的数据 train 和 dev 数据集，预处理时已经整合到一起，'train_dev_data.json'. Defaults to "".
    """
    if isinstance(test_pred_file, str):
        data = json_load_by_line(test_pred_file)
    else:
        data = kflod_model_vote(test_pred_file)

    if cheat_with_clue:
        clue_data = json_load_by_line(clue_file)
        for d in data:
            if -1 < d['clue_id']:
                clue_d = clue_data[d['clue_id']]

                d['entities'] = clue_d['entities']
                for e in d['entities']:
                    e['privacy'] = clue_d['text'][e['pos_b']:e['pos_e'] + 1]

    # to csv
    csv_data = []
    base = 0
    for d in data:
        if d['sub_id'] == 0:
            base = 0
        for e in d['entities']:
            if '\n' in e['privacy']:
                print(d)
            pos_b = base + e['pos_b']
            pos_e = pos_b + len(e['privacy']) - 1
            csv_data.append([d['id'], e['category'], pos_b, pos_e, str(e['privacy'])])
        base += len(d['text'])

    csv_data.sort(key=lambda x: x[3])
    csv_data.sort(key=lambda x: x[2])
    csv_data.sort(key=lambda x: x[0])

    csv_data = pd.DataFrame(csv_data, columns=['ID', 'Category', 'Pos_b', 'Pos_e', 'Privacy'])
    csv_data.to_csv(file_out, index=False)


def kflod_model_vote(test_pred_file):
    # 输出 entity 预测的 阈值
    min_map = {'position': 5, 'name': 5, 'movie': 5, 'organization': 5,
                'company': 5, 'game': 5, 'book': 5, 'address': 5,
                'scene': 5, 'government': 5, 'email': 5, 'mobile': 5,
                'QQ': 5, 'vx': 5}

    data_list = [json_load_by_line(f) for f in test_pred_file]

    # 以下处理方式，可以处理 data_list 中每个 test_pred 长度不同，预测有偏差的情况
    # 简单按照 index 并行遍历，无法处理 长度不一致的情况
    data = []
    entities = {}  # for count
    for model_pred in data_list:
        for sample in model_pred:
            new_sample = copy.deepcopy(sample)
            new_sample['entities'] = []

            # vote 使用 k fold 训练时，每个 fold 保存的模型的预测结果进行 vote
            for e in sample['entities']:
                # count
                ekey = e['category'] + e['privacy'] + str(e['pos_b']) + str(e['pos_e'])
                if ekey not in entities:
                    entities[ekey] = 0
                entities[ekey] += 1

                # check and collect.  使用 == 保证只输出一次 ekey 对应的 entity 只输出一次
                if entities[ekey] == min_map[e['category']]:
                    new_sample['entities'].append(e)

            # output
            if len(new_sample['entities']) > 0:
                data.append(new_sample)

    return data