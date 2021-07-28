import pickle
import torch
import numpy as np

from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from torchcontrib.optim import SWA
from utils.callbacks.earlystop import EarlyStopping
from torch.utils.tensorboard import SummaryWriter
from rich.progress import track
from tqdm import tqdm

from config import Config, id2type
from utils.tools import console, json_load_by_line, json_dump_by_line, aggregate_by_key
from model.advTrain import FGM


class MetricMeter:
    """储存 metric 累计数据的数据结构"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.val = val
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count


class Trainer:
    """
    func：
        train, eval, test: 用于训练模型流程，使用SWA + FGM
        collate_fn_test，collate_fn_train：定义了 生成 Model 输入数据的逻辑
        calculate_f1，result_stats，add_bieo：其他功能函数
    """
    def __init__(self, generate_result=True):
        self.loss_mnonitor = MetricMeter()
        self.correct_num = MetricMeter()
        self.pred_num = MetricMeter()
        self.true_num = MetricMeter()
        self.w2v_array = None

        if generate_result:
            self.text_map_file = {'dev': 'map_train_data.json',
                            'test': 'map_test_data.json'}
            self.token_data_set_file = {'dev': 'token_train_data.json',
                                   'test': 'token_test_data.json'}
            self.seg_data_set_file = {"dev": "seg_train_data.json",
                                  "test": "seg_test_data.json"}

    def train(self, train_input, model):
        train_data = train_input['train_data']
        dev_data = train_input['dev_data']
        epoch_start = train_input['epoch_start']

        ###### train data
        train_steps = int((len(train_data) - 1) / Config.batch_size) + 1
        train_dataloader = DataLoader(train_data,
                                      batch_size=Config.batch_size,
                                      collate_fn=self.collate_fn_train,
                                      shuffle=True)

        ###### 差分学习率
        params_lr = []
        for key, value in model.get_params().items():
            if key in Config.lr:
                params_lr.append({"params": value, 'lr': Config.lr[key]})
        optimizer = torch.optim.Adam(params_lr)
        # Stochastic Weight Averaging
        optimizer = SWA(optimizer)

        ###### callbacks
        early_stopping = EarlyStopping(model,
                                       Config.WEIGHT_PATH,
                                       mode='max',
                                       patience=3)
        learning_schedual = get_linear_schedule_with_warmup(
            optimizer,
            len(train_dataloader) // Config.batch_size // 2,
            len(train_dataloader) * Config.epochs)
        # 对抗训练
        fgm = FGM(model)

        ###### 训练过程记录
        writer = SummaryWriter(log_dir=Config.LOG_PATH, flush_secs=30)

        ####### 开始训练
        ending_flag = False
        detach_flag = False
        swa_flag = False
        for epoch in range(epoch_start, Config.epochs):

            console.log(f" ===== Epoch {epoch} ===== ")
            tqdm_loader = tqdm(train_dataloader, total=len(train_dataloader))

            for step, (inputs, targets, others) in enumerate(tqdm_loader):
                if epoch > 0 and step == 0:
                    model.detach_ptm(False)
                    detach_flag = False
                if epoch == 0 and step == 0:  #  首轮 detach
                    model.detach_ptm(True)
                    detach_flag = True

                # train #####################################################
                preds = model(inputs, en_decode=Config.verbose)
                loss = model.cal_loss(preds, targets, inputs['mask'])
                loss.backward()

                # 对抗训练
                if (not detach_flag) and Config.en_fgm:
                    fgm.attack(emb_name='word_embeddings')  # 在embedding上添加对抗扰动
                    preds_adv = model(inputs, en_decode=False)
                    loss_adv = model.cal_loss(preds_adv, targets, inputs['mask'])
                    loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                    fgm.restore(emb_name='word_embeddings')  # 恢复embedding参数

                # torch.nn.utils.clip_grad_norm(model.parameters(), 1)
                optimizer.step()
                optimizer.zero_grad()
                with torch.no_grad():
                    logs = {}
                    if Config.verbose:
                        # predict
                        pred_entity_point = model.find_entity(preds['pred'], inputs['mask'])

                        cn, pn, tn = self.result_stats(pred_entity_point,
                                                       others['raw_entity'])

                        self.loss_mnonitor.update(loss.cpu().numpy())
                        self.correct_num.update(cn)
                        self.pred_num.update(pn)
                        self.true_num.update(tn)

                        logs['loss'] = self.loss_mnonitor.avg
                        logs['precise'], logs['recall'], logs['f1'] = self.calculate_f1(self.correct_num.sum,
                                                                                        self.pred_num.sum,
                                                                                        self.true_num.sum,
                                                                                        verbose=Config.verbose)
                    else:
                        self.loss_mnonitor.update(loss.cpu().numpy())
                        logs['loss'] = self.loss_mnonitor.avg

                    writer.add_scalar("train/loss", logs['loss'])

                    # update lr
                    learning_schedual.step()

                    if step + 1 == train_steps:
                        model.eval()
                        # dev #####################################################
                        eval_inputs = {
                            'data': dev_data,
                            'type_data': 'dev',
                            'outfile': train_input['dev_res_file']
                        }
                        dev_result = self.eval(eval_inputs, model)

                        logs['dev_loss'] = dev_result['loss']
                        logs['dev_precise'] = dev_result['precise']
                        logs['dev_recall'] = dev_result['recall']
                        logs['dev_f1'] = dev_result['f1']

                        writer.add_scalar("dev/loss", logs['loss'])

                        if logs['dev_f1'] > 0.80:
                            torch.save(model.state_dict(),
                                "{}/auto_save_{:.6f}.bin".format(Config.WEIGHT_PATH, logs['dev_f1']))

                        # 更新 swa 模型所有权重
                        if (epoch > 3 or swa_flag) and Config.en_swa:
                            optimizer.update_swa()
                            swa_flag = True

                        early_stop, best_score = early_stopping(logs['dev_f1'])

                        # end #####################################################
                        if (epoch + 1 == Config.epochs) or early_stop:
                            ending_flag = True
                            if swa_flag:
                                # 训练结束时使用收集到的swa moving average
                                optimizer.swap_swa_sgd()
                                # optimizer.bn_update(
                                #     train_dataloader,
                                #     model)  # 更新BatchNorm的 running mean

                        model.train()

                tqdm_loader.set_postfix(**logs)

                if ending_flag:
                    return best_score

            self.loss_mnonitor.reset()
            self.correct_num.reset()
            self.pred_num.reset()
            self.true_num.reset()

    def eval(self, eval_inputs, model):
        dev_data = eval_inputs['data']
        type_data = eval_inputs['type_data']
        outfile = eval_inputs['outfile']
        dev_dataloader = DataLoader(dev_data,
                                    batch_size=Config.batch_size,
                                    collate_fn=self.collate_fn_train)

        entity_result = []
        result = {}
        metrics_data = {
            "loss": 0,
            "correct_num": 0,
            "pred_num": 0,
            "true_num": 0,
            "sampled_num": 0
        }

        if 'weight' in eval_inputs:
            model.load_state_dict(torch.load(eval_inputs['weight']))

        with torch.no_grad():
            model.eval()
            batch_index = 1
            console.log(" === Running eval === ")
            for inputs, targets, others in track(dev_dataloader, description="Evaluating ..."):
                batch_index += 1

                preds = model(inputs)
                loss = model.cal_loss(preds, targets, inputs['mask'])
                pred_entity_point = model.find_entity(preds['pred'], inputs['mask'])

                cn, pn, tn = self.result_stats(pred_entity_point,
                                               others['raw_entity'])

                metrics_data['correct_num'] += cn
                metrics_data['pred_num'] += pn
                metrics_data['true_num'] += tn
                metrics_data['loss'] += float(loss.cpu().numpy())
                metrics_data['sampled_num'] += 1

                for iid, sub_id, entities in zip(others['id'],
                                                 others['sub_id'],
                                                 pred_entity_point):
                    entity_result.append({
                        'id': iid,
                        'sub_id': sub_id,
                        'entities': entities
                    })

        result['loss'] = metrics_data['loss'] / metrics_data['sampled_num']
        result['precise'], result['recall'], result['f1'] = self.calculate_f1(metrics_data['correct_num'],
                                                                              metrics_data['pred_num'],
                                                                              metrics_data['true_num'],
                                                                              verbose=True)
        self.generate_results(entity_result, type_data, outfile)

        return result

    def test(self, test_inputs, model):
        test_data = test_inputs['data']
        type_data = test_inputs['type_data']
        outfile = test_inputs['outfile']

        test_dataloader = DataLoader(test_data,
                                     batch_size=Config.batch_size,
                                     collate_fn=self.collate_fn_test)

        if 'weight' in test_inputs:
            model.load_state_dict(torch.load(test_inputs['weight']))

        with torch.no_grad():
            model.eval()
            entity_result = []

            batch_index = 0

            console.log(" === Running predict === ")
            for inputs, others in track(test_dataloader, description="Predicting ..."):
                batch_index += 1

                preds = model(inputs)
                pred_entity_point = model.find_entity(preds['pred'], inputs['mask'])

                for iid, sub_id, entities in zip(others['id'], others['sub_id'], pred_entity_point):
                    entity_result.append({
                        'id': iid,
                        'sub_id': sub_id,
                        'entities': entities
                    })
        self.generate_results(entity_result, type_data, outfile)

        return entity_result

    @staticmethod
    def result_stats(ner_pred, ner_true):
        correct_num = pred_num = true_num = 0
        for batch_index in range(len(ner_pred)):
            for ner in ner_pred[batch_index]:
                if ner in ner_true[batch_index]:
                    correct_num += 1
            pred_num += len(ner_pred[batch_index])
            true_num += len(ner_true[batch_index])
        return correct_num, pred_num, true_num

    @staticmethod
    def calculate_f1(correct_num, pred_num, true_num, verbose=False):
        """
        correct_num: true positive.
        pred_num: true positive + false positive.
        true_num: true positive + false negative.
        """
        if correct_num == 0 or pred_num == 0 or true_num == 0:
            precise = 0.0
            recall = 0.0
            f1 = 0.0
        else:
            precise = correct_num / pred_num
            recall = correct_num / true_num
            f1 = 2 * precise * recall / (precise + recall)
        if verbose:
            return precise, recall, f1
        else:
            return f1

    def generate_results(self, pred, data_type, output_file):
        map_data = json_load_by_line(Config.DATA_PATH + self.text_map_file[data_type])
        token_data = json_load_by_line(Config.DATA_PATH + self.token_data_set_file[data_type])
        seg_data = json_load_by_line(Config.DATA_PATH + self.seg_data_set_file[data_type])

        self.text_map = aggregate_by_key(map_data, 'id')
        self.token_data = aggregate_by_key(token_data, 'id')
        self.seg_data = aggregate_by_key(seg_data, 'id')

        del map_data
        del token_data
        del seg_data

        result = []
        for p in pred:     # p: 'id':3616  'sub_id':0  'entities':[[3, 5, 7]]
            d = self.text_map[p['id']][p['sub_id']]  #   'id':3616  'sub_id':0  'text':'奥尔堡vs凯尔特推荐：10'  'text_map':[-1, 0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 13]
            td = self.token_data[p['id']][p['sub_id']]  #   'id':3616  'sub_id':0  'text':[101, 1952, 2209, 1836, 8349, 1132, 2209, 4294, 2972, 5773, 8038, 8108, 102]  'entities':[{'category': 3, 'pos_b': 1, 'pos_e': 3}, {'category': 3, 'pos_b': 5, 'pos_e': 7}]  'lattice':[[1, 3, 279444], [8, 9, 878]]  'loss_mask':None
            sd = self.seg_data[p['id']][p['sub_id']]  #    'id':3616   'sub_id':0   'clue_id':6000   'text':'奥尔堡vs凯尔特推荐：10'   'entities':[{'category': 'organization', 'pos_b': 0, 'pos_e': 2, 'privacy': '奥尔堡'}, {'category': 'organization', 'pos_b': 5, 'pos_e': 7, 'privacy': '凯尔特'}]   'lattice':[['奥尔堡', 0, 2], ['推荐', 8, 9]]

            sample = {'id': d['id'], 'sub_id': d['sub_id'], 'clue_id': sd['clue_id'],
                      'text': d['text'], 'entities': []}

            for entity in p['entities']:
                # 映射到原 text 中
                try:
                    pos_b = d['text_map'][entity[1]]   # d['text_map'] 映射会原 text 中文本
                except IndexError:
                    continue

                if entity[2] + 1 == len(d['text_map']):  # 没有用啊
                    pos_e = d['text_map'][entity[2]] + 1
                else:
                    pos_e = d['text_map'][entity[2] + 1]
                privacy = d['text'][pos_b: pos_e].strip()

                raw_entity = {'privacy': privacy,
                              'category': id2type[entity[0]],
                              'pos_b': pos_b,
                              'pos_e': pos_b + len(privacy) - 1}

                # 在 tokenize 的 myencode 方法处理后的 text 中
                sample['entities'].append(raw_entity)

                # # for debug
                # token_entity = {'category': entity[0], 'pos_b': entity[1], 'pos_e': entity[2]}
                # if raw_entity not in sd['entities'] and token_entity in td['entities']:
                #     print(token_entity)
                #     print(raw_entity)
                #     print(sd['text'])
                #     print(sd['entities'])

            result.append(sample)

        json_dump_by_line(result, output_file)

    def collate_fn_test(self, batch):

        token_ids_padded = []
        text_mask = []
        char_len = []

        iid = []        # 原数据集中id
        sub_id = []     # 切分后子 id
        token_ids = []  # 未pad的token ids

        word = []
        word_mask = []
        word_pos_b = []
        word_pos_e = []

        model_input = dict()
        raw_data = dict()

        max_len = 0
        lattice_conbine_len = 0
        for sample in batch:
            # sample['text']： 这里是 token id，包含特殊token
            text_length = len(sample['text'])
            max_len = max_len if max_len > text_length else text_length
            lattice_length = len(sample['lattice'])
            if text_length + lattice_length > lattice_conbine_len:
                lattice_conbine_len = text_length + lattice_length

        for sample in batch:
            iid.append(sample['id'])
            sub_id.append(sample['sub_id'])
            token_ids.append(sample['text'])

            text_length = len(sample['text'])
            char_len.append(text_length)
            lattice_len = len(sample['lattice'])

            # pad and mask   to max_len
            token_ids_padded.append(sample['text'] + [0] * (max_len - text_length))
            text_mask.append([1] * text_length + [0] * (max_len - text_length))

            if Config.use_w2v:
                word_ = [0] * text_length         # word index
                word_pos_b_ = [0] * text_length   # head index
                word_pos_e_ = [0] * text_length   # tail index
                # mask指出lattice所在位置，（concat之后）
                word_mask_ = [0] * text_length + [1] * lattice_len + \
                    [0] * (lattice_conbine_len - lattice_len - text_length)

                for lattice in sample['lattice']:
                    word_.append(lattice[2])
                    word_pos_b_.append(lattice[0])
                    word_pos_e_.append(lattice[1])

                word_ += [0] * (lattice_conbine_len - len(word_))  # word index
                word_pos_b_ += [0] * (lattice_conbine_len - len(word_pos_b_))    # head index
                word_pos_e_ += [0] * (lattice_conbine_len - len(word_pos_e_))    # tail index

                word.append(word_)
                word_mask.append(word_mask_)
                word_pos_b.append(word_pos_b_)
                word_pos_e.append(word_pos_e_)

        token_ids_padded = torch.tensor(token_ids_padded).cuda()
        text_mask = torch.tensor(text_mask).float().cuda()

        model_input = {'text': token_ids_padded, 'mask': text_mask}
        raw_data = {'id': iid, 'sub_id': sub_id, 'token_ids': token_ids}

        if Config.use_w2v:
            # 作为 输入tensor，不进行更新
            model_input['word_idx'] = torch.tensor(word).cuda()
            model_input['word_mask'] = torch.tensor(word_mask).float().cuda()
            model_input['word_pos_b'] = torch.tensor(word_pos_b).long().cuda()
            model_input['word_pos_e'] = torch.tensor(word_pos_e).long().cuda()
            model_input['char_len'] = torch.tensor(char_len).long()

        # model_input
        # 'text'：padded token ids
        # 'mask': mask of padded token ids
        # 'word_idx': words of lattice
        # 'word_mask': mask of lattice, padded to the max seq len + lattice len
        # 'word_pos_b': head position of lattice
        # 'word_pos_e': tail position of lattice
        # 'char_len': length of token ids, used to build flat inputs

        # raw_data
        # 'id': sample id
        # 'sub_id': cut sub id of a samople
        # 'token_ids': token ids without pad
        return model_input, raw_data

    def collate_fn_train(self, batch):
        inputs, others = self.collate_fn_test(batch)
        max_len = inputs['mask'].size(1)

        loss_mask = []
        ner_label = []
        raw_entity = []

        for batch_index, sample in enumerate(batch):
            # loss mask 将 token ids 中的部分位置，设置为0
            if sample['loss_mask'] is not None:
                loss_mask.append(sample['loss_mask'] + [0] * (max_len - len(sample['loss_mask'])))
            else:
                loss_mask.append([1] * len(sample['text']) + [0] * (max_len - len(sample['text'])))

            _ner_label = np.zeros([Config.num_types, max_len])
            _raw_entity = []
            # 分类别标记 ner label
            for entity in sample['entities']:
                _raw_entity.append([entity['category'], entity['pos_b'], entity['pos_e']])
                self.add_bieo(_ner_label[entity['category']], entity['pos_b'], entity['pos_e'])

            ner_label.append(_ner_label)
            raw_entity.append(_raw_entity)

        others.update({'raw_entity': raw_entity})

        ner_label = torch.tensor(ner_label).long()   # [batch, num_types, max_len]
        loss_mask = torch.tensor(loss_mask).long()   # [batch, max_len]
        targets = {'y_true': ner_label.cuda(), 'loss_mask': loss_mask.cuda()}

        # targets：
        #  'y_true': 14种类别，14行标记，每一行 用 0表示非目标，1表示开头，2表示中间，3表示结尾
        #  'loss_mask': char 字符位于 max len 中的 mask
        return inputs, targets, others

    @staticmethod
    def add_bieo(line, s, e):
        "line： 为输入的某一个类别的待标记序列"
        # 转换成 0 0 0 1 2 2 2 2 3 0 0 0
        if s == e:
            if line[s] == 0:
                line[s] = 1
            return
        else:
            # 检测是否已经标记
            for bioe in line[s:e + 1]:
                if bioe != 0:
                    return
            line[s] = 1
            line[s + 1:e] = 2
            line[e] = 3
            return