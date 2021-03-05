import os
import random

from torch.utils.data import Dataset

from utils.tools import json_load_by_line, aggregate_by_key, json_load, json_dump
from config import Config


class MyDataset(Dataset):
    """basic dataset 类"""
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class DatasetFactory:
    def __init__(self, config):
        self.config = config

        # token data format:
        #   'id': 原id,
        #   'sub_id': 切分后切分部分排序id,
        #   'text': encode 之后的 token id,
        #   'entities': entities,
        #   'lattice': lattice,
        #   'loss_mask': mask_seq （when use loss mask）
        train_data = json_load_by_line(Config.DATA_PATH + 'token_train_data.json')  # len 12871

        # 预处理时，切分了数据，所以，此时，通过一个 id 为 key的字典，聚合同id的 样本
        self.train_agg = aggregate_by_key(train_data, 'id')   # len 7596
        self.train_size = len(self.train_agg)

        del train_data

        if os.path.exists(Config.DATA_PATH + 'token_seg_train_data_loss_mask.json'):
            self.train_data_with_mask = json_load_by_line(Config.DATA_PATH + \
                                            'token_seg_train_data_loss_mask.json')
            self.train_data_with_mask_agg = aggregate_by_key(self.train_data_with_mask, 'id')
        else:
            self.train_data_with_mask = None
            self.train_data_with_mask_agg = None

        # train_map, 就是 id 的shuffle
        if self.config.reshuffle or not os.path.exists(Config.DATA_PATH + 'train_map.json'):
            self.train_map = [i for i in range(self.train_size)]
            random.shuffle(self.train_map)
            json_dump(self.train_map, Config.DATA_PATH + 'train_map.json')
        else:
            self.train_map = json_load(Config.DATA_PATH + 'train_map.json')

        # k fold
        self.part_size = int(len(self.train_map) / self.config.fold_num)
        self.train_size = int(len(self.train_map) * (1 - self.config.dev_rate))

    def fetch_dataset(self, data_type, fold_index=0):
        """
        data_type: train, dev, test
        fold_index: 小于 config.fold_num
        """
        if data_type in ('train', 'train_loss_mask'):
            data_agg = self.train_agg if data_type == 'train' else self.train_data_with_mask_agg

            # k fold: train 部分数据提取
            if self.config.en_cross:
                data_map = self.train_map[:fold_index * self.part_size]
                if fold_index + 1 < self.config.fold_num:
                    data_map += self.train_map[(fold_index + 1) * self.part_size:]
            else:
                data_map = self.train_map[:self.train_size]

            data = []
            for dindex in data_map:
                data.extend(data_agg[dindex])

        elif data_type in ('dev'):
            data_agg = self.train_agg if data_type == 'dev' else self.train_data_with_mask_agg

            # k fold: dev 部分数据提取
            if self.config.en_cross:
                if fold_index + 1 < self.config.fold_num:
                    data_map = self.train_map[fold_index * self.part_size: (fold_index + 1) * self.part_size]
                else:
                    data_map = self.train_map[fold_index * self.part_size:]
            else:
                data_map = self.train_map[self.train_size:]

            data = []
            for dindex in data_map:
                data.extend(data_agg[dindex])

        elif data_type == 'test':
            data = json_load_by_line(Config.DATA_PATH + 'token_test_data.json')

        return MyDataset(data)