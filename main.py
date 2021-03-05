import os
import torch

from data.dataset import DatasetFactory
from model.trainer import Trainer
from model.mymodel import CascadeFLAT
from utils.postprocess.process import analyze_dev_results, generate_submit
from utils.tools import console
from config import Config


datamanager = DatasetFactory(config=Config)
trainer = Trainer()
model = CascadeFLAT().cuda()


def run_train_kflod():
    console.log(f" === RUNNING Train with {Config.fold_num} flod=== ")

    for fold_idx in range(Config.fold_num):
        weight_path = os.path.join(Config.WEIGHT_PATH,
                                    'model_{}.bin'.format(fold_idx))

        train_data = datamanager.fetch_dataset("train", fold_idx)
        dev_data = datamanager.fetch_dataset("dev", fold_idx)
        inputs = {
            'train_data': train_data,
            'dev_data': dev_data,
            'dev_res_file': os.path.join(Config.RESULT_PATH,'dev_result.json'),
            'epoch_start': 0
        }
        trainer.train(inputs, model)
        torch.save(model.state_dict(), weight_path)


        if Config.en_eval:
            inputs = {'data': dev_data, 'type_data': 'dev'}

            inputs['weight'] = weight_path
            inputs['outfile'] = os.path.join(
                Config.RESULT_PATH, 'dev_result_{}.json'.format(fold_idx))

            results = trainer.eval(inputs, model)
            console.log(results)

            if fold_idx == (Config.fold_num - 1):
                analyze_dev_results(Config.RESULT_PATH,
                                    os.path.join(Config.DATA_PATH, 'seg_train_data.json'),
                                    os.path.join(Config.RESULT_PATH, 'dev_result_all.json'),
                                    verbose=False)


def run_train():
    "Config.en_cross == False"

    console.log(" === RUNNING Train === ")

    weight_path = os.path.join(Config.WEIGHT_PATH, 'model.bin')

    train_data = datamanager.fetch_dataset("train")
    dev_data = datamanager.fetch_dataset("dev")

    inputs = {
        'train_data': train_data,
        'dev_data': dev_data,
        'dev_res_file': os.path.join(Config.RESULT_PATH, 'dev_result.json'),
        'epoch_start': 0
    }

    trainer.train(inputs, model)
    torch.save(model.state_dict(), weight_path)


    if Config.en_eval:
        inputs['weight'] = weight_path
        inputs['outfile'] = os.path.join(Config.RESULT_PATH, 'dev_result.json')

        results = trainer.eval(inputs, model)
        console.log(results)

        analyze_dev_results(Config.RESULT_PATH,
                            os.path.join(Config.DATA_PATH, 'seg_train_data.json'),
                            en_cross=False,
                            verbose=False)


def run_train_lossmask():
    from utils.postprocess.selfsupervise import generate_loss_mask
    from utils.preprocess.format import format_data

    # generate input tokens with mask
    # 需要先训练生成 dev_result_all.json 文件
    generate_loss_mask(os.path.join(Config.RESULT_PATH, 'dev_result_all.json'),
                       os.path.join(Config.DATA_PATH, 'seg_train_data.json'),
                       os.path.join(Config.DATA_PATH, 'seg_train_data_loss_mask.json'))

    format_data("train_loss_mask")

    console.log(" === Loss mask generated. Begin Training... === ")

    if Config.en_cross:
        run_train_kflod()
    else:
        run_train()


def run_test():

    console.log(" === RUNNING Test === ")

    test_data = datamanager.fetch_dataset("test")
    inputs = {'data': test_data, 'type_data': 'test'}

    if Config.en_cross:
        for fold_idx in range(Config.fold_num):
            inputs['weight'] = Config.WEIGHT_PATH + 'swa_model_{}.bin'.format(fold_idx)
            inputs['outfile'] = Config.RESULT_PATH + 'test_result_{}.json'.format(fold_idx)

            trainer.test(inputs, model)

        generate_submit([Config.RESULT_PATH + 'test_result_{}.json'.format(fold_idx) \
                         for fold_idx in range(Config.fold_num)],
                        Config.RESULT_PATH + 'predict_vote.csv',
                        cheat_with_clue=False)

    else:
        inputs['weight'] = Config.WEIGHT_PATH + 'swa_model.bin'
        inputs['outfile'] = Config.RESULT_PATH + 'test_result.json'

        trainer.test(inputs, model)

        generate_submit(Config.RESULT_PATH + 'test_result.json',
                        Config.RESULT_PATH + 'predict.csv',
                        cheat_with_clue=False)

        # generate_submit(Config.RESULT_PATH + 'test_result.json',
        #                 Config.RESULT_PATH + 'predict.csv',
        #                 cheat_with_clue=True,
        #                 clue_file=Config.ADDITIONAL_DATA_PATH + 'train_dev_data.json')



if __name__ == '__main__':
    if Config.en_cross:
        run_train_kflod()
    else:
        run_train()

    # run_test()

    # run_train_lossmask()

    del datamanager
    del trainer
    del model





