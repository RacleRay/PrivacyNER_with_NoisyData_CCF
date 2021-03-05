from utils.preprocess.process import convert2json, convert2json_clue, process_wordvec
from utils.preprocess.format import format_data
from utils.tools import show_runtime
from config import Config


if __name__ == '__main__':
    ######################################################
    # deal with origin data
    @show_runtime
    convert2json(Config.DATA_PATH + 'train_data/',
                 Config.DATA_PATH + 'train_data.json',
                 label_dir=Config.DATA_PATH + 'train_label/')

    @show_runtime
    convert2json(Config.DATA_PATH + 'test_data/',
                 Config.DATA_PATH + 'test_data.json')

    # deal with additional data
    @show_runtime
    convert2json_clue(
        [
            Config.ADDITIONAL_DATA_PATH + 'train.json',
            Config.ADDITIONAL_DATA_PATH + 'dev.json'
        ],
        Config.ADDITIONAL_DATA_PATH + 'train_dev_data.json'
    )

    # deal word2vec
    @show_runtime
    process_wordvec(Config.W2V_PATH + 'sgns.merge.word',
                    Config.W2V_PATH + 'w2v_vector.pkl',
                    Config.W2V_PATH + 'w2v_vocab.pkl')

    ######################################################
    # 结构化数据
    @show_runtime
    format_data("train")

    @show_runtime
    format_data("test")