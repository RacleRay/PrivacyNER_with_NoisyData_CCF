from bert4keras.snippets import sequence_padding, DataGenerator
import pandas as pd
import os


def read_txt(filename, use_line=True):
    """
    读取 txt 数据
    filename : str
    use_line : bool
    return   : list
    """
    with open(filename, 'r', encoding='utf8') as f:
        if use_line:
            ret = f.readlines()
        else:
            ret = f.read()
    return ret


def cut_sent(txt, symbol, max_len=250):
    """
    将一段文本切分成多个句子
    txt     : str
    symbol  : list e.g ['。', '！', '？', '?']
    max_len : int
    return  : list
    """
    new_sentence = []
    sen = []
    # 使用 symbol 对文本进行切分
    for i in txt:
        if i in symbol and len(sen) != 0:
            if len(sen) <= max_len:
                sen.append(i)
                new_sentence.append(''.join(sen))
                sen = []
                continue
            # 对于超过 max_len 的句子，使用逗号进行切分
            else:
                sen.append(i)
                tmp = ''.join(sen).split('，')
                for j in tmp[:-1]:
                    j += '，'
                    new_sentence.append(j)
                new_sentence.append(tmp[-1])
                sen = []
                continue
        sen.append(i)

    # 如果最后一个 sen 没有 symbol ，则加入切分的句子中。
    if len(sen) > 0:
        # 对于超过 max_len 的句子，使用逗号进行切分
        if len(sen) <= max_len:
            new_sentence.append(''.join(sen))
        else:
            tmp = ''.join(sen).split('，')
            for j in tmp[:-1]:
                j += '，'
                new_sentence.append(j)
            new_sentence.append(tmp[-1])
    return new_sentence


def agg_sent(text_list, symbol, max_len, treshold):
    """
    将文本切分成句子，然后将尽量多的合在一起，如果小于 treshold 就不拆分
    text_list : list
    symbol    : list e.g ['。', '！', '？', '?']
    max_len  : int
    treshold  : int
    return    : list, list
    """
    cut_text_list = []
    cut_index_list = []
    for text in text_list:

        temp_cut_text_list = []
        text_agg = ''
        # 如果没超过 treshold ，则将文本放入
        if len(text) < treshold:
            temp_cut_text_list.append(text)
        else:
            # 将一段文本切分成多个句子
            sentence_list = cut_sent(text, symbol, max_len)
            # 尽量多的把句子合在一起
            for sentence in sentence_list:
                if len(text_agg) + len(sentence) <= treshold:
                    text_agg += sentence
                else:
                    temp_cut_text_list.append(text_agg)
                    text_agg = sentence
            # 加上最后一个句子
            temp_cut_text_list.append(text_agg)

        cut_index_list.append(len(temp_cut_text_list))
        cut_text_list += temp_cut_text_list

    return cut_text_list, cut_index_list


def gen_data(label_path, data_path, output_path, output_file, symbol, max_len,
             treshold):
    """
    生成IOB标记的数据

    label_path : 标签路径
    data_path  : 数据路径
    output_path: 输出路径
    output_file: 输出文件
    symbol     : list e.g ['。', '！', '？', '?']
    max_len    : int
    treshold   : int
    """
    q_dic = {}
    tmp = pd.read_csv(label_path).values
    for _, entity_cls, start_index, end_index, entity_name in tmp:
        start_index = int(start_index)
        end_index = int(end_index)
        length = end_index - start_index + 1
        for r in range(length):
            if r == 0:
                q_dic[start_index] = ("B-%s" % entity_cls)
            else:
                q_dic[start_index + r] = ("I-%s" % entity_cls)

    content_str = read_txt(data_path)
    cut_text_list, cut_index_list = agg_sent(content_str, symbol, max_len,
                                             treshold)
    i = 0
    for idx, line in enumerate(cut_text_list):
        output_path_ = "%s/%s-%s-new.txt" % (output_path, output_file, idx)
        with open(output_path_, "w", encoding="utf-8") as w:
            for str_ in line:
                if str_ is " " or str_ == "" or str_ == "\n" or str_ == "\r":
                    pass
                else:
                    if i in q_dic:
                        tag = q_dic[i]
                    else:
                        tag = "O"  # 大写字母O
                    w.write('%s %s\n' % (str_, tag))
                i += 1
            w.write('%s\n' % "END O")


def load_data(filename):
    """
    加载生成的数据
    filename : str
    return   : list
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        f = f.read()
        # 按照 '\n\n' 获取数据数据（聚合的句子）
        for l in f.split('\n\n'):
            if not l:
                continue
            d, last_flag = [], ''
            for c in l.split('\n'):
                try:
                    char, this_flag = c.split(' ')
                except:
                    print('Exception:{}end'.format(c))
                    continue
                if this_flag == 'O' and last_flag == 'O':
                    d[-1][0] += char
                elif this_flag == 'O' and last_flag != 'O':
                    d.append([char, 'O'])
                elif this_flag[:1] == 'B':
                    d.append([char, this_flag[2:]])
                else:
                    try:
                        d[-1][0] += char
                    except:
                        print(l)
                        print(d)
                        continue
                last_flag = this_flag
            D.append(d)
    return D


def checkout(filename):
    """
    检查提交数据
    filename : str
    """
    all_lines = read_txt(filename)
    for line in all_lines:
        if not line.split('\n')[-1] == '':
            print(line)

        else:
            if len((line.split('\n')[0]).split(',')) != 5:
                print(line)


if __name__ == '__main__':
    pass
