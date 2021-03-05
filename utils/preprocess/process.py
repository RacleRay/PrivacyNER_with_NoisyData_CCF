import os
import csv
import json
import time
import pickle
from rich.progress import track
from utils.tools import console


def convert2json(text_dir, outfile, label_dir=None):
    text_file_list = os.listdir(text_dir)
    text_file_list.sort()
    data = []

    fp = open(outfile, 'w+', encoding='utf-8')
    for text_file in track(text_file_list, description=f"To json file {outfile}"):

        d = {'id': None, 'text': None, 'entities': []}
        with open(text_dir + text_file, 'r', encoding='utf-8') as f:
            d['text'] = f.read()
            d['id'] = int(text_file[:-4])

        if label_dir is not None:
            label_file = label_dir + "{}.csv".format(d['id'])
            reader = csv.reader(open(label_file, "r", encoding='utf-8'))
            for item in reader:
                if reader.line_num == 1: # 忽略表头
                    continue

                if item[4] != d['text'][int(item[2]):int(item[3]) + 1]:
                    console.log(f'标注错误 {d["id"]} 号文本：{d}, Item: {item}')

                if not d['id'] == int(item[0]):
                    console.log(f'标注文本不匹配 {d["id"]} 号文本, 标注为 {int(item[0])}')

                entity = {'privacy': item[4],
                          'category': item[1],
                          'pos_b': int(item[2]),
                          'pos_e': int(item[3])}
                d['entities'].append(entity)

        data.append(d)

    data.sort(key=lambda x: x['id'])
    for d in data:
        string = json.dumps(d, ensure_ascii=False) + '\n'
        fp.write(string)

    fp.flush()
    time.sleep(2)
    fp.close()
    console.rule("[bold red] save json file")
    console.log(f"{outfile}")


def convert2json_clue(infile_list, outfile):
    "都有标注，全部统一处理到一个 outfile"
    idx = 0
    fp_out = open(outfile, 'w+', encoding='utf-8')
    for filename in infile_list:
        fp = open(filename, 'r', encoding='utf-8')

        for line in track(fp.readlines(), description=f"Dealing {filename}"):
            d = json.loads(line)

            sample = {'id': idx, 'text': d['text'], 'entities': []}

            if 'label' in d:
                for category in d['label'].keys():
                    for privacy in d['label'][category].keys():
                        for entity in d['label'][category][privacy]:
                            sample['entities'].append({'privacy': privacy,
                                                        'category': category,
                                                        'pos_b': entity[0],
                                                        'pos_e': entity[1]})

                string = json.dumps(sample, ensure_ascii=False) + '\n'
                fp_out.write(string)
                idx += 1
            else:
                console.log(f"clue文本没有标注： {d['id']} 号文本, 标注为")

        fp.close()

    fp_out.flush()
    time.sleep(2)
    fp_out.close()

    console.rule("[bold red] save json file")
    console.log(f"{outfile}")


def process_wordvec(input_file, vec_save, vocab_save):
    "将 vocab 和 vec 分离，也可以不这样做，先统一格式也方便处理"
    raw_fp = open(input_file, 'r', encoding='utf-8')
    vocab_fp = open(vocab_save, 'wb+')
    vec_fp = open(vec_save, 'wb+')

    vocab_list = {'PAD': 0}
    vec_list = [[0.0] * 300]

    line = raw_fp.readline()

    console.log("Processing wordvec file ......")
    idx = 0
    while line != "":
        if idx == 0:
            line = raw_fp.readline()
            idx += 1
            continue

        data = line.split()

        vocab_list[data[0]] = idx
        vec_list.append([float(d) for d in data[1:]])

        line = raw_fp.readline()
        idx += 1

    raw_fp.close()

    pickle.dump(vocab_list, vocab_fp)
    pickle.dump(vec_list, vec_fp)

    vocab_fp.close()
    vec_fp.close()

    del vocab_list
    del vec_list

    console.rule("[bold red] save word vec file")
    console.log(f"{vocab_save}")
    console.log(f"{vec_save}")