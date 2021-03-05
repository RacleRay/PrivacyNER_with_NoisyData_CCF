import time
import json
from functools import wraps
from rich.console import Console

console = Console()


def is_chinese(uchar):
    """判断一个unicode是否是汉字"""
    if u'\u4e00' <= uchar <= u'\u9fa5':
        return True
    else:
        return False


def is_all_chinese(word_str):
    for c in word_str:
        if not '\u4e00' <= c <= '\u9fa5':
            return False
    return True


def json_load_by_line(file):
    data = []
    fp = open(file, 'r', encoding='utf-8')
    for d in fp.readlines():
        d = json.loads(d)
        data.append(d)
    fp.close()
    return data


def json_dump(data, file):
    fp = open(file, 'w+', encoding='utf-8')
    json.dump(data, fp, ensure_ascii=False, indent=4)
    fp.flush()
    time.sleep(2)
    fp.close()


def json_dump_by_line(data, file):
    fp = open(file, 'w+', encoding='utf-8')
    for d in data:
        string = json.dumps(d, ensure_ascii=False) + '\n'
        fp.write(string)
    fp.flush()
    time.sleep(2)
    fp.close()


def json_load(file):
    fp = open(file, 'r', encoding='utf-8')
    d = json.load(fp)
    fp.close()
    return d


def aggregate_by_key(data, key):
    result = {}
    for d in data:
        result.setdefault(d[key], []).append(d)
    return result


def show_runtime(func):
    @wraps(func)
    def _deco(*args, **kwargs):
        start = time.time()
        ret = func(*args, **kwargs)
        end = time.time()
        delta = end - start
        print("[Time] %s runed %.2f seconds" % (func.__name__, delta))
        return ret
    return _deco