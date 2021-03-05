from transformers import BertTokenizer
from utils.tools import console


class MyTokenizer(BertTokenizer):
    def __init__(self,
                 vocab_file,
                 do_lower_case=True,
                 do_basic_tokenize=True,
                 never_split=['[unused1]', '[unused2]', '[unused3]'],
                 unk_token="[UNK]",
                 sep_token="[SEP]",
                 pad_token="[PAD]",
                 cls_token="[CLS]",
                 mask_token="[MASK]",
                 tokenize_chinese_chars=True,
                 **kwargs):
        super().__init__(vocab_file,
                         do_lower_case=do_lower_case,
                         do_basic_tokenize=do_basic_tokenize,
                         never_split=never_split,
                         unk_token=unk_token,
                         sep_token=sep_token,
                         pad_token=pad_token,
                         cls_token=cls_token,
                         mask_token=mask_token,
                         tokenize_chinese_chars=tokenize_chinese_chars,
                         **kwargs)

    def my_decode(self, tokens):
        "id转string。处理掉 ## 符号，BertTokenizer基于wordpiece方式产生。"
        tokens = self.convert_ids_to_tokens(tokens)  # BertTokenizer 内置
        tokens = " ".join(tokens).replace("##", "").strip()
        tokens = tokens.split(' ')
        return tokens

    def my_encode(self,
                  text,
                  text_pair=None,
                  add_special_tokens=True,
                  padding=False,
                  truncation=False,
                  max_length=None,
                  stride=0,
                  return_tensors=None,
                  **kwargs):
        "string转id"
        text = self._add_user_sign(text)
        return super().encode(text=text,
                              text_pair=text_pair,
                              add_special_tokens=add_special_tokens,
                              padding=padding,
                              truncation=truncation,
                              max_length=max_length,
                              stride=stride,
                              return_tensors=return_tensors,
                              **kwargs)

    @staticmethod
    def _add_user_sign(string):
        "根据自定义标记，处理字符串.  [unused] 都定义为不切分的token"
        new_string = ''
        index = 0
        chinese_index = 0
        while index < len(string):
            s = string[index]
            if MyTokenizer.is_chinese(s):
                chinese_index = index

            # 前一个字为中文的空格转为[unused1]，不进行 切分
            if s in {' ', '\xa0', '\u3000'} and chinese_index == index - 1:
                new_string += ' [unused1] '
            elif s == '“':
                new_string += ' [unused2] '
            elif s == '”':
                new_string += ' [unused3] '
            else:
                new_string += s

            index += 1
        return new_string

    @staticmethod
    def is_chinese(uchar):
        """判断一个unicode是否是汉字"""
        if u'\u4e00' <= uchar <= u'\u9fa5':
            return True
        else:
            return False

    def get_token_map(self, text):
        """
        return:
            decode2raw_map: 处理后字符index对应原text中的index 的映射
            raw2decode_map: 原text中字符的index对应处理后字符的index 的映射
        """

        # add_special_tokens：False
        # 【21，43，41】
        text_encoded = self.my_encode(text, add_special_tokens=False)

        # 【你，好，啊】  tokenize 后 处理掉 ## 等特殊符号 得到的 字符
        text_decoded = self.my_decode(text_encoded)

        if len(text_encoded) != len(text_decoded):
            console.log('get_token_map 方法中，[bold red] token 与 id 序列长度不一样')  # 因为处理 ## 的缘故

        # 【0，0，0】
        decode2raw_map = [0] * len(text_decoded)
        text_index = 0
        decode_index = 0

        ### 找到  text_decoded 中 每一个元素，  对应  text 中的起始位置
        # text_index 是 在text中遍历的index
        # decode_index 是 tokenize 处理得到的 字符中 一个一个找，可能出现  [unused1], [UNK], 多个字母的情况
        while True:
            sub_str = text[text_index]
            sub_str_encode = self.my_encode(sub_str, add_special_tokens=False)

            # 以下处理顺序需要仔细调整，不然map不会正确对齐
            if text_decoded[decode_index] in {'[unused1]', '[unused2]', '[unused3]'}:
                decode2raw_map[decode_index] = text_index
                decode_index += 1
                text_index += 1

            elif not sub_str_encode:
                text_index += 1

            elif text_decoded[decode_index] == '[UNK]':  # 为一个UNK 在 text 可能对应多个 字符
                unknow_start = text_index
                unknow_end = text_index
                while True:
                    if unknow_end == len(text) - 1:
                        decode2raw_map[decode_index] = unknow_start
                        text_index = unknow_end + 1  # 下一个，其实len(text)已经结束，函数最后会检查越界
                        decode_index += 1
                        break

                    unknow_end += 1

                    sub_str = text[unknow_start: unknow_end + 1]
                    sub_str_encode = self.my_encode(sub_str,
                                                    add_special_tokens=False)
                    # len(sub_str_encode) >= 2 已经找到 UNK 对应的完整 text
                    if len(sub_str_encode) >= 2 and sub_str_encode[0] == 100:  # '[UNK]'
                        decode2raw_map[decode_index] = unknow_start
                        text_index = unknow_end if text[unknow_end - 1] != ' ' else (unknow_end - 1)
                        decode_index += 1
                        break

            elif len(sub_str_encode) > 1:  # 一个text字符 对应 多个 token id
                sub_str_decode = self.my_decode(sub_str_encode)
                for _char in sub_str_decode:
                    if _char == text_decoded[decode_index]:
                        decode2raw_map[decode_index] = text_index
                        decode_index += 1
                    else:
                        console.log(f'{sub_str}: sub_str_encode 出现 多个, 继续匹配')
                text_index += 1

            elif len(text_decoded[decode_index]) == 1:  # 一对一
                decode2raw_map[decode_index] = text_index
                text_index += 1
                decode_index += 1

            elif len(text_decoded[decode_index]) > 1:
                # 多个字符(英文单词， 数字组合) 对应一个 token 或 日语多对多
                sub_len = 2
                while True:
                    sub_str_encode = self.my_encode(
                        text[text_index:text_index + sub_len],
                        add_special_tokens=False
                    )
                    sub_str_decode = self.my_decode(sub_str_encode)

                    if sub_str_decode[0] == text_decoded[decode_index] or "".join(sub_str_decode) == text_decoded[decode_index]:
                        break
                    elif len(sub_str_decode) > len(text_decoded[decode_index]):
                        console.log(f'token map 映射出现不齐: sub string decode结果为{sub_str_decode}， \
                                      text decode结果为{text_decoded[decode_index]}')

                    sub_len += 1

                decode2raw_map[decode_index] = text_index
                text_index += sub_len
                decode_index += 1

            else:
                console.log(f'{sub_str}：[bold red] token 检查所有情况不满足，跳过当前text字符')
                text_index += 1

            # 越界
            if text_index >= len(text) or decode_index >= len(text_decoded):
                break

        if decode2raw_map[-1] == 0 and len(decode2raw_map) > 1:
            console.log(f'{text}:[bold red] token 与 text 映射出现问题. decode2raw结果为{decode2raw_map}')

        raw2decode_map = [0] * len(text)
        for i, j in enumerate(decode2raw_map):
            if raw2decode_map[j] == 0:
                raw2decode_map[j] = i

        last_j = 0
        for i, j in enumerate(raw2decode_map):
            if j != last_j and j != 0:
                last_j = j
            raw2decode_map[i] = last_j

        # 特殊标记 占位
        decode2raw_map = [-1] + decode2raw_map + [len(text)]
        raw2decode_map = [i + 1 for i in raw2decode_map]

        return decode2raw_map, raw2decode_map
