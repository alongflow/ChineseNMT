# -*- coding: utf-8 -*-
# @Time    : 2023/12/21 11:29
# @Author  : liuyl
# @File    : inference.py
# @Software: VSCode

import config
import numpy as np
import torch
from train import translate
from utils import english_tokenizer_load
from model import make_model


def one_sentence_translate(sent, beam_search=True):
    # 初始化模型
    model = make_model(config.src_vocab_size, config.tgt_vocab_size,
                       config.n_layers, config.d_model, config.d_ff,
                       config.n_heads, config.dropout)
    BOS = english_tokenizer_load().bos_id()  # 2
    EOS = english_tokenizer_load().eos_id()  # 3
    src_tokens = [[BOS] + english_tokenizer_load().EncodeAsIds(sent) + [EOS]]
    batch_input = torch.LongTensor(np.array(src_tokens)).to(config.device)
    translate(batch_input, model, use_beam=beam_search)


def translate_example():
    # 单句翻译示例
    sents = [
        "The pyramid of ocean life ...",
        "Now, to bring that home, I thought I'd invent a little game.",
        "Thank you. Thank you.",
        "In winter, there's a lot of sleeping going on; you enjoy your family life inside.",
        "Whatever that means to you.",
    ]
    answers = [
        "海洋生物的食物链……",
        "然后，为了能彻底了解这个问题，我想邀请大家做个小游戏",
        "谢谢，谢谢大家",
        "在冬天，睡眠时间很长。 人们在室内享受家庭生活。",
        "不管那对于你的意义是什么",
    ]
    for i, sent in enumerate(sents):
        print(f"<----------- {i+1}th sentence ----------->")
        print(f"原文：{sent}")
        print(f"答案：{answers[i]}")
        print("译文：", end='')
        one_sentence_translate(sent, beam_search=True)


if __name__ == '__main__':
    translate_example()
