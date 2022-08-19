import pandas as pd
import numpy as np
import json, MeCab, os
from pathlib import Path
from transformers import BertTokenizer
import jaconv
import unicodedata
import neologdn
import regex


def uth_bert_preprocess(text, nfkc=False, h2z=True):
    # Normalization Form Compatibility Composition (NFKC)
    if nfkc:
        text = unicodedata.normalize("NFKC", text)
    # full-width characterization
    text = regex.sub(r'(\d)([,])(\d+)', r'\1\3', text)
    text = text.replace(",", "、")
    text = text.replace("，", "、")
    if h2z:
        text = (jaconv.h2z(text, kana=True, digit=True, ascii=True))
    # remove full-width space
    text = text.replace("\u3000", "")
    return text


def mecab_wakati(sentence, hyper):
    mecab = MeCab.Tagger("-d {0} -u {1}".format(hyper.neologd_path, hyper.manbyo_path))
    tokenizer = BertTokenizer(Path("{0}".format(hyper.bert_path)) / "vocab.txt", do_lower_case=False, do_basic_tokenize=False)
    wakatis = []
    for ws in mecab.parse(sentence).split('\n'):
        if ws == "EOS":
            break
        else:
            wakatis.extend(tokenizer.tokenize(ws.split("\t")[0]))
    return wakatis


def make_vector(tokens, hyper):
    tokenizer = BertTokenizer(Path("{0}".format(hyper.bert_path)) / "vocab.txt", do_lower_case=False, do_basic_tokenize=False)
    return [tokenizer.convert_tokens_to_ids(["[CLS]"] + tokens[:hyper.max_text_len] + ["[SEP]"])]
