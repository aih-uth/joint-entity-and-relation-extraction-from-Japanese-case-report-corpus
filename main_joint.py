import os, sys, random, logging
import pandas as pd
import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
import logging
from lib.util import load_data_cv, load_tokenizer, train_val_split_doc, make_idx, make_train_vecs, make_test_vecs, create_re_labels, save_csv, save_ner_result, save_re_result, cut_length
from lib.loop import train_val_loop, test_loop
import argparse
import json
from lib.util import eval_re_strict, eval_ner_strict


def main():
    logger.info("----------{0}-JOINTの実験を開始----------".format(args.bert_type))
    # 実験データ
    df = load_data_cv(args)
    df = cut_length(df, args.max_words)
    # 交差検証用の分割ファイル
    json_open = open('./data/index/train_test_index.json', 'r')
    train_test_indxe_dct = json.load(json_open)
    for fold in range(0, 5, 1):
        logger.info("----------{0}-fold----------".format(fold))
        # 事前に定義したindexを取得 0_fold_train_name
        train_index = train_test_indxe_dct["{0}_fold_train_name".format(fold)]
        test_index = train_test_indxe_dct["{0}_fold_test_name".format(fold)]
        # 取り出す
        X_train = df.query('name in @train_index')
        X_test = df.query('name in @test_index')

        logger.info("train: {0},  test: {1}".format(len(set(X_train["name"])), len(set(X_test["name"]))))


        # トーカナイザー
        bert_tokenizer = load_tokenizer(args)
        # 検証
        X_train, X_val = train_val_split_doc(X_train, args)

        logger.info("train: {0},  val: {1},  test: {2}".format(len(set(X_train["unique_no"])), len(set(X_val["unique_no"])), len(set(X_test["unique_no"]))))

        # ベクトル
        tag2idx, rel2idx = make_idx(pd.concat([X_train, X_val, X_test]), args)
        # 訓練ベクトルを作成
        train_vecs, ner_train_labels = make_train_vecs(X_train, bert_tokenizer, tag2idx)
        # 検証
        val_vecs, ner_val_labels = make_test_vecs(X_val, bert_tokenizer, tag2idx)
        # テスト
        test_vecs, ner_test_labels = make_test_vecs(X_test, bert_tokenizer, tag2idx)

        logger.info("train: {0}, val: {1}, test: {2}".format(len(train_vecs), len(val_vecs), len(test_vecs)))
        
        # 関係ラベルを作成
        re_train_gold_labels = create_re_labels(X_train, rel2idx)
        re_val_gold_labels = create_re_labels(X_val, rel2idx)
        re_test_gold_labels = create_re_labels(X_test, rel2idx)
        # 学習
        train_val_loop(train_vecs, ner_train_labels, re_train_gold_labels, 
                       X_val, val_vecs, ner_val_labels, re_val_gold_labels, 
                       tag2idx, rel2idx, fold, 
                       args, device, logger)
        # テスト
        res_df = test_loop(X_test, test_vecs, ner_test_labels, re_test_gold_labels, fold, tag2idx, rel2idx, args, device)
        # 評価
        strict_ner = eval_ner_strict(res_df)
        strict_re = eval_re_strict(res_df)
        strict_ignore_re = eval_re_strict(res_df, ignore_tags=True)
        # 保存
        save_ner_result(strict_ner, args, fold, tag2idx, "strict")
        save_re_result(strict_re, args, fold, rel2idx, "strict")
        save_re_result(strict_ignore_re, args, fold, rel2idx, "strict_ignore_tags")
        save_csv(res_df, args, fold)


if __name__ == '__main__':
    # 引数
    parser = argparse.ArgumentParser()
    parser.add_argument('--bert_path', type=str, default='/Users/shibata/Documents/BERT_v2/UTH_BERT_BASE_512_MC_BPE_WWM_V25000_352K')
    parser.add_argument('--bert_type', type=str, default='UTH')
    parser.add_argument('--data_path', type=str, default="./data/csv/CR_conll_format_arbitrary_UTH_for_experiment.csv")
    parser.add_argument('--max_epoch', type=int, default=250)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_words', type=int, default=510)
    parser.add_argument('--task', type=str, default="Joint")
    parser.add_argument('--idx_flag', type=str, default="F")
    parser.add_argument('--re_weight', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1478754)
    parser.add_argument('--re_val_eval', type=str, default="soft")

    
    args = parser.parse_args()

    if torch.cuda.is_available():
        print('use cuda device')
        device = torch.device("cuda")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
    else:
        print('use cpu')
        device = torch.device('cpu')
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # フォルダ作成
    for SAMPLE_DIR in ["./logs", "./models/{0}/{1}".format(args.task, args.bert_type), "./results/{0}/{1}/NER".format(args.task, args.bert_type), "./results/{0}/{1}/RE".format(args.task, args.bert_type)]:
        if not os.path.exists(SAMPLE_DIR):
            os.makedirs(SAMPLE_DIR)
    logger = logging.getLogger('LoggingTest')
    logger.setLevel(10)
    sh = logging.StreamHandler()
    logger.addHandler(sh)
    fh = logging.FileHandler('./logs/JOINT_{0}.log'.format(args.bert_type), "w")
    logger.addHandler(fh)
    formatter = logging.Formatter('%(asctime)s:%(lineno)d:%(levelname)s:%(message)s')
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    main()
