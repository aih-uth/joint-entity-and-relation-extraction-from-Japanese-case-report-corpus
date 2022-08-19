import os, sys, random, logging
import pandas as pd
import numpy as np
import torch
import torch.utils.data
import logging
from lib.util import load_data_cv, load_tokenizer, train_val_split_doc, make_idx, make_train_vecs_pipeline, make_test_vecs_pipeline, save_csv_pipeline, save_ner_result
from lib.loop import train_val_loop_ner, test_loop_ner
import argparse
import json
from lib.util import eval_ner_strict, eval_ner_soft


def main():
    logger.info("----------{0}-NERの実験を開始----------".format(args.bert_type))
    # 実験データ
    df = load_data_cv(args)
    # 交差検証用の分割ファイル
    json_open = open('./data/index/train_test_index.json', 'r')
    train_test_indxe_dct = json.load(json_open)
    for fold in range(0, 5, 1):
        logger.info("----------{0}-fold----------".format(fold))
        # 事前に定義したindexを取得
        train_index = train_test_indxe_dct["{0}_fold_train_name".format(fold)]
        test_index = train_test_indxe_dct["{0}_fold_test_name".format(fold)]
        # 取り出す
        X_train = df.query('name in @train_index')
        X_test = df.query('name in @test_index')
        # トーカナイザー
        bert_tokenizer = load_tokenizer(args)
        # 検証
        X_train, X_val = train_val_split_doc(X_train)
        # ベクトル
        tag2idx, _ = make_idx(pd.concat([X_train, X_val, X_test]), args)
        # 訓練ベクトルを作成
        train_vecs, ner_train_labels = make_train_vecs_pipeline(X_train, bert_tokenizer, tag2idx)
        # 検証
        val_vecs, ner_val_labels = make_test_vecs_pipeline(X_val, bert_tokenizer, tag2idx, "NER")
        # テスト
        test_vecs, ner_test_labels = make_test_vecs_pipeline(X_test, bert_tokenizer, tag2idx, "NER")
        logger.info("train: {0}, val: {1}, test: {2}".format(len(train_vecs), len(val_vecs), len(test_vecs)))
        # 学習
        train_val_loop_ner(train_vecs, ner_train_labels,
                           X_val, val_vecs, ner_val_labels, 
                           tag2idx, fold, args, device, logger)
        # テスト
        res_df = test_loop_ner(X_test, test_vecs, ner_test_labels, fold, tag2idx, args, device)
        # 評価
        sreict_ner = eval_ner_strict(res_df)
        soft_ner = eval_ner_soft(res_df)
        # 保存
        save_ner_result(sreict_ner, args, fold, tag2idx, "strict")
        save_ner_result(soft_ner, args, fold, tag2idx, "soft")
        save_csv_pipeline(res_df, args, fold, "NER")


if __name__ == '__main__':
    if torch.cuda.is_available():
        print('use cuda device')
        seed=1478754
        device = torch.device("cuda")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    else:
        print('use cpu')
        device = torch.device('cpu')
        torch.manual_seed(999)
        np.random.seed(999)
    # 引数
    parser = argparse.ArgumentParser()
    parser.add_argument('--bert_path', type=str, default='/Users/shibata/Documents/BERT_v2/UTH_BERT_BASE_512_MC_BPE_WWM_V25000_352K')
    parser.add_argument('--bert_type', type=str, default='UTH')
    parser.add_argument('--data_path', type=str, default="./data/csv/CR_conll_format_arbitrary_UTH_for_experiment.csv")
    parser.add_argument('--max_epoch', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_words', type=int, default=10)
    parser.add_argument('--task', type=str, default="Pipeline")
    parser.add_argument('--idx_flag', type=str, default="F")
    args = parser.parse_args()
    # フォルダ作成
    for SAMPLE_DIR in ["./models/{0}/{1}/NER".format(args.task, args.bert_type), "./results/{0}/{1}/NER".format(args.task, args.bert_type)]:
        if not os.path.exists(SAMPLE_DIR):
            os.makedirs(SAMPLE_DIR)
    logger = logging.getLogger('LoggingTest')
    logger.setLevel(10)
    sh = logging.StreamHandler()
    logger.addHandler(sh)
    fh = logging.FileHandler('./logs/Pipeline_NER_{0}.log'.format(args.bert_type), "w")
    logger.addHandler(fh)
    formatter = logging.Formatter('%(asctime)s:%(lineno)d:%(levelname)s:%(message)s')
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    main()