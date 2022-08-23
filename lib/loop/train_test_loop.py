import random, json
import copy
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from lib.models import BERT_TF, compute_re_loss, compute_ner_loss, compute_loss
from lib.util import create_df, create_output_df, evaluate_rel, evaluate_ner, simple_evaluate_re, result2df, evaluate_rel_v2
from tqdm import tqdm
import transformers
from lib.util import eval_ner_strict, decode_ner_pipeline, result2df_for_ner, get_weight


def batch_processing(model, sentence, tag, batch_re, device, train_is, hyper):
    # 1バッチごとに処理
    rel_logits, ner_logits = [], []
    for batch in range(0, hyper.batch_size, 1):
        ner_logit, rel_logit = model(sentence[batch].unsqueeze(0), 
                                     tag[batch].unsqueeze(0), 
                                     train_is, 
                                     device)
        ner_logits.append(ner_logit)
        rel_logits.append(rel_logit)
    # 損失の計算
    total_loss, ner_loss, rel_loss = compute_loss(model, ner_logits, tag, rel_logits, batch_re, device)
    # 誤差伝搬
    # total_loss.backward()
    return ner_logits, rel_logits, total_loss, ner_loss, rel_loss


def train_val_loop(train_vecs, ner_train_labels, re_train_gold_labels, 
                   X_val, val_vecs, ner_val_labels, re_val_gold_labels, 
                   tag2idx, rel2idx, fold, 
                   args, device, logger):
    # 訓練
    best_val_F =  -1e5
    # モデルを定義
    model = BERT_TF(args, tag2idx, rel2idx, device).to(device)
    # 最適化関数
    optimizer = optim.AdamW([
                            {'params': model.bert_model.parameters(), 'lr': 3e-5, 'weight_decay': 0.01},
                            {'params': model.linear.parameters(), 'lr': 1e-3, 'weight_decay': 0.01},
                            {'params': model.crf.parameters(), 'lr': 1e-3, 'weight_decay': 0.01},
                            {'params': model.label_embedding.parameters(), 'lr': 1e-3, 'weight_decay': 0.01},
                            {'params': model.rel_classifier.parameters(), 'lr': 1e-3, 'weight_decay': 0.01}],
                            eps=1e-03
                            )
    # Total number of training steps is [number of batches] x [number of epochs]. 
    warmup_steps = int(args.max_epoch * len(train_vecs) * 0.1 / args.batch_size)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, 
                                                             num_warmup_steps=warmup_steps, 
                                                             num_training_steps=(len(train_vecs)/args.batch_size)*args.max_epoch)
    # 念の為; BERTの全レイヤーの勾配を更新
    for _, param in model.named_parameters():
        param.requires_grad = True
    model = torch.nn.DataParallel(model)

    weights = get_weight(rel2idx, device, args)

    loss_dct = {"epoch": [], "train_NER_loss": [], "train_RE_loss": [], "val_NER_loss": [], "val_RE_loss": []}
    
    # Loop
    for epoch in range(args.max_epoch):
        # epochごとに訓練データをシャッフルする
        train_indice = list(range(len(train_vecs)))
        random.shuffle(train_indice)
        # モデルを学習モードに
        model.train()
        # 損失　
        ner_running_loss, re_running_loss = 0, 0
        # バッチごとの処理
        pbar_train = tqdm(range(0, len(train_vecs), args.batch_size))
        for ofs in pbar_train:
            pbar_train.set_description('モデルを学習中!')
            # 勾配を初期化
            optimizer.zero_grad()
            #　バッチ分だけ取り出す (1)
            begin_index = ofs
            end_index = min(len(train_vecs), ofs + args.batch_size)
            batch_indice = train_indice[begin_index:end_index]
            #　バッチ分だけ取り出す (2)
            batch_X = copy.deepcopy([train_vecs[inx] for inx in batch_indice])
            batch_ner = copy.deepcopy([ner_train_labels[inx] for inx in batch_indice])
            batch_re = copy.deepcopy([re_train_gold_labels[inx] for inx in batch_indice])
            # ベクトル
            sentence = [torch.tensor(x) for x in batch_X]
            tag = [torch.tensor(x) for x in batch_ner]
            # PADDING
            sentence = pad_sequence(sentence, padding_value=0, batch_first=True).to(device)
            tag = pad_sequence(tag, padding_value=0, batch_first=True).to(device)
            # 予測
            ner_logits, rel_logits = model(sentence, tag, True, device)
            # 損失の計算
            re_loss = compute_re_loss(rel_logits, batch_re, device, weights)
            ner_loss = compute_ner_loss(model, ner_logits, tag)
            # 誤差伝搬
            (re_loss + ner_loss).backward()
            # 勾配を更新
            optimizer.step()
            scheduler.step()
            # 合計の損失
            ner_running_loss += ner_loss.item()
            re_running_loss += re_loss.item()
        logger.info("訓練")
        logger.info("{0}エポック目のNERの損失値: {1}".format(epoch, ner_running_loss))
        logger.info("{0}エポック目のREの損失値: {1}\n".format(epoch, re_running_loss))
        # 検証
        ner_preds, re_preds = [], []
        with torch.inference_mode():
            # 勾配を更新しない
            model.eval()
            # 
            val_ner_running_loss, val_re_running_loss = 0, 0
            # バッチごとの処理
            pbar_val = tqdm(range(0, len(val_vecs), args.batch_size))
            for ofs in pbar_val:
                pbar_train.set_description('モデルを検証中!')
                #　バッチ分だけ取り出す (1)
                begin_index = ofs
                end_index = min(len(val_vecs), ofs + args.batch_size)
                #　バッチ分だけ取り出す (2)
                batch_X = copy.deepcopy(val_vecs[begin_index: end_index])
                batch_ner = copy.deepcopy(ner_val_labels[begin_index: end_index])
                batch_re = copy.deepcopy(re_val_gold_labels[begin_index: end_index])
                # ベクトル
                sentence = [torch.tensor(x) for x in batch_X]
                tag = [torch.tensor(x) for x in batch_ner]
                # PADDING
                sentence = pad_sequence(sentence, padding_value=0, batch_first=True).to(device)
                tag = pad_sequence(tag, padding_value=0, batch_first=True).to(device)
                # 予測
                ner_logits, rel_logits = model(sentence, None, False, device)
                ner_preds.append(ner_logits)
                re_preds.append(rel_logits)
                # 損失の計算
                re_loss = compute_re_loss(rel_logits, batch_re, device, weights)
                ner_loss = compute_ner_loss(model, ner_logits, tag)
                # 合計の損失
                val_ner_running_loss += ner_loss.item()
                val_re_running_loss += re_loss.item()
        # 評価（NER）
        ner_preds_decode = decode_ner_pipeline(model, ner_val_labels, ner_preds, tag2idx, val_vecs)
        res_df = result2df_for_ner(X_val, ner_preds_decode)
        sreict_ner = eval_ner_strict(res_df)
        ner_val_F = sreict_ner["micro avg"]["f1"] 
        # 評価（RE: 簡易）
        rel_res = simple_evaluate_re(re_val_gold_labels, re_preds, rel2idx)
        rel_val_f1 = rel_res["micro avg"]["f1-score"]
        # 平均値
        val_F = 0.5 * ner_val_F + 0.5 * rel_val_f1
        # 保存
        logger.info("検証")
        logger.info("{0}エポック目のNERの損失値: {1}".format(epoch, val_ner_running_loss))
        logger.info("{0}エポック目のREの損失値: {1}".format(epoch, val_re_running_loss))

        logger.info("{0}エポック目のNERのMicro Avg: {1}".format(epoch, ner_val_F))
        logger.info("{0}エポック目のREのMicro Avg: {1}".format(epoch, rel_val_f1))
        logger.info("{0}エポック目の平均F1: {1}\n".format(epoch, val_F))

        loss_dct["epoch"].append(epoch)
        loss_dct["train_NER_loss"].append(ner_running_loss)
        loss_dct["train_RE_loss"].append(re_running_loss)
        loss_dct["val_NER_loss"].append(val_ner_running_loss)
        loss_dct["val_RE_loss"].append(val_re_running_loss)

        # Early Stopping
        if val_F > best_val_F:
            logger.info("{0}エポック目で更新\n".format(epoch))
            torch.save(model.module.state_dict(), 
                       './models/{0}/{1}/model_{2}.pt'.format(args.task, args.bert_type, fold))
            best_val_F = val_F
        else:
            logger.info("{0}エポック目は現状維持\n".format(epoch))

    with open('./results/{0}/{1}/loss_{2}.json'.format(args.task, args.bert_type, fold), 'w') as f:
        json.dump(loss_dct, f, indent=4)

            
def test_loop(X_test, test_vecs, ner_test_labels, re_test_gold_labels, 
              fold, tag2idx, rel2idx, args, device):
    # テスト
    with torch.inference_mode():
        # モデルを定義
        model = BERT_TF(args, tag2idx, rel2idx, device).to(device)
        #検証データでの損失が最良となったベストモデルを読み込む
        model.load_state_dict(torch.load('./models/{0}/{1}/model_{2}.pt'.format(args.task, args.bert_type, fold)))
        model = torch.nn.DataParallel(model)
        # 勾配を更新しない
        model.eval()
        # 結果
        ner_preds, re_preds = [], []
        # バッチごとの処理
        for ofs in range(0, len(test_vecs), args.batch_size):
            #　バッチ分だけ取り出す (1)
            begin_index = ofs
            end_index = min(len(test_vecs), ofs + args.batch_size)
            #　バッチ分だけ取り出す (2)
            batch_X = test_vecs[begin_index: end_index]
            batch_ner = ner_test_labels[begin_index: end_index]
            batch_re = re_test_gold_labels[begin_index: end_index]
            # ベクトル
            sentence = [torch.tensor(x) for x in batch_X]
            tag = [torch.tensor(x) for x in batch_ner]
            # PADDING
            sentence = pad_sequence(sentence, padding_value=0, batch_first=True).to(device)
            tag = pad_sequence(tag, padding_value=0, batch_first=True).to(device)
            # 予測
            ner_logits, rel_logits = model(sentence, None, False, device)
            ner_preds.append(ner_logits)
            re_preds.append(rel_logits)
    # 新しい評価方法
    res_df = result2df(X_test, ner_preds, re_preds, rel2idx, model, tag2idx)
    return res_df