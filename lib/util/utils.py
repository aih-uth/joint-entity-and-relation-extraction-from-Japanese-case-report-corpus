import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
import json, os
import collections
from transformers import BertModel, BertTokenizer
from sklearn.metrics import classification_report
from seqeval.metrics import classification_report as ner_eval


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def load_data_cv(args):
    df = pd.read_csv(args.data_path)
    list_df = []
    for ids in df["unique_no"].unique():
        tmp_df = df[df["unique_no"]==ids]
        if tmp_df.shape[0] <= args.max_words:
            list_df.append(tmp_df)
    return pd.concat(list_df)   


def load_tokenizer(args):
    return BertTokenizer(Path(args.bert_path) / "vocab.txt", do_lower_case=False, do_basic_tokenize=False)


def cut_length(df, num_words):
    list_df = []
    for ids in df["unique_no"].unique():
        tmp_df = df[df["unique_no"]==ids]
        if tmp_df.shape[0] <= num_words:
            list_df.append(tmp_df)
    return pd.concat(list_df)   


def train_val_split_doc(X_train):
    # 訓練データのid一覧
    ids = list(sorted(X_train["name"].unique()))
    # シャッフル
    random.seed(1478754)
    random.shuffle(ids)
    # 分割
    train_ids, val_ids = ids[:int(len(ids) * 0.8)], ids[int(len(ids) * 0.8):]
    # 確認
    assert len(set(ids[:int(len(ids) * 0.8)]) & set((ids[int(len(ids) * 0.8):])))==0, "trainとvalが重複してますよ"
    # 分割
    X_train, X_val = X_train[X_train["name"].isin(train_ids)].copy(), X_train[X_train["name"].isin(val_ids)].copy()
    return X_train, X_val


def make_idx(df, args):
    if args.idx_flag == "T":
        # タグ
        tag_vocab = list(sorted(set([x for x in df["IOB"]])))
        tag2idx = {x: i + 2 for i, x in enumerate(tag_vocab)}
        tag2idx["PAD"] = 0
        tag2idx["UNK"] = 1
        # 関係
        rel_vocab = list(sorted(set([y for x in df["rel_type"] for y in x.split(",")])))
        rel2idx = {x: i + 2 for i, x in enumerate(rel_vocab)}
        rel2idx["PAD"] = 0
        rel2idx["UNK"] = 1
    else:
        tag_vocab = list(sorted(set([x for x in df["IOB"]])))
        tag2idx = {x: i + 1 for i, x in enumerate(tag_vocab)}
        tag2idx["PAD"] = 0
        rel_vocab = list(sorted(set([y for x in df["rel_type"] for y in x.split(",")])))
        rel2idx = {}
        last_value = 1
        for _, x in enumerate(rel_vocab):
            if x == "None":
                pass
            else:
                rel2idx["R-" + x] = last_value
                rel2idx["L-" + x] = last_value + 1
                last_value +=2
        rel2idx["PAD"] = 0
        rel2idx["None"] = max(list(rel2idx.values())) + 1
    return tag2idx, rel2idx


def make_train_vecs(df, tokenizer, tag2idx):
    vecs1, vecs2 = [], []
    # テキスト
    for no in df["unique_no"].unique():
        # 取得
        tmp_df = df[df["unique_no"] == no]
        # 単語ベクトル
        ids = tokenizer.convert_tokens_to_ids(["[CLS]"] + list(tmp_df["word"]) + ["[SEP]"])
        # NER
        ner = [tag2idx[x] for x in list(tmp_df["IOB"])]
        # REL
        # ADD
        vecs1.append(ids)
        vecs2.append(ner)
    return vecs1, vecs2


def make_test_vecs(df, tokenizer, tag2idx):
    vecs1, vecs2 = [], []
    # テキスト
    for no in df["unique_no"].unique():
        # 取得
        tmp_df = df[df["unique_no"] == no]
        # 単語ベクトル
        ids = tokenizer.convert_tokens_to_ids(["[CLS]"] + list(tmp_df["word"]) + ["[SEP]"])
        # NER
        ner = [tag2idx[x] if x in tag2idx else tag2idx["UNK"] for x in list(tmp_df["IOB"])]
        # REL
        # ADD
        vecs1.append(ids)
        vecs2.append(ner)
    return vecs1, vecs2


def create_re_labels(df, rel2idx):
    gold_labels = []
    for ids in df["unique_no"].unique():
        tmp_df = df[df["unique_no"]==ids]
        # 固有表現、タグ、serialを取得する
        tokens, labels, indexs = list(tmp_df["word"]), list(tmp_df["IOB"]), list(tmp_df["serial"])
        seqs, tags, ids = [], [], []
        for i in range(0, len(tokens), 1):
            if labels[i].startswith("B-"):
                if i == len(labels) - 1:
                    seqs.append(tokens[i])
                    tags.append(labels[i])
                    ids.append([int(indexs[i])])
                else:
                    tmp1, tmp2, tmp3 = [tokens[i]], [labels[i]], [int(indexs[i])]
                    for j in range(i+1, len(tokens), 1):
                        if labels[j].startswith("I-"):
                            tmp1.append(tokens[j])
                            tmp2.append(labels[j])
                            tmp3.append(int(indexs[j]))
                            if j ==  len(labels) - 1:
                                seqs.append(" ".join(tmp1))
                                tags.append(" ".join(tmp2))
                                ids.append(tmp3)
                        else:
                            seqs.append(" ".join(tmp1))
                            tags.append(" ".join(tmp2))
                            ids.append(tmp3)
                            break  
        # 関係、tailの位置、headの位置を得る
        index2unnamed = {y: x for x, y in zip(tmp_df["serial"], tmp_df["index"])}
        gold_rels, gold_tails, gold_unnamed = [], [], []
        for index, types, tails in zip(tmp_df["serial"], tmp_df["rel_type"], tmp_df["rel_tail"]):
            if types == "None": continue
            for typex, tail in zip(types.split(","), tails.split(",")):
                gold_rels.append(typex)
                gold_tails.append(index2unnamed[int(tail)])
                gold_unnamed.append(index)

        rel_label = torch.full((tmp_df.shape[0], tmp_df.shape[0]), rel2idx["None"])
        for rel, tail, index in zip(gold_rels, gold_tails, gold_unnamed):
            head_index = index
            tail_index = tail
            # 関係、スタート位置、終了位置
            # 全indexを得る
            for i, idx in enumerate(ids):
                if idx[0] == head_index:
                    all_head_index, head_index = idx, i
                elif idx[0] == tail_index:
                    all_tail_index, tail_index = idx, i
            # 置換
            for h_i in all_head_index:
                for t_i in all_tail_index:
                    if h_i > t_i:
                        rel_label[t_i, h_i] = rel2idx["L-" + rel]
                    elif t_i > h_i:
                        rel_label[h_i, t_i] = rel2idx["R-" + rel]
                    else:
                        pass
        gold_labels.append(rel_label)
    return gold_labels


def create_output_df(tmp_df, rel_preds_np, pred_tags, rel2idx):
    idx2rel = {v: k for k, v in rel2idx.items()}
    rel_type_list, rel_tail_list = [[] for  i in range(0, rel_preds_np.shape[0], 1)], [[] for  i in range(0, rel_preds_np.shape[0], 1)]
    for i, preds in enumerate(rel_preds_np):
        for j, pred in enumerate(preds):
            pred_rel = idx2rel[pred]
            if pred_rel == "None" or pred_rel == "PAD": continue
            if idx2rel[pred][0] == "R":
                rel_tail_list[i].append(str(j))
                rel_type_list[i].append(pred_rel[2:])
            else:
                rel_tail_list[j].append(str(i))
                rel_type_list[j].append(pred_rel[2:])

    rel_type_list_for_df = [",".join(rel) if len(rel) != 0 else "None" for rel in rel_type_list]
    rel_tail_list_for_df = [",".join(rel) if len(rel) != 0 else "None"  for rel in rel_tail_list]
    tmp_df["pred_IOB"] = pred_tags
    tmp_df["pred_rel_type"] = rel_type_list_for_df
    tmp_df["pred_rel_tail"] = rel_tail_list_for_df
    return tmp_df
    
    
def evaluate_rel(res_df, rel2idx):
    tmp1, tmp2 = [], []
    for iob, typex, tail in zip(res_df["pred_IOB"], res_df["pred_rel_type"], res_df["pred_rel_tail"]):
        if iob[0] == "I":
            tmp1.append("None")
            tmp2.append("None")
        else:
            tmp1.append(typex)
            tmp2.append(tail)
    res_df["pred_rel_type"], res_df["pred_rel_tail"] = tmp1, tmp2
    
    list_df = []
    for ids in res_df["unique_no"].unique():
        tmp_df = res_df[res_df["unique_no"]==ids]
        # indexとtagの関係
        idx2tag = {x: y[0] for x, y in zip(tmp_df["serial"], tmp_df["pred_IOB"])}
        # rel_tailはindex準拠、pred_rel_tailはserial準拠になってるから変換
        unnamed2index = {int(x): int(y) for x, y in zip(tmp_df["serial"], tmp_df["index"])}
        updata_types, update_tails = [], []
        for types, tails in zip(tmp_df["pred_rel_type"], tmp_df["pred_rel_tail"]):
          if types == "None":
            updata_types.append(types)
            update_tails.append(tails)
          else:
            tmp_types, tmp_tails = [], []
            for typex, tail in zip(types.split(","), tails.split(",")):
                if idx2tag[int(tail)] != "B":
                    pass
                else:
                    if str(unnamed2index[int(tail)]) == str(-999):
                        pass
                    else:
                        tmp_types.append(typex)
                        tmp_tails.append(str(unnamed2index[int(tail)]))
            if len(tmp_types) != 0:
                updata_types.append(",".join(tmp_types))
                update_tails.append(",".join(tmp_tails))
            else:
                updata_types.append("None")
                update_tails.append("None")
        tmp_df["pred_rel_type"], tmp_df["pred_rel_tail"] = updata_types, update_tails
        list_df.append(tmp_df)
    res_df = pd.concat(list_df)
    
    re_gold_results, re_pred_results = [], []
    for i, ids in enumerate(res_df["unique_no"].unique()):
        tmp_df = res_df[res_df["unique_no"] == ids]
        # 予測結果
        for types, tails in zip(tmp_df["pred_rel_type"], tmp_df["pred_rel_tail"]):
            pred = ["None"] * tmp_df.shape[0]
            if types == "None":
                pass
            else:
                for typex, tail in zip(types.split(","), tails.split(",")):
                    pred[int(tail)] = typex
            re_pred_results.extend(pred)
        # 正解
        for types, tails in zip(tmp_df["rel_type"], tmp_df["rel_tail"]):
            label = ["None"] * tmp_df.shape[0]
            if types == "None":
                pass
            else:
                for typex, tail in zip(types.split(","), tails.split(",")):
                    label[int(tail)] = typex
            re_gold_results.extend(label)
            
    rel_labels = list(set([key.replace("R-", "").replace("L-", "") for key in rel2idx.keys() if key not in ["None", "PAD"]]))
    res = classification_report(re_gold_results, re_pred_results, output_dict=True, labels=rel_labels)
    return res, res_df


def simple_evaluate_re(re_val_gold_labels, re_preds, rel2idx):
    golds4eval, preds4eval = [], []
    for re_gold, re_logit in zip(re_val_gold_labels, re_preds):
        _, rel_preds = re_logit.max(dim=1)
        rel_preds_np = torch.triu(rel_preds, diagonal=1).detach().cpu().numpy()[0].tolist()
        rel_gold_np = torch.triu(re_gold, diagonal=1).detach().cpu().numpy().tolist()
        for gold, pred in zip(rel_gold_np, rel_preds_np):
            golds4eval.extend(gold)
            preds4eval.extend(pred)
    re_labels = [v for k, v in rel2idx.items() if k not in {"PAD": "PAD", "None": "None"}]
    return classification_report(golds4eval, preds4eval, output_dict=True, labels=re_labels)


def evaluate_ner(model, labels, preds, tag2idx):
    idx2tag = {v: k for k, v in tag2idx.items()}
    # NERの予測結果のDECODE
    pred_tags = []
    for pred in preds:
        pred_tag = model.module.crf.decode(pred)
        pred_tags.append([idx2tag[tag] for tag in pred_tag[0]])
    labels = [[idx2tag[y] for y in x] for x in labels]
    res = ner_eval(labels, pred_tags, output_dict=True)
    return res
    
    
def create_df(df, model, ner_preds, re_preds, tag2idx, rel2idx):
    # DFを作る
    list_df = []
    for batch, idx in enumerate(df["unique_no"].unique()):
        tmp_df = df[df["unique_no"] == idx]
        ner_logit = ner_preds[batch]
        re_logit = re_preds[batch]
        # NERの予測結果をデコード
        idx2tag = {v: k for k, v in tag2idx.items()}
        pred_tags = model.module.crf.decode(ner_logit)[0]
        pred_tags = [idx2tag[tag] for tag in pred_tags]
        # REの予測結果のデコード
        # 各要素の最大値を各バッチから計算して取得 (rel_scoresは最大値)
        rel_scores, rel_preds = re_logit.max(dim=1)
        # 対角 (左下)は落とす
        rel_preds_np = torch.triu(rel_preds, diagonal=1).detach().cpu().numpy()[0]
        # DFを作成
        tmp_df = create_output_df(tmp_df, rel_preds_np, pred_tags, rel2idx)
        list_df.append(tmp_df)  
    return pd.concat(list_df)


def save_re_result(i_th_res, args, fold, rel2idx, typex):
    with open('./results/{0}/{1}/RE/RE_{3}_RESULT_{2}.json'.format(args.task, args.bert_type, fold, typex), 'w') as f:
        json.dump(i_th_res, f, indent=4, cls=NpEncoder)
    with open('./results/{0}/{1}/RE/RE_rel2idx_{2}.json'.format(args.task, args.bert_type, fold), 'w') as f:
        json.dump(rel2idx, f, indent=4, cls=NpEncoder)


def save_ner_result(i_th_res, args, fold, tag2idx, typex):
    with open('./results/{0}/{1}/NER/NER_{3}_RESULT_{2}.json'.format(args.task, args.bert_type, fold, typex), 'w') as f:
        json.dump(i_th_res, f, indent=4, cls=NpEncoder)
    with open('./results/{0}/{1}/NER/NER_tag2idx_{2}.json'.format(args.task, args.bert_type, fold), 'w') as f:
        json.dump(tag2idx, f, indent=4, cls=NpEncoder)


def save_csv(fold_res_df, args, fold):
    fold_res_df.to_csv('./results/{0}/{1}/PRED_{2}.csv'.format(args.task, args.bert_type, fold), index=False)


def result2df(X_test, ner_preds, re_preds, rel2idx, model, tag2idx):
    list_df = []
    idx2tag = {v: k for k, v in tag2idx.items()}
    for batch, idx in enumerate(X_test["unique_no"].unique()):
        # if idx != "180_3": continue
        # DataFrame
        tmp_df = X_test[X_test["unique_no"]==idx]
        # 予測結果の変換
        _, rel_logit = re_preds[batch].max(dim=1)
        rel_logit = torch.triu(rel_logit, diagonal=1).detach().cpu().numpy()[0].tolist()
        ner_logit = model.module.crf.decode(ner_preds[batch])[0]
        # NERを代入
        tmp_df["pred_IOB"] = [idx2tag[tag] for tag in ner_logit]
        # 固有表現、タグ、serialを取得する
        tokens, labels, indexs = list(tmp_df["word"]), list(tmp_df["pred_IOB"]), list(tmp_df["serial"])
        seqs, tags, ids = [], [], []
        for i in range(0, len(tokens), 1):
            if labels[i].startswith("B-"):
                if i == len(labels) - 1:
                    seqs.append(tokens[i])
                    tags.append(labels[i])
                    ids.append([int(indexs[i])])
                else:
                    tmp1, tmp2, tmp3 = [tokens[i]], [labels[i]], [int(indexs[i])]
                    for j in range(i+1, len(tokens), 1):
                        if labels[j].startswith("I-"):
                            tmp1.append(tokens[j])
                            tmp2.append(labels[j])
                            tmp3.append(int(indexs[j]))
                            if j ==  len(labels) - 1:
                                seqs.append(" ".join(tmp1))
                                tags.append(" ".join(tmp2))
                                ids.append(tmp3)
                        else:
                            seqs.append(" ".join(tmp1))
                            tags.append(" ".join(tmp2))
                            ids.append(tmp3)
                            break  

        # Bタグのindex
        begin_index = [idx[0] for idx in ids]
        # Iタグのindex
        inside_index = [idxx for idx in ids for idxx in idx[1: ]]
        # 予測結果を集計
        idx2rel = {v: k for k, v in rel2idx.items()}
        decode_rels = []

        for i in range(0, tmp_df.shape[0], 1):
            # 各行の予測結果
            i_th_rel_logit = rel_logit[i]#.detach().cpu().numpy()
            i_th_ner_logit = ner_logit[i]#.detach().cpu().numpy()
            if i in inside_index: continue
            # 各行の各要素を確認
            for j in range(0, tmp_df.shape[0], 1):
                # 同様
                if j in inside_index: continue
                # 関係がある場合の処理 (右下は0、つまりPADになることに注意)
                if i_th_rel_logit[j] != rel2idx["None"] and i_th_rel_logit[j] != rel2idx["PAD"]:

                    rel_pred = idx2rel[i_th_rel_logit[j]]

                    if rel_pred[0] == "R":
                        # この値 (headとtail)はserialであり、通し番号である (正解ラベルはindex列)
                        decode_rels.append({"head": i, "tail": j, "rel": rel_pred[2:]})
                    else:
                        decode_rels.append({"head": j, "tail": i, "rel": rel_pred[2:]})
        # データフレームへ代入
        types, tails, indexs = [], [], []
        # 変換辞書
        unnamed2index = {int(x): int(y) for x, y in zip(tmp_df["serial"], tmp_df["index"])}
        for i in range(0, tmp_df.shape[0], 1):
            tmp1, tmp2, tmp3 = [], [], []
            for res in decode_rels:
                if res["head"] == i:
                    tmp1.append(str(unnamed2index[res["tail"]]))
                    tmp2.append(str(res["rel"]))
                    tmp3.append(str(res["tail"]))
            if len(tmp1) != 0:
                tails.append(",".join(tmp1))
                types.append(",".join(tmp2))
                indexs.append(",".join(tmp3))
            else:
                tails.append("None")
                types.append("None")
                indexs.append("None")
        tmp_df["pred_rel_tail"] = tails
        tmp_df["pred_rel_type"] = types
        tmp_df["pred_rel_unnamed"] = indexs
        list_df.append(tmp_df)
    return pd.concat(list_df)


def evaluate_rel_v2(res_df, rel2idx):
    re_gold_results, re_pred_results = [], []
    for i, ids in enumerate(res_df["unique_no"].unique()):
        tmp_df = res_df[res_df["unique_no"] == ids]
        # 予測結果
        for types, tails in zip(tmp_df["pred_rel_type"], tmp_df["pred_rel_tail"]):
            pred = ["None"] * tmp_df.shape[0]
            if types == "None":
                pass
            else:
                for typex, tail in zip(types.split(","), tails.split(",")):
                    # サブワードにかかる場合は落とす
                    if tail == str(-999):
                        pass
                    else:
                        pred[int(tail)] = typex
            re_pred_results.extend(pred)
        # 正解
        for types, tails in zip(tmp_df["rel_type"], tmp_df["rel_tail"]):
            label = ["None"] * tmp_df.shape[0]
            if types == "None":
                pass
            else:
                for typex, tail in zip(types.split(","), tails.split(",")):
                    label[int(tail)] = typex
            re_gold_results.extend(label)


    rel_labels = list(set([key.replace("R-", "").replace("L-", "") for key in rel2idx.keys() if key not in ["None", "PAD"]]))
    res = classification_report(re_gold_results, re_pred_results, output_dict=True, labels=rel_labels)
    return res


def result2df_for_ner(X_test, ner_preds_decode):
    list_df = []
    for i, idx in enumerate(X_test["unique_no"].unique()):
        # DataFrame
        tmp_df = X_test[X_test["unique_no"]==idx]
        # NERを代入
        tmp_df["pred_IOB"] = ner_preds_decode[i]
        list_df.append(tmp_df)
    return pd.concat(list_df)


def result2df_for_re(X_test, re_preds, rel2idx, tag2idx):
    list_df = []
    idx2tag = {v: k for k, v in tag2idx.items()}
    for batch, idx in enumerate(X_test["unique_no"].unique()):
        #print(idx)
        # DataFrame
        tmp_df = X_test[X_test["unique_no"]==idx]
        # 予測結果の変換
        _, rel_logit = re_preds[batch].max(dim=1)
        rel_logit = torch.triu(rel_logit, diagonal=1).detach().cpu().numpy()[0].tolist()
        # 固有表現、タグ、serialを取得する
        tokens, labels, indexs = list(tmp_df["word"]), list(tmp_df["pred_IOB"]), list(tmp_df["serial"])
        seqs, tags, ids = [], [], []
        for i in range(0, len(tokens), 1):
            if labels[i].startswith("B-"):
                if i == len(labels) - 1:
                    seqs.append(tokens[i])
                    tags.append(labels[i])
                    ids.append([int(indexs[i])])
                else:
                    tmp1, tmp2, tmp3 = [tokens[i]], [labels[i]], [int(indexs[i])]
                    for j in range(i+1, len(tokens), 1):
                        if labels[j].startswith("I-"):
                            tmp1.append(tokens[j])
                            tmp2.append(labels[j])
                            tmp3.append(int(indexs[j]))
                            if j ==  len(labels) - 1:
                                seqs.append(" ".join(tmp1))
                                tags.append(" ".join(tmp2))
                                ids.append(tmp3)
                        else:
                            seqs.append(" ".join(tmp1))
                            tags.append(" ".join(tmp2))
                            ids.append(tmp3)
                            break  
        # Bタグのindex
        begin_index = [idx[0] for idx in ids]
        # Iタグのindex
        inside_index = [idxx for idx in ids for idxx in idx[1: ]]
        # 予測結果を集計
        idx2rel = {v: k for k, v in rel2idx.items()}
        decode_rels = []
        for i in range(0, tmp_df.shape[0], 1):
            # 各行の予測結果
            i_th_rel_logit = rel_logit[i]#.detach().cpu().numpy()
            if i in inside_index: continue
            # 各行の各要素を確認
            for j in range(0, tmp_df.shape[0], 1):
                # 同様
                if j in inside_index: continue
                if i_th_rel_logit[j] != rel2idx["None"] and i_th_rel_logit[j] != rel2idx["PAD"]:
                    rel_pred = idx2rel[i_th_rel_logit[j]]
                    if rel_pred[0] == "R":
                        decode_rels.append({"head": i, "tail": j, "rel": rel_pred[2:]})
                    else:
                        decode_rels.append({"head": j, "tail": i, "rel": rel_pred[2:]})
        # データフレームへ代入
        types, tails, indexs = [], [], []
        # 変換辞書
        unnamed2index = {int(x): int(y) for x, y in zip(tmp_df["serial"], tmp_df["index"])}
        #print(decode_rels)
        #print(unnamed2index)
        for i in range(0, tmp_df.shape[0], 1):
            tmp1, tmp2, tmp3 = [], [], []
            for res in decode_rels:
                if res["head"] == i:
                    tmp1.append(str(unnamed2index[res["tail"]]))
                    tmp2.append(str(res["rel"]))
                    tmp3.append(str(res["tail"]))
            if len(tmp1) != 0:
                tails.append(",".join(tmp1))
                types.append(",".join(tmp2))
                indexs.append(",".join(tmp3))
            else:
                tails.append("None")
                types.append("None")
                indexs.append("None")
        tmp_df["pred_rel_tail"] = tails
        tmp_df["pred_rel_type"] = types
        tmp_df["pred_rel_unnamed"] = indexs
        list_df.append(tmp_df)
    return pd.concat(list_df)


def make_train_vecs_pipeline(df, tokenizer, tag2idx):
    vecs1, vecs2 = [], []
    # テキスト
    for no in df["unique_no"].unique():
        # 取得
        tmp_df = df[df["unique_no"] == no]
        # 単語ベクトル
        ids = tokenizer.convert_tokens_to_ids(["[CLS]"] + list(tmp_df["word"]) + ["[SEP]"])
        # NER
        ner = [tag2idx[x] for x in list(tmp_df["IOB"])]
        # REL
        # ADD
        vecs1.append(ids)
        vecs2.append(ner)
    return vecs1, vecs2


def make_test_vecs_pipeline(df, tokenizer, tag2idx, exp_type):
    vecs1, vecs2 = [], []
    # テキスト
    for no in df["unique_no"].unique():
        # 取得
        tmp_df = df[df["unique_no"] == no]
        # 単語ベクトル
        ids = tokenizer.convert_tokens_to_ids(["[CLS]"] + list(tmp_df["word"]) + ["[SEP]"])
        # NER
        if exp_type == "NER":
            ner = [tag2idx[x] if x in tag2idx else tag2idx["UNK"] for x in list(tmp_df["IOB"])]
        else:
            ner = [tag2idx[x] if x in tag2idx else tag2idx["UNK"] for x in list(tmp_df["pred_IOB"])]
        # REL
        # ADD
        vecs1.append(ids)
        vecs2.append(ner)
    return vecs1, vecs2


def decode_ner_pipeline(model, labels, preds, tag2idx, vecs):
    idx2tag = {v: k for k, v in tag2idx.items()}
    pred_tags = []
    index = 0
    for predx in preds:
        predx = model.module.crf.decode(predx)
        for pred in predx:
            pred_tags.append([idx2tag[pred] for pred in pred[:len(vecs[index])-2]])
            index += 1
    return pred_tags


def save_csv_pipeline(fold_res_df, args, fold, name):
    fold_res_df.to_csv('./results/{0}/{1}/{3}/{2}.csv'.format(args.task, args.bert_type, fold, name), index=False)


def load_data_for_re(args):
    list_df = []
    for i in range(0, 5, 1):
        list_df.append(pd.read_csv("./results/{0}/{1}/NER/{2}.csv".format(args.task, args.bert_type, i)))
    return pd.concat(list_df) 