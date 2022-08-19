import random, json
import copy
import torch
import torch.utils.data
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from lib.models import BERT_CRF, compute_ner_loss
from lib.util import decode_ner_pipeline, result2df_for_ner
from tqdm import tqdm
import transformers
from lib.util import eval_ner_strict, eval_ner_soft


def train_val_loop_ner(train_vecs, ner_train_labels,
                       X_val, val_vecs, ner_val_labels, 
                       tag2idx, fold, args, device, logger):
    # 以下で訓練るーぷ
    best_val_F =  -1e5
    # モデルを定義
    model = BERT_CRF(args, tag2idx).to(device)
    # 最適化関数
    optimizer = optim.AdamW([
                            {'params': model.bert_model.parameters(), 'lr': 3e-5, 'weight_decay': 0.01},
                            {'params': model.linear.parameters(), 'lr': 1e-3, 'weight_decay': 0.01},
                            {'params': model.crf.parameters(), 'lr': 1e-3, 'weight_decay': 0.01}],
                            )
    # Total number of training steps is [number of batches] x [number of epochs]. 
    # (Note that this is not the same as the number of training samples).
    # num_warmup_steps = 0 -> Default value in run_glue.py
    # https://www.pdfprof.com/PDF_Image.php?idt=76193&t=28 -> warmup_stepsは学習率を上げていくSTEP? グラフで言う右上がりの箇所
    warmup_steps = int(args.max_epoch * len(train_vecs) * 0.1 / args.batch_size)
    # warmup_steps = 0
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, 
                                                             num_warmup_steps=warmup_steps, 
                                                             num_training_steps=(len(train_vecs)/args.batch_size)*args.max_epoch)
    # 念の為; BERTの全レイヤーの勾配を更新
    for _, param in model.named_parameters():
        param.requires_grad = True
    model = torch.nn.DataParallel(model)

    loss_dct = {"epoch": [], "train_NER_loss": [], "val_NER_loss": []}
    
    # Loop
    for epoch in range(args.max_epoch):
        # epochごとに訓練データをシャッフルする
        train_indice = list(range(len(train_vecs)))
        random.shuffle(train_indice)
        # モデルを学習モードに
        model.train()
        # 損失　
        ner_running_loss = 0
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
            # ベクトル
            sentence = [torch.tensor(x) for x in batch_X]
            tag = [torch.tensor(x) for x in batch_ner]
            # PADDING
            sentence = pad_sequence(sentence, padding_value=0, batch_first=True).to(device)
            tag = pad_sequence(tag, padding_value=0, batch_first=True).to(device)
            # 予測
            ner_logits = model(sentence)
            # 損失の計算
            ner_loss = compute_ner_loss(model, ner_logits, tag)
            # 誤差伝搬
            ner_loss.backward()
            # 勾配を更新
            optimizer.step()
            scheduler.step()
            # 合計の損失
            ner_running_loss += ner_loss.item()
            
        logger.info("訓練")
        logger.info("{0}エポック目のNERの損失値: {1}\n".format(epoch, ner_running_loss))

        # 検証
        ner_preds = []
        with torch.inference_mode():
            # 勾配を更新しない
            model.eval()
            # 
            val_ner_running_loss = 0
            # バッチごとの処理
            pbar_val = tqdm(range(0, len(val_vecs), args.batch_size))
            for ofs in pbar_val:
            #for ofs in range(0, len(val_vecs), hyper.batch_size):
                pbar_train.set_description('モデルを検証中!')
                #　バッチ分だけ取り出す (1)
                begin_index = ofs
                end_index = min(len(val_vecs), ofs + args.batch_size)
                #　バッチ分だけ取り出す (2)
                batch_X = copy.deepcopy(val_vecs[begin_index: end_index])
                batch_ner = copy.deepcopy(ner_val_labels[begin_index: end_index])
                # ベクトル
                sentence = [torch.tensor(x) for x in batch_X]
                tag = [torch.tensor(x) for x in batch_ner]
                # PADDING
                sentence = pad_sequence(sentence, padding_value=0, batch_first=True).to(device)
                tag = pad_sequence(tag, padding_value=0, batch_first=True).to(device)
                # 予測
                ner_logits = model(sentence)
                ner_preds.append(ner_logits)
                # 損失の計算
                ner_loss = compute_ner_loss(model, ner_logits, tag)
                # 合計の損失
                val_ner_running_loss += ner_loss.item()

        # 評価
        ner_preds_decode = decode_ner_pipeline(model, ner_val_labels, ner_preds, tag2idx, val_vecs)
        res_df = result2df_for_ner(X_val, ner_preds_decode)
        sreict_ner = eval_ner_strict(res_df)
        val_F = sreict_ner["micro avg"]["f1"] 

        # 保存
        logger.info("検証")
        logger.info("{0}エポック目のNERの損失値: {1}".format(epoch, val_ner_running_loss))
        logger.info("{0}エポック目のNERのMicro Avg: {1}".format(epoch, val_F))

        loss_dct["epoch"].append(epoch)
        loss_dct["train_NER_loss"].append(ner_running_loss)
        loss_dct["val_NER_loss"].append(val_ner_running_loss)

        # Early Stopping
        if val_F > best_val_F:
            logger.info("{0}エポック目で更新\n".format(epoch))
            torch.save(model.module.state_dict(), 
                       './models/{0}/{1}/NER/ner_model_{2}.pt'.format(args.task, args.bert_type, fold))
            best_val_F = val_F
        else:
            logger.info("{0}エポック目は現状維持\n".format(epoch))

    with open('./results/{0}/{1}/loss_{2}.json'.format(args.task, args.bert_type, fold), 'w') as f:
        json.dump(loss_dct, f, indent=4)

            
def test_loop_ner(X_test, test_vecs, ner_test_labels, fold, tag2idx, args, device):
    # テスト
    with torch.inference_mode():
        # モデルを定義
        model = BERT_CRF(args, tag2idx).to(device)
        #検証データでの損失が最良となったベストモデルを読み込む
        model.load_state_dict(torch.load('./models/{0}/{1}/NER/ner_model_{2}.pt'.format(args.task, args.bert_type, fold)))
        model = torch.nn.DataParallel(model)
        # 勾配を更新しない
        model.eval()
        # 結果
        ner_preds = []
        # バッチごとの処理
        for ofs in range(0, len(test_vecs), args.batch_size):
            #　バッチ分だけ取り出す (1)
            begin_index = ofs
            end_index = min(len(test_vecs), ofs + args.batch_size)
            #　バッチ分だけ取り出す (2)
            batch_X = test_vecs[begin_index: end_index]
            batch_ner = ner_test_labels[begin_index: end_index]
            # ベクトル
            sentence = [torch.tensor(x) for x in batch_X]
            tag = [torch.tensor(x) for x in batch_ner]
            # PADDING
            sentence = pad_sequence(sentence, padding_value=0, batch_first=True).to(device)
            tag = pad_sequence(tag, padding_value=0, batch_first=True).to(device)
            # 予測
            ner_logits = model(sentence)
            ner_preds.append(ner_logits)
    # 評価
    ner_preds_decode = decode_ner_pipeline(model, ner_test_labels, ner_preds, tag2idx, test_vecs)
    res_df = result2df_for_ner(X_test, ner_preds_decode)
    return res_df