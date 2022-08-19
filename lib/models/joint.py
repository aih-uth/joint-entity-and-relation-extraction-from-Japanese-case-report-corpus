import torch
import torch.nn as nn
import torch.utils.data
from lib.models import CRF
# from torchcrf import CRF
from transformers import BertModel
from lib.models.attention import MultiHeadAttention


class BERT_TF(nn.Module):
    def __init__(self, args, tag2idx, rel2idx, device):
        super(BERT_TF, self).__init__()
        self.bert_model = BertModel.from_pretrained(args.bert_path)
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(768, len(tag2idx))
        self.crf = CRF(tag2idx, batch_first=True)
        self.label_embedding = nn.Embedding(len(tag2idx), 50, padding_idx=0)
        # MHAのパラメーター: 実装を確認すること!!
        relation_labels = len(rel2idx)
        hidden_size = 768
        entity_label_embedding = 50
        att_hidden = 20
        self.rel_classifier = MultiHeadAttention(relation_labels, 
                                                 hidden_size + entity_label_embedding, 
                                                 att_hidden, device)
    def forward(self, sentence, tag, train_flag, device):
        input_mask = (sentence != 0)#.to(device)
        embed = self.bert_model(sentence, attention_mask=input_mask, token_type_ids=None)
        embed = embed["last_hidden_state"][:, 1: -1,:]
        embed = self.dropout(embed)
        ner_output = self.linear(embed)
        if train_flag:
            le = self.label_embedding(tag)
        else:
            le = self.label_embedding(torch.tensor(self.crf.decode(ner_output)).to(device))
        hx = torch.cat([embed, le], axis=2)
        # 関係抽出
        rel_logits = self.rel_classifier(hx, hx, hx)
        return ner_output, rel_logits


class BERT_TF_REL(nn.Module):
    def __init__(self, hyper, tag2idx, rel2idx, device):
        super(BERT_TF_REL, self).__init__()
        self.bert_model = BertModel.from_pretrained(hyper.bert_path)
        self.dropout = nn.Dropout(0.5)
        self.label_embedding = nn.Embedding(len(tag2idx), 50, padding_idx=0)
        # MHAのパラメーター: 実装を確認すること!!
        relation_labels = len(rel2idx)
        hidden_size = 768
        entity_label_embedding = 50
        att_hidden = 20
        self.rel_classifier = MultiHeadAttention(relation_labels, 
                                                 hidden_size + entity_label_embedding, 
                                                 att_hidden, device)
    def forward(self, sentence, tag):
        input_mask = (sentence != 0)#.to(device)
        embed = self.bert_model(sentence, attention_mask=input_mask, token_type_ids=None)
        embed = embed["last_hidden_state"][:, 1: -1,:]
        embed = self.dropout(embed)
        le = self.label_embedding(tag)
        hx = torch.cat([embed, le], axis=2)
        rel_logits = self.rel_classifier(hx, hx, hx)
        return rel_logits


class BERT_CRF(nn.Module):
    def __init__(self, hyper, tag2idx):
        super(BERT_CRF, self).__init__()
        self.bert_model = BertModel.from_pretrained(hyper.bert_path)
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(768, len(tag2idx))
        self.crf = CRF(tag2idx, batch_first=True)
    def forward(self, sentence):
        input_mask = (sentence != 0)#.to(device)
        embed = self.bert_model(sentence, attention_mask=input_mask, token_type_ids=None)
        embed = embed["last_hidden_state"][:, 1: -1,:]
        embed = self.dropout(embed)
        ner_output = self.linear(embed)
        return ner_output