# MEDINFO2023

## To Do

- [x] 前処理用コードの修正
- [x] 実験用コードの修正
- [x] 実験用コードのテスト (maybe)
- [ ] 論文用の学習データの決定
- [ ] 論文用の実験（検証データの評価も簡易版ではなく、テスト用の評価に置き換える?）
- [ ] READMEの修正

## Setpu

BERTs and dataset can be downloaded from the following.
- [UTH-BERT](https://ai-health.m.u-tokyo.ac.jp/home/research/uth-bert)
- [NICT-BERT](https://alaginrc.nict.go.jp/nict-bert/index.html)
- [iCorpus](https://ai-health.m.u-tokyo.ac.jp/home/research/corpus)

## Requirements

- Python 3.8+
- pandas 1.2.4
- numpy 1.20.1
- torch 1.10.1+cu113
- scikit-learn 0.24.1
- transformers 4.11.3
- seqeval 1.2.2

## Run

Modify the arguments in the code according to your environment.

```
bash run_all.sh
```

## Results (Micro-F1: テストなので60エポック、学習データは本実験で差し替えること)

### Joint

Strict: 2つの固有表現が完全に正しく抽出できており、かつ関係が正しい場合に正解
Soft: 2つの固有表現のSPANが正しく抽出できており（タグは無視）、かつ関係が正しい場合に正解

| Fold | NER (UTH) |RE (UTH: Soft)| RE (UTH: Strict)| NER (NICT) |RE (NICT: Soft)|RE (NICT: Strict)|
|:---|---:|---:|---:|---:|---:|---:|
|1 |0.897|0.778|0.730|0.886|0.774|0.727|
|2 |0.901|0.781|0.740|0.888|0.774|0.734|
|3 |0.900|0.778|0.732|0.882|0.774|0.723|
|4 |0.905|0.777|0733.|0.894|0.779|0.733|
|5 |0.903|0.789|0.743|0.895|0.791|0.741|
|Avg. |0.901|0.781|0.736|0.889|0.778|0.731|

### Pipeline

| Fold | NER (UTH) |RE (UTH: Soft)| RE (UTH: Strict)| NER (NICT) |RE (NICT: Soft)|RE (NICT: Strict)|
|:---|---:|---:|---:|---:|---:|---:|
|1 |0.906|0.776|0.735|0.899|0.776|0.735|
|2 |0.909|0.781|0.744|0.899|0.774|0.737|
|3 |0.907|0.777|0.736|0.903|0.784|0.739|
|4 |0.920|0.789|0.750|0.910|0.786|0.745|
|5 |0.909|0.783|0.739|0.908|0.795|0.752|
|Avg. |0.910|0.781|0.741|0.904|0.783|0.742|

# License
CC BY-NC-SA 4.0

## References

- Ma, Y., Hiraoka, T., & Okazaki, N. (2022). Named entity recognition and relation extraction using enhanced table filling by contextualized representations. Journal of Natural Language Processing, 29(1), 187-223.. Software available from https://github.com/YoumiMa/TablERT.
- pytorch-crf. Software available from https://pytorch-crf.readthedocs.io/en/stable/.
- Hiroki Nakayama. seqeval: A python framework for sequence labeling evaluation, 2018. Software available from https://github.com/chakki-works/seqeval.

## Citation

```
```
