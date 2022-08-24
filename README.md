# MEDINFO2023

## To Do

- [x] 前処理用コードの修正
- [ ] 実験用コードの修正
- [ ] 実験用コードのテスト
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

## Results (Micro-F1)

### Joint

Strict: 2つの固有表現が完全に正しく抽出できており、かつ関係が正しい場合に正解

Soft: 2つの固有表現のSPANが正しく抽出できており（タグは無視）、かつ関係が正しい場合に正解

| Fold | NER (UTH) |RE (UTH: Soft)| RE (UTH: Strict)| NER (NICT) |RE (NICT: Soft)|RE (NICT: Strict)|
|:---|---:|---:|---:|---:|---:|---:|
|1 |0.903|0.792|0.750|0.XXX|0.XXX|0.XXX|
|2 |0.914|0.800|0.766|0.XXX|0.XXX|0.XXX|
|3 |0.910|0.803|0.766|0.XXX|0.XXX|0.XXX|
|4 |0.914|0.803|0.762|0.XXX|0.XXX|0.XXX|
|5 |0.907|0.796|0.756|0.XXX|0.XXX|0.XXX|
|Avg. |0.910|0.799|0.760|0.XXX|0.XXX|0.XXX|

### Pipeline

| Fold | NER (UTH) |RE (UTH: Soft)| RE (UTH: Strict)| NER (NICT) |RE (NICT: Soft)|RE (NICT: Strict)|
|:---|---:|---:|---:|---:|---:|---:|
|1 |0.916|0.799|0.762|0.XXX|0.XXX|0.XXX|
|2 |0.920|0.803|0.771|0.XXX|0.XXX|0.XXX|
|3 |0.920|0.802|0.770|0.XXX|0.XXX|0.XXX|
|4 |0.928|0.808|0.771|0.XXX|0.XXX|0.XXX|
|5 |0.926|0.810|0.775|0.XXX|0.XXX|0.XXX|
|Avg. |0.922|0.804|0.769|0.XXX|0.XXX|0.XXX|

# License
CC BY-NC-SA 4.0

## References

- Ma, Y., Hiraoka, T., & Okazaki, N. (2022). Named entity recognition and relation extraction using enhanced table filling by contextualized representations. Journal of Natural Language Processing, 29(1), 187-223.. Software available from https://github.com/YoumiMa/TablERT.
- pytorch-crf. Software available from https://pytorch-crf.readthedocs.io/en/stable/.
- Hiroki Nakayama. seqeval: A python framework for sequence labeling evaluation, 2018. Software available from https://github.com/chakki-works/seqeval.

## Citation

```
```
