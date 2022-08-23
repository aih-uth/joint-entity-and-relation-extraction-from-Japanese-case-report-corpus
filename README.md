# MEDINFO2023

## To Do

- [x] 前処理用コードの修正
- [ ] 実験用コードの修正
- [ ] 実験用コードのテスト
- [ ] 論文用の学習データの決定
- [ ] 論文用の実験
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

## Results (TEST)

### Joint (Micro-F1)

| Fold | NER (UTH) |RE (UTH)| NER (NICT) |RE (NICT)|
|:---|---:|---:|---:|---:|
|1 |0.YYY|0.YYY|0.XXX|0.XXX|
|2 |0.YYY|0.YYY|0.XXX|0.XXX|
|3 |0.YYY|0.YYY|0.XXX|0.XXX|
|4 |0.YYY|0.YYY|0.XXX|0.XXX|
|5 |0.YYY|0.YYY|0.XXX|0.XXX|
|Avg. |0.YYY|0.YYY|0.XXX|0.XXX|

### Pipeline (Micro-F1)

| Fold | NER (UTH) |RE (UTH)| NER (NICT) |RE (NICT)|
|:---|---:|---:|---:|---:|
|1 |0.YYY|0.YYY|0.XXX|0.XXX|
|2 |0.YYY|0.YYY|0.XXX|0.XXX|
|3 |0.YYY|0.YYY|0.XXX|0.XXX|
|4 |0.YYY|0.YYY|0.XXX|0.XXX|
|5 |0.YYY|0.YYY|0.XXX|0.XXX|
|Avg. |0.YYY|0.YYY|0.XXX|0.XXX|

# License
CC BY-NC-SA 4.0

## References

- Ma, Y., Hiraoka, T., & Okazaki, N. (2022). Named entity recognition and relation extraction using enhanced table filling by contextualized representations. Journal of Natural Language Processing, 29(1), 187-223.. Software available from https://github.com/YoumiMa/TablERT.
- pytorch-crf. Software available from https://pytorch-crf.readthedocs.io/en/stable/.
- Hiroki Nakayama. seqeval: A python framework for sequence labeling evaluation, 2018. Software available from https://github.com/chakki-works/seqeval.

## Citation

```
```
