# Towards structuring clinical texts: joint entity and relation extraction from Japanese case report corpus

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

An NVIDIA RTX A6000 (48GB) was used for training the models.
Modify the arguments in the code according to your environment.

```
bash run_joint.sh
```

## Results (Micro-F1)

### Joint

| Fold | NER (UTH) |RE (UTH: Strict)| NER (NICT) |RE (NICT)|
|:---|---:|---:|---:|---:|
|1 |0.911|0.761|0.894|0.745|
|2 |0.909|0.757|0.886|0.735|
|3 |0.910|0.759|0.890|0.740|
|4 |0.918|0.759|0.901|0.747|
|5 |0.910|0.758|0.887|0.736|
|Avg. |0.912|0.759|0.892|0.741|



# License
CC BY-NC-SA 4.0

## References

- Ma, Y., Hiraoka, T., & Okazaki, N. (2022). Named entity recognition and relation extraction using enhanced table filling by contextualized representations. Journal of Natural Language Processing, 29(1), 187-223.. Software available from https://github.com/YoumiMa/TablERT.
- pytorch-crf. Software available from https://pytorch-crf.readthedocs.io/en/stable/.
- Hiroki Nakayama. seqeval: A python framework for sequence labeling evaluation, 2018. Software available from https://github.com/chakki-works/seqeval.

## Citation

```
```
