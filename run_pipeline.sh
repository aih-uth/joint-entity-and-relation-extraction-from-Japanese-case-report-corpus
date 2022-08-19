#uth_bert_path=../BERT/UTH_BERT_BASE_512_MC_BPE_WWM_V25000_352K
uth_bert_path=/Users/shibata/Documents/BERT_v2/UTH_BERT_BASE_512_MC_BPE_WWM_V25000_352K
nict_bert_path=../BERT/NICT_BERT-base_JapaneseWikipedia_32K_BPE

# UTH
python main_ner.py --bert_path $uth_bert_path --bert_type UTH
python main_re.py --bert_path $uth_bert_path --bert_type UTH

# NICT
#python main_ner.py --bert_path $nict_bert_path --bert_type UTH --data_path ./data/csv/CR_conll_format_arbitrary_NICT_for_experiment.csv
#python main_re.py --bert_path $nict_bert_path --bert_type UTH --data_path ./data/csv/CR_conll_format_arbitrary_NICT_for_experiment.csv
