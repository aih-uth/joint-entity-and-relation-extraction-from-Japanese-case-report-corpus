# Joint (UTH)
! python main_joint.py --bert_type UTH --max_epoch 100 --data_path ./data/csv/UTH_CR_conll_format_arbitrary_UTH.csv --max_words 510 --bert_path ../../BERT/UTH_BERT_BASE_512_MC_BPE_WWM_V25000_352K --re_weight 1

# Joint (NICT)
! python main_joint.py --bert_type NICT --max_epoch 100 --data_path ./data/csv/UTH_CR_conll_format_arbitrary_NICT.csv --max_words 510 --bert_path ../../BERT/NICT_BERT-base_JapaneseWikipedia_32K_BPE --re_weight 1
