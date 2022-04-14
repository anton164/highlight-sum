# Setup
```
conda create -n highlight-sum python=3.8
conda activate highlight-sum
```

## Run Streamlit app\
```
streamlit run app.py
```

# External repositories

in `/external-repos`
### Generate questions with `/question_generation` model
Github repo: https://github.com/patil-suraj/question_generation

1. Create separate conda env & install dependencies
```
cd external-repos/suraj-question_generation
conda create -n question_generation python=3.8
conda activate question_generation
pip install transformers==3.0.0
pip install nltk==3.7
python -m nltk.downloader punkt
```

### Generate questions with `/question_generator` model
Github repo: https://github.com/AMontgomerie/question_generator  
Notebook: https://colab.research.google.com/drive/1SyMepcPlxSVG_anRUwiMo9w5TjSewNOx?authuser=1#scrollTo=5fwqys_1hwbi

1. Create separate conda env & install dependencies
```
cd external-repos/AMontgomerie-question_generator
conda create python=3.8 -n question_generator
conda activate question_generator
pip install -r requirements.txt -qq
```

2. Generate questions
```
! python run_qg.py --text_file articles/xsum_source_2.txt --answer_style multiple_choice
```

### Generate questions with Amazon QAGen Model
Github repo: https://github.com/amazon-research/fact-check-summarization  
Notebook:  https://colab.research.google.com/drive/1fhncXX3-V9cgJNEpRF9Ch8etMtIkTv0B?authuser=1#scrollTo=uTMMMzYMdyE7    
Data: [data/qa/qagen](data/qa/qagen)

1. Create separate conda env & install dependencies
```
cd external-repos/fact-check-summarization
conda create -n qagen-model python=3.6
conda activate qagen-model
pip install -r requirements.txt
pip install --editable ./
python -m spacy download en_core_web_lg
```
2. Download [checkpoint model to cache/qagen-model folder](https://fact-check-summarization.s3.amazonaws.com/newsqa-squad-qagen-checkpoint/checkpoint2.pt)
3. Download BPE tokenizer
```
mkdir cache/bpe-dir
wget -O cache/bpe-dir/encoder.json 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
wget -O cache/bpe-dir/vocab.bpe 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'
wget -O cache/bpe-dir/dict.txt 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt'
```
4. Preprocess subset of XSUM test data:
```
mkdir cache/xsum/processed-data
python preprocess/data_prepro_clean.py --mode preprocess_xsum_huggingface --output_dir cache/xsum/processed-data --filter_level 0
```
5. Tokenize subset of XSUM test data:
```
cd preprocess
python data_prepro_clean.py --mode bpe_binarize_test --input_dir ../cache/xsum/processed-data --tokenizer_dir ../cache/bpe-dir
```
5. Prepare summaries into jsonl format
```
cd preprocess
python evaluate_hypo.py --mode convert_hypo_to_json --base_dir ../cache --sub_dir xsum/processed-data --split test --pattern .target
```
6. Generate Questions with pre-trained QAGen model
```
cd preprocess
mkdir ../cache/xsum/qagen
python sm_inference_asum.py --task gen_qa --base_dir ../cache --source_dir xsum/processed-data --output_dir ../cache/xsum/qagen --num_workers 1 --bsz 5 --beam 60 --max_len 60 --min_len 8 --checkpoint_dir ../cache/qagen-model --ckp_file checkpoint2.pt --bin_dir /Users/anton164/git/qa-sum/qa-gen-model/cache/xsum/processed-data/data_bin --diverse_beam_groups 60 --diverse_beam_strength 0.5 --batch_lines True --input_file test.target.hypo --return_token_scores True
```
7. Filter generated qa-pairs
```
python evaluate_hypo.py --mode filter_qas_dataset_lm_score --base_dir ../cache --sub_dir xsum/processed-data --pattern test.target.hypo.beam60.qas
```