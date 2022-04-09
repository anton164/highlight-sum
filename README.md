# Main repo
```
conda create -n qa-sum python=3.8
conda activate qa-sum
```

### Generate questions with QAGen Model
1. Create separate conda env & install dependencies
```
cd fact-check-summarization
conda create -n fact-check-summarization python=3.6
conda activate fact-check-summarization
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
python sm_inference_asum.py --task gen_qa --base_dir ../cache --source_dir xsum/processed-data --output_dir ../cache/xsum/qagen --num_workers 1 --bsz 5 --beam 60 --max_len 60 --min_len 8 --checkpoint_dir ../cache/qagen-model --ckp_file checkpoint2.pt --bin_dir xsum/processed-data/data_bin --diverse_beam_groups 60 --diverse_beam_strength 0.5 --batch_lines True --input_file test.target.hypo --return_token_scores True
```