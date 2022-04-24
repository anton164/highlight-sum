import pickle
import os


def batch_tokenize_preprocess(batch, tokenizer, max_source_length, max_target_length):
    source, target = batch["document"], batch["summary"]
    source_tokenized = tokenizer(
        source, padding="max_length", truncation=True, max_length=max_source_length
    )
    target_tokenized = tokenizer(
        target, padding="max_length", truncation=True, max_length=max_target_length
    )

    batch = {k: v for k, v in source_tokenized.items()}
    # Ignore padding in the loss
    batch["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in l]
        for l in target_tokenized["input_ids"]
    ]
    return batch


def tokenize(tokenizer, data, max_decode_length=60):
    tokenized = data.map(
        lambda batch: batch_tokenize_preprocess(
            batch,
            tokenizer,
            512,  # 512 according to https://github.com/google-research/pegasus/issues/159#issue-782635176
            max_decode_length,  # 60 for train / 100 for test https://github.com/huggingface/transformers/blob/main/examples/research_projects/seq2seq-distillation/README.md
        ),
        batched=True,
        remove_columns=data.column_names,
    )
    return tokenized


def tokenize_with_cache(name, tokenizer, data, max_decode_length=60):
    dir_name = "./data/tokenized/"
    os.makedirs(dir_name, exist_ok=True)
    fpath = f"{dir_name}{name}-{max_decode_length}.pickle"
    if os.path.isfile(fpath):
        try:
            with open(fpath, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print("Failed to load from cache", e)

    print(f"building tokenization cache {fpath}...")
    tokenized = tokenize(tokenizer, data, max_decode_length)

    with open(fpath, "wb") as f:
        pickle.dump(tokenized, f)
    return tokenized
