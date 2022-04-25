from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from datasets import load_dataset, load_from_disk
from preprocess.tokenize_utils import tokenize_with_cache
import argparse
from evaluation.metrics import compute_metrics
from sumtool.storage import store_model_summaries


def load_data(dataset_name, tokenizer, data_subset):
    if dataset_name == "xsum":
        dataset = load_dataset("xsum")
        data_test = dataset["test"]
    if dataset_name == "xsum-entity-filter":
        dataset = load_from_disk("data/huggingface/xsum-entity-filter")
        data_test = dataset["test"]

    if data_subset != 0:
        data_test = data_test.train_test_split(test_size=data_subset, seed=42)["test"]
    data_test = tokenize_with_cache(
        f"{dataset_name}-test-{data_subset}", tokenizer, data_test, 100
    )
    return data_test


def load_model_and_tokenizer(path):
    return (
        AutoModelForSeq2SeqLM.from_pretrained(path),
        AutoTokenizer.from_pretrained(path),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str)
    parser.add_argument("--sumtool_path", type=str, default="")
    parser.add_argument("--dataset", type=str, default="xsum-entity-filter")
    parser.add_argument("--data_subset", type=int, default=0)
    parser.add_argument("--val_batch_size", type=int, default=32)
    args = parser.parse_args()
    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    trainer = Seq2SeqTrainer(
        args=Seq2SeqTrainingArguments(
            output_dir="hf-inference",
            per_device_eval_batch_size=args.val_batch_size,
            fp16=True,
            predict_with_generate=True,
        ),
        model=model,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=lambda preds: compute_metrics(tokenizer, preds),
    )
    print("Loading data...")
    test_data = load_data(args.dataset, tokenizer, args.data_subset)
    print(test_data)
    result = trainer.predict(
        test_data,
    )
    print("Metrics", result.metrics)
    decoded_preds = tokenizer.batch_decode(result.predictions, skip_special_tokens=True)
    generated_summaries = {
        sum_id: summary
        for sum_id, summary in zip(test_data["id"], decoded_preds)
    }
    if args.sumtool_path != "":
        store_model_summaries(
            "xsum",
            args.sumtool_path,
            model.config.to_dict(),
            generated_summaries
        )
