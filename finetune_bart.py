import numpy as np
from datasets import load_dataset, load_metric, load_from_disk
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from preprocess.tokenize_utils import tokenize_with_cache
from preprocess.preprocess_data_xsum import (
    write_filtered_summary_sentences_and_entities,
)
import argparse
import wandb
import nltk

rouge_metric = load_metric("rouge")
nltk.download("punkt", quiet=True)


def load_data(dataset_name: str, is_subset=False):
    if dataset_name == "xsum-subset":
        xsum = load_dataset("xsum")
        # take subset of xsum train
        xsum_train_subset = xsum["train"].train_test_split(test_size=1000, seed=42)[
            "test"
        ]
        # split subset
        dataset = xsum_train_subset.train_test_split(test_size=0.2, seed=42)
        data_train = dataset["train"]
        data_val = dataset["test"]
    elif dataset_name == "xsum-entity-filter":
        dataset = load_from_disk("data/huggingface/xsum-entity-filter")
        data_train = dataset["train"]
        data_val = dataset["validation"]

    print(f"Dataset: {dataset_name}", dataset)
    train_tokenized = tokenize_with_cache(
        f"{dataset_name}-train", tokenizer, data_train
    )
    val_tokenized = tokenize_with_cache(f"{dataset_name}-val", tokenizer, data_val)
    return train_tokenized, val_tokenized


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


def compute_metrics(tokenizer, eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = rouge_metric.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    # Extract a few results from ROUGE
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
    ]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


if __name__ == "__main__":
    MODEL_NAME = "facebook/bart-large"
    parser = argparse.ArgumentParser()
    parser.add_argument("run_name", type=str)
    parser.add_argument("--train_epochs", type=int, default=1)
    parser.add_argument(
        "--dataset",
        type=str,
        default="xsum-subset",
        help="xsum, xsum-entity-filter, xsum-subset",
    )
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--val_batch_size", type=int, default=4)
    parser.add_argument("--grad_acc", type=int, default=1)
    parser.add_argument("--resume_from_checkpoint", type=bool, default=False)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    args = parser.parse_args()
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_tokenized, val_tokenized = load_data(args.dataset)
    wandb.login()

    training_args = Seq2SeqTrainingArguments(
        output_dir=f"{args.checkpoint_dir}/{args.run_name}",
        num_train_epochs=args.train_epochs,
        per_device_train_batch_size=args.train_batch_size,
        do_train=True,
        do_eval=True,
        evaluation_strategy="steps",
        eval_steps=500,
        per_device_eval_batch_size=args.val_batch_size,
        # learning_rate=3e-05,
        warmup_steps=500,
        weight_decay=0.1,
        label_smoothing_factor=0.1,
        predict_with_generate=True,
        logging_dir="hf-logs",
        logging_steps=50,
        save_total_limit=2,
        # optimization
        fp16=True,
        gradient_accumulation_steps=args.grad_acc,
        # gradient_checkpointing=True,
        # optim="adafactor",
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        tokenizer=tokenizer,
        compute_metrics=lambda preds: compute_metrics(tokenizer, preds),
    )

    wandb_run = wandb.init(project="highlight-sum", entity="danton-nlp")
    wandb_run.name = args.run_name
    config = wandb.config
    config.update(args)
    
    print("Training...", trainer.train(
        resume_from_checkpoint=bool(args.resume_from_checkpoint)
    ))
    print("Saving model...", trainer.save_model(
        f"{args.checkpoint_dir}/{args.run_name}/final"
    ))

    print("Eval after training", trainer.evaluate())
