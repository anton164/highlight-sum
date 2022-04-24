from datasets import load_dataset, load_from_disk
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from evaluation.metrics import compute_metrics
from preprocess.tokenize_utils import tokenize_with_cache
import argparse
import wandb
import numpy.random as np_random


def load_data(dataset_name: str):
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
        np_random.seed(42)
        data_val = dataset["validation"].select(
            np_random.choice(len(dataset["validation"]), 250, replace=False)
        )

    print(f"Dataset: {dataset_name}", dataset)
    train_tokenized = tokenize_with_cache(
        f"{dataset_name}-train", tokenizer, data_train
    )
    val_tokenized = tokenize_with_cache(f"{dataset_name}-val", tokenizer, data_val)
    return train_tokenized, val_tokenized


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
