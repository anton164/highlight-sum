from typing import List
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from datasets import load_dataset
from preprocess.tokenize_utils import tokenize_with_cache
import argparse
import torch


def entropy(p_dist: torch.Tensor) -> float:
    """"
    Calculates Shannon entropy for a probability distribution

    Args:
        p_dist: probability distribution (torch.Tensor)

    Returns:
        entropy (float)
    """
    # add epsilon because log(0) = nan
    p_dist = p_dist.view(-1) + 1e-12
    return - torch.mul(
        p_dist,
        p_dist.log()
    ).sum(0).item()


def generate_summaries_with_beam_tree(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    docs_to_summarize: List[str],
    num_beams: int = 4,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    inputs = tokenizer(
        docs_to_summarize,
        max_length=1024,
        truncation=True,
        return_tensors="pt",
        padding=True,
    )
    input_token_ids = inputs.input_ids.to(device)

    model_output = model.generate(
        input_token_ids,
        num_beams=num_beams,
        # max_length=0,
        early_stopping=True,
        return_dict_in_generate=True,
        output_scores=True,
    )

    generated_summaries = [
        tokenizer.decode(
            id, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        for id in model_output.sequences
    ]

    # reshape model_output scores to (n_seqs x seq len x n_beams x vocab)
    model_beam_scores = torch.stack(model_output.scores).reshape(
        len(model_output.scores),
        len(generated_summaries),
        num_beams,
        -1
    ).permute(1, 0, 2, 3)
    # Collect Beam Search Metadata
    beams_metadata = []
    for seq_idx in range(model_output.sequences.shape[0]):
        top_beam_indices = [x.item() for x in model_output.beam_indices[seq_idx]]
        seq_beams = {
            "beams": [list() for _ in range(num_beams)],
            "selected_beam_indices": top_beam_indices
        }
        beams_metadata.append(seq_beams)

        for idx, output_token_id in enumerate(model_output.sequences[seq_idx][1:]):
            # beam_idx = model_output.beam_indices[seq_idx][idx]
            for beam_idx in range(num_beams):
                beam_probs = torch.exp(model_beam_scores[seq_idx][idx][beam_idx])
                beam_top_alternatives = []
                top_probs = torch.topk(beam_probs, k=3)
                for i, v in zip(top_probs.indices, top_probs.values):
                    beam_top_alternatives.append(
                        {
                            "token": tokenizer.decode(i),
                            "token_id": i.item(),
                            "probability": v.item(),
                        }
                    )
                print("appending at", beam_idx)
                seq_beams["beams"][beam_idx].append({
                    "top_tokens": beam_top_alternatives,
                    "entropy": entropy(beam_probs),
                    # "token_id": output_token_id,
                    # "token": tokenizer.decode(output_token_id),
                    # "beam_token_prob": selected_beam_probs[output_token_id].item(),
                    # "beam_idx": beam_idx.item(),
                    # "token_in_input": output_token_id in input_set,
                })

    return generated_summaries, beams_metadata


def load_model_and_tokenizer(path):
    return (
        AutoModelForSeq2SeqLM.from_pretrained(path),
        AutoTokenizer.from_pretrained(path),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="facebook/bart-large-xsum")
    # parser.add_argument("--sumtool_path", type=str, default="")
    # parser.add_argument("--data_subset", type=int, default=0)
    args = parser.parse_args()
    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer(args.model_path)

    xsum_test = load_dataset("xsum")["test"]

    generate_summaries_with_beam_tree(model, tokenizer, xsum_test["document"][:2])
