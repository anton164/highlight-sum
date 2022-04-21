from datasets import load_dataset
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from typing import List, Tuple
import re
from sumtool.storage import store_model_summaries


TRACKING_ENTITY_LIST = [
    "PERSON",
    "FAC",
    "GPE",  # Geopolitical entity, i.e. countries, cities, states.
    "ORG",
    "NORP",
    "LOC",
    "EVENT",
]


def entity_match(ent, source):
    ent_split = ent.split()
    result = []
    for l in range(len(ent_split), 1, -1):
        for start_i in range(len(ent_split) - l + 1):
            sub_ent = " ".join(ent_split[start_i : (start_i + l)])
            if re.search(re.escape(sub_ent), source, re.IGNORECASE):
                result.append(sub_ent)
        if result:
            break
    if result:
        return result
    else:
        for token in ent_split:
            if token.lower() not in STOP_WORDS or token == "US":
                if re.search(re.escape(token), source, re.IGNORECASE):
                    result.append(token)
        return result
    # return []


def select_summary_sentences_and_entities(
    nlp, source: str, summary: str
) -> Tuple[str, List[spacy.tokens.span.Span]]:
    """
    This function does filters out sentences of the _summary_ that are not supported by the document.
    A summary sentence is supported if that sentence included a named entity that is in the document.
    """

    summary_nlp = nlp(summary)

    en_count_in_summary = 0

    sentences_selected = {}
    selected_entities = tuple()
    missing_entities = tuple()

    for sent in summary_nlp.sents:
        sentences_selected[sent.text] = True
    for e in summary_nlp.ents:
        if e[0].ent_type_ in TRACKING_ENTITY_LIST:
            en_count_in_summary += 1
            match_result = entity_match(e.text, source)
            if match_result:
                selected_entities = selected_entities + (e,)
            else:
                missing_entities = missing_entities + (e,)
                sentences_selected[e.sent.text] = False
    result = []
    for sent in summary_nlp.sents:
        if sentences_selected[sent.text]:
            result.append(sent.text)
    return " ".join(result), selected_entities, missing_entities


def write_filtered_summary_sentences_and_entities(example):
    supported_sents, supported_ents, missing_ents = select_summary_sentences_and_entities(
        nlp, example["document"], example["summary"]
    )
    example["supported_summary_sentences"] = supported_sents
    example["supported_summary_entities"] = [ent.text for ent in supported_ents]
    example["missing_summary_entities"] = [ent.text for ent in missing_ents]

    return example


def write_index(example, index):
    example["id"] = index
    return example


def build_metadata_dict(dataset) -> dict:
    return {
        example['id']: {
            "supported_summary_sentences": example['supported_summary_sentences'],
            "supported_summary_entities": example['supported_summary_entities'],
            "missing_summary_entities": example['missing_summary_entities'],
            "source": example['document']
        }
        for example in dataset
    }


if __name__ == "__main__":
    xsum_test = load_dataset("xsum", split="test")

    nlp = spacy.load("en_core_web_lg")
    test_sample_annotated = xsum_test.map(
        write_filtered_summary_sentences_and_entities, num_proc=4
    )

    c_supported_sents = 0
    for x in test_sample_annotated:
        if len(x["supported_summary_sentences"]) > 0:
            c_supported_sents += 1
    print(f"Supported sentences: {c_supported_sents}/{len(test_sample_annotated)} ({c_supported_sents/len(test_sample_annotated):.2%})")


    id2gold_summary = dict(
        zip(test_sample_annotated["id"], test_sample_annotated["summary"])
    )

    xsum_gold_standard_config = {
        "Data": "XSUM",
    }
    store_model_summaries(
        "xsum",
        "gold",
        xsum_gold_standard_config,
        id2gold_summary,
        build_metadata_dict(test_sample_annotated)
    )
