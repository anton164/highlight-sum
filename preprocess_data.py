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

    for sent in summary_nlp.sents:
        sentences_selected[sent.text] = True
    for e in summary_nlp.ents:
        if e[0].ent_type_ in TRACKING_ENTITY_LIST:
            en_count_in_summary += 1
            match_result = entity_match(e.text, source)
            if match_result:
                selected_entities = selected_entities + (e,)
            else:
                sentences_selected[e.sent.text] = False
    result = []
    for sent in summary_nlp.sents:
        if sentences_selected[sent.text]:
            result.append(sent.text)
    return " ".join(result), selected_entities


def should_exclude_summary(summary: str) -> bool:
    """
    Exclude summary iff number of words is less than 8 OR it matches
    particular regexs.
    """
    if len(summary.split()) < 8:
        return True

    match = (
        re.search(re.escape("on FoxNews.com"), summary, re.IGNORECASE)
        or re.search(re.escape("from FoxNews.com"), summary, re.IGNORECASE)
        or re.search(
            re.escape("Collection of all USATODAY.com"), summary, re.IGNORECASE
        )
        or re.search(re.escape("washingtonpost.com"), summary, re.IGNORECASE)
    )

    if match:
        return True
    else:
        return False


def should_exclude_source(source):
    if len(source.split()) < 50:
        return True
    if (
        (source.startswith("Image ") and source[6] in "0123456789")
        or source.startswith("Photo: ")
        or '"params":' in source
    ):
        return True
    return False


def mark_exclusion(example):
    example["valid_summary"] = not should_exclude_summary(example["text"])
    example["valid_source"] = not should_exclude_source(example["text"])
    return example


def write_filtered_summary_sentences_and_entities(example):
    supported_sents, supported_ents = select_summary_sentences_and_entities(
        nlp, example["text"], example["summary"]
    )
    example["supported_summary_sentences"] = supported_sents
    example["supported_summary_entities"] = [ent.text for ent in supported_ents]

    return example


def write_index(example, index):
    example["id"] = index
    return example


def build_metadata_dict(dataset) -> dict:
    return {
        example['id']: {
            "supported-summary-sentences": example['supported_summary_sentences'],
            "supported-summary-entities": example['supported_summary_entities'],
            "source": example['source'],
            "url": example['url'],
            "density_bin": example['density_bin'],
            "compression_bin": example['compression_bin'],
            "coverage_bin": example['coverage_bin'],
        }
        for example in dataset
    }


if __name__ == "__main__":
    newsroom_test = load_dataset("newsroom", split="test", data_dir="data/newsroom-raw")
    newsroom_test = newsroom_test.map(write_index, with_indices=True, num_proc=3)

    test_sample = newsroom_test.train_test_split(test_size=200, seed=42)["test"]
    test_sample = test_sample.map(mark_exclusion, num_proc=3)

    nlp = spacy.load("en_core_web_lg")
    test_sample_annotated = test_sample.map(
        write_filtered_summary_sentences_and_entities, num_proc=3
    )

    id2gold_summary = dict(
        zip(test_sample_annotated["id"], test_sample_annotated["summary"])
    )

    newsroom_gold_standard_config = {
        "Authors": "Grusky, Max and Naaman, Mor and Artzi, Yoav",
        "Link": "https://arxiv.org/abs/1804.11283",
        "Paper": "Newsroom: A Dataset of 1.3 Million Summaries with Diverse Extractive Strategies",
    }
    store_model_summaries(
        "newsroom",
        "gold",
        newsroom_gold_standard_config,
        id2gold_summary,
        build_metadata_dict(test_sample_annotated)
    )
