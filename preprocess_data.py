from datasets import load_dataset
import re
import spacy

def mark_exclusion(example):
    example["valid_summary"] = not should_exclude_summary(example['text'])
    return example

def should_exclude_summary(summary: str) -> bool:
    """
    Exclude summary iff number of words is less than 8 OR it matches
    particular regexs.
    """
    if len(summary.split()) < 8:
        return True

    match = re.search(re.escape('on FoxNews.com'), summary, re.IGNORECASE) or \
        re.search(re.escape('from FoxNews.com'), summary, re.IGNORECASE) or \
        re.search(re.escape('Collection of all USATODAY.com'), summary, re.IGNORECASE) or \
        re.search(re.escape('washingtonpost.com'), summary, re.IGNORECASE)

    if match:
        return True
    else:
        return False


if __name__ == "__main__":
    newsroom_test = load_dataset('newsroom', split='test', data_dir="data/newsroom")
    newsroom_test_marked = newsroom_test.map(mark_exclusion, num_proc=3)
