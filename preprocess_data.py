from datasets import load_dataset
import re

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
    xsum_test = load_dataset('xsum')['test']

    good_examples = []
    for example in xsum_test:
        good_examples.append
