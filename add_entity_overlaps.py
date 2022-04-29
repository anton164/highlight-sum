from sumtool.storage import get_summaries, store_summary_metrics
import argparse
from data_utils import load_xsum_dict
from preprocess.preprocess_data_xsum import nlp, select_summary_sentences_and_entities
from tqdm import tqdm


SUMTOOL_DATASET = "xsum"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        type=str
    )
    args = parser.parse_args()

    summaries = get_summaries(SUMTOOL_DATASET, args.model)
    xsum_test = load_xsum_dict("test")

    summary_metadata = {}

    for sum_id, data in tqdm(list(summaries.items())):
        _, supported_entities, missing_entities = select_summary_sentences_and_entities(
            nlp,
            xsum_test[sum_id]["document"],
            data["summary"]
        )

        summary_metadata[sum_id] = {
            "entities_in_source": [ent.text for ent in supported_entities],
            "entities_not_in_source": [ent.text for ent in missing_entities]
        }


    store_summary_metrics(
        SUMTOOL_DATASET,
        args.model,
        summary_metadata
    )
