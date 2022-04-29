from datasets import load_dataset


def load_xsum_dict(split):
    return {x["id"]: x for x in load_dataset("xsum")[split]}
