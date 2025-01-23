from collections import OrderedDict

import numpy as np
from datasets import get_dataset_config_names

BASE_DATASET = "wikipedia"
BASE_DATASET_LANG = "en"
BASE_DATASET_VERSION = "20220301"
BASE_CONFIG = f"{BASE_DATASET_VERSION}.{BASE_DATASET_LANG}"

QUERY_DATASET = "squad"


# Check whether the dataset exists remotely
def check_dataset_exists_remote():
    configs = get_dataset_config_names(BASE_DATASET, trust_remote_code=True)
    if BASE_CONFIG in configs:
        return True
    else:
        return False


def get_full_filename(data_dir, filename):
    full_filename = filename
    if not filename.startswith(data_dir):
        full_filename = f'{data_dir}/{filename}'
    return full_filename


def get_model_prefix(model_name):
    if model_name:
        model_prefix = model_name.replace("/", "_")
    else:
        model_prefix = "text-embedding-ada-002"
    return model_prefix


def remove_duplicate_embeddings(source_array):
    cnt1 = len(source_array)
    ## don't maintain the original order
    # unique_array = list(set(map(tuple, source_array)))
    ## maintain the original order
    unique_array = list(OrderedDict.fromkeys(map(tuple, source_array)))
    cnt2 = len(unique_array)

    return unique_array, cnt1 - cnt2,


def is_zero_embedding(embedding):
    return not np.any(embedding)


def normalize_vector(vector):
    assert not is_zero_embedding(vector), "Zero vector found!"
    norm = np.linalg.norm(vector)
    return (vector / norm).astype(np.float32)