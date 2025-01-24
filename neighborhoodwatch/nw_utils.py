import os
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


def get_model_data_homedir(output_homedir, model_name, query_count, base_count, k):
    model_prefix = get_model_prefix(model_name)
    return f"{output_homedir}/{model_prefix}/q{query_count}_b{base_count}_k{k}"


def setup_model_output_folder(output_homedir, model_name, query_count, base_count, k):
    data_dir = get_model_data_homedir(output_homedir, model_name, query_count, base_count, k)
    partial_data_dir = f"{data_dir}/partial"
    if not os.path.exists(partial_data_dir):
        os.makedirs(partial_data_dir)

    return data_dir


def get_source_query_dataset_filename(knn_model_data_homedir, model_name, row_count, output_dimension=None, output_dtype=None):
    if output_dtype is not None:
        filename_base = f"{model_name.replace('/', '_')}_{output_dimension}_{output_dtype}_query_vector_data_{row_count}"
    else:
        filename_base = f"{model_name.replace('/', '_')}_{output_dimension}_query_vector_data_{row_count}"

    return f'{knn_model_data_homedir}/{filename_base}.parquet'


def get_source_base_dataset_filename(knn_model_data_homedir, model_name, row_count, output_dimension=None, output_dtype=None):
        if output_dtype is not None:
            filename_base = f"{model_name.replace('/', '_')}_{output_dimension}_{output_dtype}_base_vector_data_{row_count}"
        else:
            filename_base = f"{model_name.replace('/', '_')}_{output_dimension}_base_vector_data_{row_count}"

        return f'{knn_model_data_homedir}/{filename_base}.parquet'


def get_partial_indices_filename(knn_model_data_homedir: str, partial_set_cnt: int):
    is_final = partial_set_cnt == -1
    if is_final:
        return f'{knn_model_data_homedir}/partial/final_indices.parquet'
    else:
        return f'{knn_model_data_homedir}/partial/indices{partial_set_cnt}.parquet'


def get_partial_distances_filename(knn_model_data_homedir, partial_set_cnt:int):
    is_final = partial_set_cnt == -1
    if is_final:
        return f'{knn_model_data_homedir}/partial/final_distances.parquet'
    else:
        return f'{knn_model_data_homedir}/partial/distances{partial_set_cnt}.parquet'