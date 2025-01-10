from collections import OrderedDict

from datasets import get_dataset_config_names
import numpy as np

BASE_DATASET = "wikipedia"
BASE_DATASET_LANG = "en"
BASE_DATASET_VERSION = "20220301"
BASE_CONFIG = f"{BASE_DATASET_VERSION}.{BASE_DATASET_LANG}"

QUERY_DATASET = "squad"


# TODO: add a check for the openai api key
def check_openai_api_key():
    pass


# TODO: add a check for the gcp credentials
def check_gcp_credentials():
    pass


# Check whether the dataset exists remotely
def check_dataset_exists_remote():
    configs = get_dataset_config_names(BASE_DATASET, trust_remote_code=True)
    if BASE_CONFIG in configs:
        return True
    else:
        return False


valid_names = ['text-embedding-ada-002',
               'text-embedding-3-small',
               'text-embedding-3-large',
               'textembedding-gecko',
               'text-multilingual-embedding',
               'text-embedding-004',
               'intfloat/e5-large-v2',
               'intfloat/e5-base-v2',
               'intfloat/e5-small-v2',
               'colbertv2.0',
               'nividia-nemo',
               'cohere/embed-english-v3.0',
               'cohere/embed-english-light-3.0',
               'jinaai/jina-embeddings-v2-small-en',
               'jinaai/jina-embeddings-v2-base-en',
               'voyage-3-large',
               'voyage-3-lite']


# Programmatically get the embedding size from the model name
# No need for manual input of the dimensions
def get_embedding_size(model_name: str, output_dimension_size=None):
    # OpenAI embedding models
    if model_name == 'text-embedding-ada-002' or model_name == 'text-embedding-3-small':
        default_model_dimension = 1536
    elif model_name == 'text-embedding-3-large':
        default_model_dimension = 3072
    # VertexAI embedding models
    elif model_name == 'textembedding-gecko' or model_name == 'text-multilingual-embedding' or model_name == 'text-embedding-004':
        default_model_dimension = 768
    # HuggingFace embedding models
    elif model_name == 'intfloat/e5-large-v2':
        default_model_dimension = 1024
    elif model_name == 'intfloat/e5-base-v2':
        default_model_dimension = 768
    elif model_name == 'intfloat/e5-small-v2':
        default_model_dimension = 384
    # Colbert models
    elif model_name == 'colbertv2.0':
        default_model_dimension = 128
    # Cohere models
    elif model_name == 'cohere/embed-english-v3.0':
        default_model_dimension = 1024
    elif model_name == 'cohere/embed-english-light-3.0':
        default_model_dimension = 384
    elif model_name == 'voyage-3-large':
        default_model_dimension = 1024
    elif model_name == 'voyage-3-lite':
        default_model_dimension = 512
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    if output_dimension_size is not None:
        if model_name == 'text-embedding-3-small' or model_name == 'text-embedding-3-large':
            assert (output_dimension_size <= default_model_dimension)
            return output_dimension_size
        elif model_name == 'voyage-3-large':
            assert (output_dimension_size in [256, 512, 1024, 2048])
            return output_dimension_size

    return default_model_dimension


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
