from datasets import get_dataset_config_names

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
               'intfloat/e5-large-v2',
               'intfloat/e5-base-v2',
               'intfloat/e5-small-v2',
               'colbertv2.0']


# Programmatically get the embedding size from the model name
# No need for manual input of the dimensions
def get_embedding_size(model_name: str, reduced_dimension_size=None):
    # OpenAI embedding models
    if model_name == 'text-embedding-ada-002' or model_name == 'text-embedding-3-small':
        default_model_dimension = 1536
    elif model_name == 'text-embedding-3-large':
        default_model_dimension = 3072
    # VertexAI embedding models
    elif model_name == 'textembedding-gecko':
        default_model_dimension = 768
    # HuggingFace embedding models
    elif model_name == 'intfloat/e5-large-v2':
        default_model_dimension = 1024
    elif model_name == 'intfloat/e5-base-v2':
        default_model_dimension = 768
    elif model_name == 'intfloat/e5-small-v2':
        default_model_dimension = 384
    elif model_name == 'colbertv2.0':
        default_model_dimension = 128
    elif model_name == 'nvidia-nemo':
        default_model_dimension = 1024
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    # Reduced output dimension only applies to OpenAI latest text embedding models
    if model_name == 'text-embedding-3-small' or model_name == 'text-embedding-3-large':
        if reduced_dimension_size is not None:
            assert (reduced_dimension_size <= default_model_dimension)
            return min(default_model_dimension, reduced_dimension_size)
        else:
            return default_model_dimension
    else:
        return default_model_dimension


# def get_recommended_sentence_batch_size(model_name):
#     sentence_batch_size = 10000
#
#     if model_name == 'text-embedding-ada-002':
#         sentence_batch_size = 2046
#     elif model_name == 'text-embedding-3-small':
#         sentence_batch_size = 8191
#     elif model_name == 'text-embedding-3-large':
#         sentence_batch_size = 8191
#     elif model_name == 'textembedding-gecko':
#         sentence_batch_size = 3072
#     elif model_name == 'intfloat/e5-large-v2':
#         sentence_batch_size = 512
#     elif model_name == 'intfloat/e5-base-v2':
#         sentence_batch_size = 512
#     elif model_name == 'intfloat/e5-small-v2':
#         sentence_batch_size = 512
#     elif model_name == 'nvidia-nemo':
#         sentence_batch_size = 512
#     else:
#         raise ValueError(f"Unsupported model_name: {model_name}")
#
#     return sentence_batch_size


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