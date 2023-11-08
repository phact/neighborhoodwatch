import urllib.request
import transformers
from datasets import get_dataset_config_names


BASE_DATASET = "wikipedia"
BASE_DATASET_LANG = "en"
BASE_DATASET_VERSION = "20220301"
BASE_CONFIG=f"{BASE_DATASET_VERSION}.{BASE_DATASET_LANG}"

QUERY_DATASET = "squad"


# TODO: add a check for the openai api key
def check_openai_api_key():
    pass


# TODO: add a check for the gcp credentials
def check_gcp_credentials():
    pass


# Check whether the dataset exists remotely
def check_dataset_exists_remote():
    configs = get_dataset_config_names(BASE_DATASET)
    if BASE_CONFIG in configs:
        return True
    else:
        return False
    

# Programtically get the embedding size from the model name
# No need for manual input of the dimensions
def get_embedding_size(model_name):
    model = transformers.AutoModel.from_pretrained(model_name)
    return model.config.hidden_size


def get_full_filename(data_dir, filename):
    full_filename = filename
    if not filename.startswith(data_dir):
        full_filename = f'{data_dir}/{filename}'
    return full_filename


def get_model_prefix(model_name):
    if model_name:
        model_prefix = model_name.replace("/", "_")
    else:
        model_prefix = "ada_002"
    return model_prefix