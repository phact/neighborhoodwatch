import os
import numpy as np
import pytest

##
# Target dataset directory for testing purpose
# NOTE: DO NOT MAKE it the same as the production dataset directory
##   
test_dataset_dir = 'knn_dataset_test'
if not os.path.exists(test_dataset_dir):
    os.makedirs(test_dataset_dir)

##
# Tunable parameters for testing purpose
# Adjust based on your needs
## 
query_count = 100
base_count = 1000
k = 3

##
# TODO: Improve the model testing logic to support
#       multiple models at the same time
##
# This requires OpenAI API key
test_openai = False
openai_model_name = 'text-embedding-3-large'
openai_dimensions = 1536

# This requiers GCP access token
test_gcp_gecko = False
gecko_model_name = 'textembedding-gecko'
geco_dimensions = 768

test_huggingface = True
hf_e5s_model_name = 'intfloat/e5-base-v2'
hf_e5s_dimensions = 768

if test_openai:
    model_name = openai_model_name
    dimensions = openai_dimensions
elif test_gcp_gecko:
    model_name = gecko_model_name
    dimensions = geco_dimensions
elif test_huggingface:
    model_name = hf_e5s_model_name
    dimensions = hf_e5s_dimensions


model_prefix = model_name.replace("/", "_")


def get_final_indices_filename(model_name, dimensions):
    return f'{test_dataset_dir}/{model_name}_{dimensions}_final_indices.parquet'


def get_final_distances_filename(model_name, dimensions):
    return f'{test_dataset_dir}/{model_name}_{dimensions}_final_distances.parquet'