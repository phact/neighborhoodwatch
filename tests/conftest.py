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
test_openai = True
openai_model_name = 'text-embedding-3-large'
openai_dimensions = 1536

# This requiers GCP access token
test_gcp_gecko = False
gecko_model_name = 'textembedding-gecko'
geco_dimensions = 384

test_huggingface = False
hf_e5s_model_name = 'intfloat/multilingual-e5-small'
hf_e5s_dimensions = 384


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
