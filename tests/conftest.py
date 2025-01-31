import os
import numpy as np
import pytest

from neighborhoodwatch.model_generator import get_effective_input_dimension_size
from neighborhoodwatch.nw_utils import setup_model_output_folder, get_source_query_dataset_filename, \
    get_source_base_dataset_filename

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
test_query_count = 100
test_base_count = 1000
test_k = 10
test_output_dimension = None
test_output_dtype = None

##
#  TBD: add more test cases with 1 for each model
test_openai = False
test_gcp_gecko = False
test_huggingface = True

if test_openai:
    test_model_name = 'text-embedding-3-large'
elif test_gcp_gecko:
    test_model_name = 'intfloat/e5-base-v2'
else:
    test_model_name = 'intfloat/e5-base-v2'

test_model_prefix = test_model_name.replace("/", "_")
test_dimensions = get_effective_input_dimension_size(model_name=test_model_name, target_dimension_size=output_dimension)
test_model_output_dir = setup_model_output_folder(test_dataset_dir, model_name, query_count, base_count, k)

test_query_vector_filename = get_source_query_dataset_filename(model_output_dir,
                                                                 model_name,
                                                                 query_count,
                                                                 output_dimension,
                                                                 output_dtype)

test_base_vector_filename = get_source_base_dataset_filename(model_output_dir,
                                                               model_name,
                                                               base_count,
                                                               output_dimension,
                                                               output_dtype)