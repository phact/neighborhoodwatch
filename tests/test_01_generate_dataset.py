from neighborhoodwatch.generate_dataset import \
    generate_base_dataset, generate_query_dataset, get_embeddings_from_map

import tests.conftest as test_settings


def test_generate_query_dataset():
     generate_query_dataset(test_settings.test_dataset_dir, 
                            test_settings.query_count, 
                            test_settings.model_name)


def test_generate_base_dataset():
    generate_base_dataset(test_settings.test_dataset_dir, 
                          f"{test_settings.model_prefix}_query_vector_data_{test_settings.query_count}.parquet", 
                          test_settings.base_count, 
                          test_settings.model_name)


def test_get_embeddings_from_map():
    response = get_embeddings_from_map([(0,['hi there', 'how are you']),(1,['Im good', 'and you', 'sup'])], None)
    print(response)