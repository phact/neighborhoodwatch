import tests.conftest as test_settings

from neighborhoodwatch.generate_dataset import generate_query_dataset, generate_base_dataset, get_embeddings_from_map
from neighborhoodwatch.model_generator import get_embedding_generator_for_model


def test_generate_query_dataset():
    filename = generate_query_dataset(test_settings.test_dataset_dir,
                                      test_settings.model_name,
                                      test_settings.query_count,
                                      test_settings.dimensions)
    print(f"query_dataset_file: {filename}")


def test_generate_base_dataset():
    filename = generate_base_dataset(test_settings.test_dataset_dir,
                                     test_settings.model_name,
                                     test_settings.source_query_vector_filename,
                                     test_settings.base_count,
                                     test_settings.dimensions)
    print(f"base_dataset_file: {filename}")


def test_get_embeddings_from_map():
    generator = get_embedding_generator_for_model(model_name=test_settings.model_name,
                                                  output_dimension=test_settings.dimensions)

    response, zero_cnt = get_embeddings_from_map([(0, ['hi there', 'how are you']), (1, ['Im good', 'and you', 'sup'])],
                                                 generator)
    print(f"response={response}\nzero_embedding_count={zero_cnt}")
