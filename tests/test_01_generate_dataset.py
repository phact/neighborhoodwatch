import tests.conftest as test_settings

from neighborhoodwatch.generate_dataset import generate_query_dataset, generate_base_dataset, get_embeddings_from_map, \
    VertexAIEmbeddingGenerator, OpenAIEmbeddingGenerator, IntfloatE5EmbeddingGenerator


def test_generate_query_dataset():
    filename = generate_query_dataset(test_settings.test_dataset_dir,
                                      test_settings.model_name,
                                      test_settings.query_count,
                                      test_settings.dimensions)
    print(f"query_dataset_file: {filename}")


def test_generate_base_dataset():
    filename = generate_base_dataset(test_settings.test_dataset_dir,
                                     test_settings.model_name,
                                     f"{test_settings.model_prefix}_{test_settings.dimensions}_query_vector_data_{test_settings.query_count}.parquet",
                                     test_settings.base_count,
                                     test_settings.dimensions)
    print(f"base_dataset_file: {filename}")


def test_get_embeddings_from_map():
    if test_settings.model_name == 'textembedding-gecko':
        generator = VertexAIEmbeddingGenerator(model_name=test_settings.model_name)
    # OpenAI, older model (ada-002)
    elif test_settings.model_name == "text-embedding-ada-002":
        generator = OpenAIEmbeddingGenerator(model_name=test_settings.model_name)
    elif test_settings.model_name == "text-embedding-3-small" or test_settings.model_name == "text-embedding-3-large":
        generator = OpenAIEmbeddingGenerator(model_name=test_settings.model_name, output_dimension_size=test_settings.dimensions)
    # OpenAI, newer model (3-small, 3-large)
    elif test_settings.model_name == "text-embedding-3-small" or test_settings.model_name == "text-embedding-3-large":
        generator = OpenAIEmbeddingGenerator(model_name=test_settings.model_name, output_dimension_size=test_settings.dimensions)
    # Default to Huggingface mode e5-small-v2
    else:
        generator = IntfloatE5EmbeddingGenerator(model_name=test_settings.model_name)

    response, zero_cnt = get_embeddings_from_map([(0, ['hi there', 'how are you']), (1, ['Im good', 'and you', 'sup'])],
                                                 generator)
    print(f"response={response}\nzero_embedding_count={zero_cnt}")
