import os
import h5py
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from neighborhoodwatch.parquet_to_format import \
    generate_query_vectors_fvec, generate_indices_ivec, \
    generate_base_vectors_fvec, generate_distances_fvec, \
    count_vectors, get_nth_vector, dot_product, generate_hdf5_file, generate_ivec_fvec_files
from neighborhoodwatch.nw_utils import normalize_vector

import tests.conftest as test_settings


dummy_model_name = 'test_04'
query_count = 100
base_count = 1000
dimensions = 384
k = 10


def generate_test_files(query_count,
                        base_count,
                        dimensions,
                        k,
                        indices_parquet_filename = None, 
                        distances_parquet_filename = None, 
                        query_vectors_parquet_filename = None,
                        base_vectors_parquet_filename = None):
    if indices_parquet_filename is not None:
        data = {
            f"col_{i}": np.random.randint(0, 100000, size=base_count) for i in range(k)
        }
        df_indices = pd.DataFrame(data)
        vector_cols = [f"col_{i}" for i in range(k)]
        normalized_data = df_indices[vector_cols].apply(normalize_vector, axis=1, result_type='expand')
        df_indices[vector_cols] = normalized_data
        df_indices.to_parquet(indices_parquet_filename)

    if distances_parquet_filename is not None:
        data = {
            f"col_{i}": np.random.randint(0, 100000, size=base_count) for i in range(k)
        }
        df_distances = pd.DataFrame(data)
        vector_cols = [f"col_{i}" for i in range(k)]
        normalized_data = df_distances[vector_cols].apply(normalize_vector, axis=1, result_type='expand')
        df_distances[vector_cols] = normalized_data
        df_distances.to_parquet(distances_parquet_filename)

    if query_vectors_parquet_filename is not None:
        data = {
            "document_id_idx": [f"{i}_0" for i in range(query_count)],
            "text": ["Sample text" for _ in range(query_count)],
            "title": ["Sample title" for _ in range(query_count)],
            "document_id": np.random.randint(0, 100, size=query_count)
        }
        for i in range(dimensions):
            data[f"embedding_{i}"] = np.random.rand(query_count).astype(np.float32)

        df_query = pd.DataFrame(data)
        vector_cols = [f"embedding_{i}" for i in range(dimensions)]
        normalized_data = df_query[vector_cols].apply(normalize_vector, axis=1, result_type='expand')
        df_query[vector_cols] = normalized_data
        df_query.to_parquet(query_vectors_parquet_filename)

    if base_vectors_parquet_filename is not None:
        data = {
            "document_id_idx": [f"{i}_0" for i in range(base_count)],
            "text": ["Sample text" for _ in range(base_count)],
            "title": ["Sample title" for _ in range(base_count)],
            "document_id": np.random.randint(0, 100, size=base_count)
        }
        for i in range(dimensions):
            data[f"embedding_{i}"] =np.random.rand(base_count).astype(np.float32)

        df_base = pd.DataFrame(data)
        vector_cols = [f"embedding_{i}" for i in range(dimensions)]
        normalized_data = df_base[vector_cols].apply(normalize_vector, axis=1, result_type='expand')
        df_base[vector_cols] = normalized_data
        df_base.to_parquet(base_vectors_parquet_filename)


def test_generate_indices_ivec():
    indices_parquet_filename = f"{dummy_model_name}_{dimensions}_final_indices.parquet"

    if indices_parquet_filename not in os.listdir(test_settings.test_dataset_dir):
        generate_test_files(query_count,
                            base_count,
                            dimensions,
                            k,
                            indices_parquet_filename=f"{test_settings.test_dataset_dir}/{indices_parquet_filename}")
    
    filename = generate_indices_ivec(test_settings.test_dataset_dir, 
                                     f"{test_settings.test_dataset_dir}/{indices_parquet_filename}",
                                     base_count, 
                                     query_count, 
                                     k, 
                                     dummy_model_name,
                                     dimensions)
    
    assert filename == f"{test_settings.test_dataset_dir}/{dummy_model_name}_{dimensions}_indices_b{base_count}_q{query_count}_k{k}.ivec"
    assert count_vectors(test_settings.test_dataset_dir, filename) == base_count


def test_generate_distances_fvec():
    distances_parquet_filename = f"{dummy_model_name}_{dimensions}_final_distances.parquet"

    if distances_parquet_filename not in os.listdir(test_settings.test_dataset_dir):
        generate_test_files(query_count,
                            base_count,
                            dimensions,
                            k,
                            distances_parquet_filename=f"{test_settings.test_dataset_dir}/{distances_parquet_filename}")
    
    filename = generate_distances_fvec(test_settings.test_dataset_dir, 
                                       f"{test_settings.test_dataset_dir}/{distances_parquet_filename}",
                                       base_count, 
                                       query_count, 
                                       k, 
                                       dummy_model_name,
                                       dimensions)
    
    assert filename == f"{test_settings.test_dataset_dir}/{dummy_model_name}_{dimensions}_distances_b{base_count}_q{query_count}_k{k}.fvec"
    assert count_vectors(test_settings.test_dataset_dir, filename) == base_count


def test_generate_query_vectors_fvec():
    query_vectors_parquet_filename = f"{dummy_model_name}_{dimensions}_query_vectors.parquet"

    if query_vectors_parquet_filename not in os.listdir(test_settings.test_dataset_dir):
        generate_test_files(query_count,
                            base_count,
                            dimensions,
                            k,
                            query_vectors_parquet_filename=f"{test_settings.test_dataset_dir}/{query_vectors_parquet_filename}")
    
    filename, _ = generate_query_vectors_fvec(test_settings.test_dataset_dir,
                                               f"{test_settings.test_dataset_dir}/{query_vectors_parquet_filename}",
                                               query_count,
                                               dummy_model_name,
                                               dimensions)
    assert filename == f"{test_settings.test_dataset_dir}/{dummy_model_name}_{dimensions}_query_vectors_{query_count}.fvec"
    assert count_vectors(test_settings.test_dataset_dir, filename) == query_count


def test_generate_base_vectors_fvec():
    base_vectors_parquet_filename = f"{dummy_model_name}_{dimensions}_base_vectors.parquet"

    if base_vectors_parquet_filename not in os.listdir(test_settings.test_dataset_dir):
        generate_test_files(query_count,
                            base_count,
                            dimensions,
                            k,
                            base_vectors_parquet_filename=f"{test_settings.test_dataset_dir}/{base_vectors_parquet_filename}")

    filename, _ = generate_base_vectors_fvec(test_settings.test_dataset_dir,
                                              f"{test_settings.test_dataset_dir}/{base_vectors_parquet_filename}",
                                              base_count,
                                              dummy_model_name,
                                              dimensions)

    assert filename == f"{test_settings.test_dataset_dir}/{dummy_model_name}_{dimensions}_base_vectors_{base_count}.fvec"
    assert count_vectors(test_settings.test_dataset_dir, filename) == base_count

    table = pq.read_table(f"{test_settings.test_dataset_dir}/{base_vectors_parquet_filename}")
    for i in range(len(table)):
        vec_vector = get_nth_vector(test_settings.test_dataset_dir, filename, i)
        #assert len(vec_vector) == len(parquet_vector), "expected {} but got {}".format(len(vec_vector), len(parquet_vector))
        for j in range(len(vec_vector)):
            vec_value = vec_vector[j]
            parquet_vector = table.column(f'embedding_{j}')
            parquet_value = parquet_vector[i].as_py()
            assert vec_value == parquet_value, "expected {} but got {}".format(parquet_value, vec_value)


def test_generate_hdf5():
    test_generate_indices_ivec()
    test_generate_distances_fvec()
    test_generate_query_vectors_fvec()
    test_generate_base_vectors_fvec()

    query_vector_fvec, query_df_hdf5, base_vector_fvec, base_df_hdf5, indices_ivec, distances_fvec = \
        generate_ivec_fvec_files(test_settings.test_dataset_dir,
                                 dummy_model_name,
                                 dimensions,
                                 f"{test_settings.test_dataset_dir}/{dummy_model_name}_{dimensions}_base_vectors.parquet",
                                 f"{test_settings.test_dataset_dir}/{dummy_model_name}_{dimensions}_query_vectors.parquet",
                                 f"{test_settings.test_dataset_dir}/{dummy_model_name}_{dimensions}_final_indices.parquet",
                                 f"{test_settings.test_dataset_dir}/{dummy_model_name}_{dimensions}_final_distances.parquet",
                                 base_count,
                                 query_count,
                                 k)

    generate_hdf5_file(test_settings.test_dataset_dir,
                       dummy_model_name,
                       dimensions,
                       base_df_hdf5,
                       query_df_hdf5,
                       f"{test_settings.test_dataset_dir}/{dummy_model_name}_{dimensions}_final_indices.parquet",
                       f"{test_settings.test_dataset_dir}/{dummy_model_name}_{dimensions}_final_distances.parquet",
                       base_count,
                       query_count,
                       k)
    hdf5_filename = f"{dummy_model_name}_{dimensions}_base_{base_count}_query_{query_count}_k{k}.hdf5"
    assert os.path.exists(f"{test_settings.test_dataset_dir}/{hdf5_filename}")

    f = h5py.File(f"{test_settings.test_dataset_dir}/{hdf5_filename}", 'r')
    assert 'neighbors' in f
    assert 'distances' in f
    assert 'test' in f
    assert 'train' in f


def compute_similarities(dataset_dir, query_vector, filename, indexes):
    last_similarity = 1
    failed_count = 0

    for index in indexes:
        base_vector = get_nth_vector(dataset_dir, filename, index)
        similarity = dot_product(query_vector, base_vector)
        try:
            assert (last_similarity >= similarity), f"For index {index}: Last similarity {last_similarity}, similarity {similarity}"
        except AssertionError as e:
            failed_count += 1
            print(f"Assertion failed: {e} off by {similarity - last_similarity}")

        last_similarity = similarity
    
    print(f"Failed count {failed_count} out of {len(indexes)}")


def test_similarity():
    # test_generate_indices_ivec()
    # test_generate_query_vectors_fvec()
    # test_generate_base_vectors_fvec()

    ivec_index_filename = f"{dummy_model_name}_{dimensions}_indices_b{base_count}_q{query_count}_k{k}.ivec"
    fvec_query_vector_filename = f"{dummy_model_name}_{dimensions}_query_vectors_{query_count}.fvec"
    fvec_base_vector_filename = f"{dummy_model_name}_{dimensions}_query_vectors_{query_count}.fvec"

    for n in range(count_vectors(test_settings.test_dataset_dir, fvec_query_vector_filename)):
        nth_query_vector = get_nth_vector(test_settings.test_dataset_dir, fvec_query_vector_filename, n)
        first_indexes = get_nth_vector(test_settings.test_dataset_dir, ivec_index_filename, n)
        compute_similarities(test_settings.test_dataset_dir,
                             nth_query_vector,
                             fvec_base_vector_filename,
                             first_indexes)


def test_actual_similarity():
    # test_generate_indices_ivec()
    # test_generate_query_vectors_fvec()
    # test_generate_base_vectors_fvec()

    ##
    # NOTE: this requires the actual dataset to be generated first
    #       by running the following command:
    #          poetry run nw 100 1000 -k 100 -m 'intfloat/e5-base-v2' --data-dir knn_dataset
    #
    ivec_index_filename = f"intfloat_e5-base-v2_768_indices_b1000_q100_k100.ivec"
    fvec_query_vector_filename = f"intfloat_e5-base-v2_768_query_vectors_100.fvec"
    fvec_base_vector_filename = f"intfloat_e5-base-v2__768_base_vectors_1000.fvec"
    actual_dataset_dir = 'knn_dataset/intfloat_e5-base-v2/q100_b1000_k100'

    if os.path.exists(actual_dataset_dir) and \
       os.path.exists(f"{actual_dataset_dir}/{ivec_index_filename}") and \
       os.path.exists(f"{actual_dataset_dir}/{fvec_query_vector_filename}") and \
       os.path.exists(f"{actual_dataset_dir}/{fvec_base_vector_filename}"):
        for n in range(count_vectors(actual_dataset_dir, fvec_query_vector_filename)):
            nth_query_vector = get_nth_vector(actual_dataset_dir, fvec_query_vector_filename, n)
            first_indexes = get_nth_vector(actual_dataset_dir, ivec_index_filename, n)
            compute_similarities(actual_dataset_dir,
                                 nth_query_vector,
                                 fvec_base_vector_filename,
                                 first_indexes)
