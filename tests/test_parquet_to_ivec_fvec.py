import os

import numpy as np
import pandas as pd
import pytest
from neighborhoodwatch.parquet_to_format import generate_query_vectors_fvec, generate_indices_ivec, \
    generate_base_vectors_fvec, generate_distances_fvec, count_vectors, get_nth_vector, dot_product
import pyarrow.parquet as pq

def generate_test_files():
    indices_parquet_filename = 'indices.parquet'
    query_vectors_parquet_filename = 'query_vectors.parquet'
    base_vectors_parquet_filename = 'base_vectors.parquet'

    num_rows = 10
    num_columns = 100

    data = {
        f"col_{i}": np.random.randint(0, 100000, size=num_rows) for i in range(num_columns)
    }
    df_indices = pd.DataFrame(data)
    df_indices.to_parquet(indices_parquet_filename)

    dimensions = 1536

    data = {
        "document_id_idx": [f"{i}_0" for i in range(num_rows)],
        "text": ["Sample text" for _ in range(num_rows)],
        "title": ["Sample title" for _ in range(num_rows)],
        "document_id": np.random.randint(0, 100, size=num_rows)
    }
    for i in range(dimensions):
        data[f"embedding_{i}"] = np.random.rand(num_rows)

    df = pd.DataFrame(data)
    df.to_parquet(base_vectors_parquet_filename)

    data = {
        "document_id_idx": [f"{i}_0" for i in range(num_rows)],
        "text": ["Sample text" for _ in range(num_rows)],
        "title": ["Sample title" for _ in range(num_rows)],
        "document_id": np.random.randint(0, 100, size=num_rows)
    }
    for i in range(dimensions):
        data[f"embedding_{i}"] = np.random.rand(num_rows)

    df = pd.DataFrame(data)
    df.to_parquet(query_vectors_parquet_filename)


def compute_similarities(query_vector, filename, indexes):
    last_similarity = 1
    failed_count = 0
    for index in indexes:
        base_vector = get_nth_vector(filename, index)
        similarity = dot_product(query_vector, base_vector)
        try:
            assert (last_similarity >= similarity), f"For index {index}: Last similarity {last_similarity}, similarity {similarity}"
        except AssertionError as e:
            failed_count += 1
            print(f"Assertion failed: {e} off by {similarity - last_similarity}")

        last_similarity = similarity
    print(f"Failed count {failed_count} out of {len(indexes)}")


def test_generate_query_vectors_fvec():
    input_parquet = 'query_vectors.parquet'
    if input_parquet not in os.listdir():
        generate_test_files()
    # TODO derive these
    base_count = 10
    query_count = 10
    dimensions = 1536
    filename = generate_query_vectors_fvec(input_parquet, base_count, query_count, dimensions)
    assert filename == 'ada_002_10_query_vectors_10.fvec'
    assert count_vectors(filename) == query_count

def test_generate_indices_ivec():
    input_parquet = 'final_indices.parquet'
    if input_parquet not in os.listdir():
        generate_test_files()
    base_count = 10
    query_count = 10
    k = 3
    filename = generate_indices_ivec(input_parquet, base_count, query_count, k)
    assert filename == 'ada_002_10_indices_query_10.ivec'
    assert count_vectors(filename) == base_count

def test_generate_distances_fvec():
    input_parquet = '../final_distances.parquet'
    if input_parquet not in os.listdir():
        generate_test_files()
    base_count = 1000
    query_count = 1000
    k = 3
    filename = generate_distances_fvec(input_parquet, base_count, query_count, k)
    assert filename == 'ada_002_10_indices_query_10.ivec'
    assert count_vectors(filename) == base_count


def test_generate_base_vectors_fvec():
    input_parquet = 'base_vectors.parquet'
    if input_parquet not in os.listdir():
        generate_test_files()
    base_count = 10
    dimensions = 1536
    filename = generate_base_vectors_fvec(input_parquet, base_count, dimensions)
    assert filename == 'ada_002_10_base_vectors.fvec'
    assert count_vectors(filename) == base_count

    table = pq.read_table(input_parquet)
    for i in range(len(table)):
        vec_vector = get_nth_vector(filename,i)
        #assert len(vec_vector) == len(parquet_vector), "expected {} but got {}".format(len(vec_vector), len(parquet_vector))
        for j in range(len(vec_vector)):
            vec_value = vec_vector[j]
            parquet_vector = table.column(f'embedding_{j}')
            parquet_value = parquet_vector[i].as_py()
            assert vec_value == parquet_value, "expected {} but got {}".format(parquet_value, vec_value)


def test_similarity():
    test_generate_indices_ivec()
    test_generate_base_vectors_fvec()
    test_generate_query_vectors_fvec()
    ivec_index_filename = 'ada_002_10_indices_query_10.ivec'
    fvec_query_vector_filename = 'ada_002_10_query_vectors_10.fvec'
    fvec_base_vector_filename = 'ada_002_10_base_vectors.fvec'

    assert count_vectors(ivec_index_filename) == count_vectors(fvec_query_vector_filename)
    for n in range(count_vectors(fvec_query_vector_filename)):
        nth_query_vector = get_nth_vector(fvec_query_vector_filename, n)
        first_indexes = get_nth_vector(ivec_index_filename, n)
        compute_similarities(nth_query_vector, fvec_base_vector_filename, first_indexes)

def test_actual_similarity():
    test_generate_indices_ivec()
    test_generate_base_vectors_fvec()
    test_generate_query_vectors_fvec()
    ivec_index_filename = '../wikipedia_squad/1k/pages_ada_002_1000_indices_query_1000.ivec'
    fvec_query_vector_filename = '../wikipedia_squad/1k/pages_ada_002_1000_query_vectors_1000.fvec'
    fvec_base_vector_filename = '../wikipedia_squad/1k/pages_ada_002_1000_base_vectors.fvec'

    assert count_vectors(ivec_index_filename) == count_vectors(fvec_query_vector_filename)
    for n in range(count_vectors(fvec_query_vector_filename)):
        nth_query_vector = get_nth_vector(fvec_query_vector_filename, n)
        first_indexes = get_nth_vector(ivec_index_filename, n)
        compute_similarities(nth_query_vector, fvec_base_vector_filename, first_indexes)
