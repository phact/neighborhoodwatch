import os

import numpy as np
import pandas as pd
import pytest
import struct
from neighborhoodwatch.parquet_to_ivec_fvec import generate_query_vectors_fvec, generate_indices_ivec, generate_base_vectors_fvec

def generate_test_files():
    indices_parquet_filename = 'indices.parquet'
    query_vectors_parquet_filename = 'query_vectors.parquet'
    base_vectors_parquet_filename = 'base_vectors.parquet'

    num_rows = 10
    num_columns = 100  # Adjust based on your needs

    data = {
        f"col_{i}": np.random.randint(0, 100000, size=num_rows) for i in range(num_columns)
    }
    df_indices = pd.DataFrame(data)
    df_indices.to_parquet(indices_parquet_filename)

    dimensions = 1536  # Adjust based on your needs

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

def count_vectors(filename):
    with open(filename, 'rb') as f:
        count = 0
        while True:
            dim = f.read(4)
            if not dim:
                break
            dim = struct.unpack('i', dim)[0]
            f.read(4 * dim)
            count += 1
        return count

def get_first(filename):
    return get_nth(filename, 0)

def get_nth(filename, n):
    format_char = 'f'
    if filename.endswith("ivec"):
        format_char = 'i'
    with open(filename, 'rb') as f:
        dimension = struct.unpack('i', f.read(4))[0]
        f.seek(4 * n * (dimension+1), 1)

        vector = struct.unpack(format_char * dimension, f.read(4 * dimension))

    return vector

def dot_product(A, B):
    return sum(a * b for a, b in zip(A, B))

def compute_similarities(query_vector, filename, indexes):
    last_similarity = 100000
    for index in indexes:
        base_vector = get_nth(filename, index)
        similarity = dot_product(query_vector, base_vector)
        try:
            assert (last_similarity >= similarity), f"For index {index}: Last similarity {last_similarity}, similarity {similarity}"
        except AssertionError as e:
            print(f"Assertion failed: {e}")

        last_similarity = similarity


def test_generate_query_vectors_fvec():
    input_parquet = 'query_vectors.parquet'
    if input_parquet not in os.listdir():
        generate_test_files()
    base_count = 10
    query_count = 5
    filename = generate_query_vectors_fvec(input_parquet, base_count, query_count)
    assert filename == 'pages_ada_002_10_query_vectors_5.fvec'
    assert count_vectors(filename) == query_count

def test_generate_indices_ivec():
    input_parquet = 'final_indices.parquet'
    if input_parquet not in os.listdir():
        generate_test_files()
    base_count = 10
    query_count = 5
    filename = generate_indices_ivec(input_parquet, base_count, query_count)
    assert filename == 'pages_ada_002_10_indices_query_5.ivec'
    assert count_vectors(filename) == base_count


def test_generate_base_vectors_fvec():
    input_parquet = 'base_vectors.parquet'
    if input_parquet not in os.listdir():
        generate_test_files()
    base_count = 10
    query_count = 5
    filename = generate_base_vectors_fvec(input_parquet, base_count, query_count)
    assert filename == 'pages_ada_002_10_base_vectors.fvec'
    assert count_vectors(filename) == base_count

def test_similarity():
    test_generate_indices_ivec()
    test_generate_base_vectors_fvec()
    test_generate_query_vectors_fvec()
    ivec_index_filename = 'pages_ada_002_10_indices_query_5.ivec'
    fvec_query_vector_filename = 'pages_ada_002_10_query_vectors_5.fvec'
    fvec_base_vector_filename = 'pages_ada_002_10_base_vectors.fvec'

    for n in range(count_vectors(ivec_index_filename)):
        print(n)
        nth_query_vector = get_nth(fvec_query_vector_filename, n)
        first_indexes = get_nth(ivec_index_filename, n)
        compute_similarities(nth_query_vector, fvec_base_vector_filename, first_indexes)