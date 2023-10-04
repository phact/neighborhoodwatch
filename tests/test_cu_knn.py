import os

import numpy as np
import pandas as pd
import pytest
import struct
import neighborhoodwatch.cu_knn

def normalize_vector(row):
    vector = row.values
    norm = np.linalg.norm(vector)
    if norm == 0:  # prevent division by zero
        return vector
    return vector / norm
def generate_test_files():
    query_vectors_parquet_filename = 'query_vectors.parquet'
    base_vectors_parquet_filename = 'base_vectors.parquet'

    num_rows = 10
    dimensions = 1536  # Adjust based on your needs

    data = {
        "document_id_idx": [f"{i}_0" for i in range(num_rows)],
        "text": ["Sample text" for _ in range(num_rows)],
        "title": ["Sample title" for _ in range(num_rows)],
        "document_id": np.random.randint(0, 100, size=num_rows)
    }
    for i in range(dimensions):
        data[f"embedding_{i}"] = np.random.rand(num_rows).astype(np.float32)

    df = pd.DataFrame(data)

    vector_cols = [f"embedding_{i}" for i in range(dimensions)]
    normalized_data = df[vector_cols].apply(normalize_vector, axis=1, result_type='expand')
    df[vector_cols] = normalized_data

    tolerance = 1e-7
    norms = df[vector_cols].apply(np.linalg.norm, axis=1)
    for norm in norms:
        assert abs(norm - 1) < tolerance, "Vector is not normalized!"

    df.to_parquet(base_vectors_parquet_filename)

    data = {
        "document_id_idx": [f"{i}_0" for i in range(num_rows)],
        "text": ["Sample text" for _ in range(num_rows)],
        "title": ["Sample title" for _ in range(num_rows)],
        "document_id": np.random.randint(0, 100, size=num_rows)
    }
    for i in range(dimensions):
        data[f"embedding_{i}"] = np.random.rand(num_rows).astype(np.float32)

    df = pd.DataFrame(data)

    vector_cols = [f"embedding_{i}" for i in range(dimensions)]
    normalized_data = df[vector_cols].apply(normalize_vector, axis=1, result_type='expand')
    df[vector_cols] = normalized_data

    df.to_parquet(query_vectors_parquet_filename)

def test_cu_knn():
    query_filename = 'query_vectors.parquet'
    sorted_data_filename = 'base_vectors.parquet'
    if query_filename not in os.listdir() or sorted_data_filename not in os.listdir():
        generate_test_files()

    query_count = 10
    base_count = 10
    dimensions = 1536
    k = 2
    neighborhoodwatch.cu_knn.main(query_filename, query_count, sorted_data_filename, base_count, dimensions, False, k)