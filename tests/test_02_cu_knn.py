import os
import numpy as np
import pandas as pd

from neighborhoodwatch.cu_knn import compute_knn
import tests.conftest as test_settings


def normalize_vector(row):
    vector = row.values
    norm = np.linalg.norm(vector)
    if norm == 0:  # prevent division by zero
        return vector
    return (vector / norm).astype(np.float32)


def generate_test_files(query_vectors_parquet_filename, 
                        base_vectors_parquet_filename, 
                        query_count,
                        base_count, 
                        dimensions):
    data = {
        "document_id_idx": [f"{i}_0" for i in range(base_count)],
        "text": ["Sample text" for _ in range(base_count)],
        "title": ["Sample title" for _ in range(base_count)],
        "document_id": np.random.randint(0, 100, size=base_count)
    }
    for i in range(dimensions):
        data[f"embedding_{i}"] = np.random.rand(base_count).astype(np.float32)

    df = pd.DataFrame(data)

    vector_cols = [f"embedding_{i}" for i in range(dimensions)]
    normalized_data = df[vector_cols].apply(normalize_vector, axis=1, result_type='expand')
    df[vector_cols] = normalized_data

    ##
    # Somehow 1e-7 is not enough for the test to always pass. It fails with the following error:
    # ----------------------------- Captured stderr call -----------------------------    
    # E           AssertionError: Vector is not normalized!
    # E           assert 1.1920928955078125e-07 < 1e-07
    # E            +  where 1.1920928955078125e-07 = abs((0.9999998807907104 - 1))
    ## 
    # tolerance = 1e-7
    tolerance = 2e-7
    norms = df[vector_cols].apply(np.linalg.norm, axis=1)
    for norm in norms:
        assert abs(norm - 1) < tolerance, "Vector is not normalized!"

    df.to_parquet(base_vectors_parquet_filename)

    data = {
        "document_id_idx": [f"{i}_0" for i in range(query_count)],
        "text": ["Sample text" for _ in range(query_count)],
        "title": ["Sample title" for _ in range(query_count)],
        "document_id": np.random.randint(0, 100, size=query_count)
    }
    for i in range(dimensions):
        data[f"embedding_{i}"] = np.random.rand(query_count).astype(np.float32)

    df = pd.DataFrame(data)

    vector_cols = [f"embedding_{i}" for i in range(dimensions)]
    normalized_data = df[vector_cols].apply(normalize_vector, axis=1, result_type='expand')
    df[vector_cols] = normalized_data

    df.to_parquet(query_vectors_parquet_filename)


def test_cu_knn():
    query_vector_filename = 'query_vectors.parquet'
    base_vector_filename = 'base_vectors.parquet'
    
    dummy_model_name = 'test_02'
    query_count = 100
    base_count = 1000
    dimensions = 384
    k = 10

    if f"{dummy_model_name}_{query_vector_filename}" not in os.listdir(test_settings.test_dataset_dir) or \
       f"{dummy_model_name}_{base_vector_filename}" not in os.listdir(test_settings.test_dataset_dir):
        generate_test_files(f"{test_settings.test_dataset_dir}/{dummy_model_name}_{query_vector_filename}",
                            f"{test_settings.test_dataset_dir}/{dummy_model_name}_{base_vector_filename}",
                            query_count,
                            base_count,
                            dimensions)
    
    compute_knn(test_settings.test_dataset_dir,
                dummy_model_name,
                dimensions,
                f"{dummy_model_name}_{query_vector_filename}",
                query_count,
                f"{dummy_model_name}_{base_vector_filename}",
                base_count,
                test_settings.get_final_indices_filename(dummy_model_name, dimensions),
                test_settings.get_final_distances_filename(dummy_model_name, dimensions),
                mem_tune=False,
                k=k)