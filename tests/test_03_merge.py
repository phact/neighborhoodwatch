import os
import numpy as np
import pandas as pd

from neighborhoodwatch.merge import merge_indices_and_distances

import tests.conftest as test_settings


def generate_test_files(indices_parquet_filename, 
                        distances_parquet_filename,
                        indices_count,
                        k):
    data = {f"{i}": np.random.randint(0, 100000, size=indices_count) for i in range(k)}
    data["RowNum"] = [i for i in range(indices_count)]

    df_indices = pd.DataFrame(data)
    df_indices.to_parquet(indices_parquet_filename)

    data = {f"embedding_{i}": np.random.rand(indices_count) for i in range(k)}
    data["RowNum"] = [i for i in range(indices_count)]

    df = pd.DataFrame(data)
    df.to_parquet(distances_parquet_filename)


def test_merge():
    indices_parquet_filename = 'indices0.parquet'
    distances_parquet_filename = 'distances0.parquet'
    
    dummy_model_name = 'test_03'
    indices_count = 1000
    dimensions = 384
    k = 10

    if indices_parquet_filename not in os.listdir(test_settings.test_dataset_dir) or \
       distances_parquet_filename not in os.listdir(test_settings.test_dataset_dir):
        generate_test_files(f"{test_settings.test_dataset_dir}/{dummy_model_name}_{dimensions}_{indices_parquet_filename}",
                            f"{test_settings.test_dataset_dir}/{dummy_model_name}_{dimensions}_{distances_parquet_filename}",
                            indices_count,
                            k)
    
    merge_indices_and_distances(test_settings.test_dataset_dir, 
                                dummy_model_name, 
                                k)