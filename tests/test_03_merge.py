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
    dummy_model_name = 'test_03'
    indices_count = 1000
    dimensions = 384
    k = 10

    indices_parquet_filename = f"{test_settings.test_dataset_dir}/{dummy_model_name}_{dimensions}_indices0.parquet"
    distances_parquet_filename = f"{test_settings.test_dataset_dir}/{dummy_model_name}_{dimensions}_distances0.parquet"

    if indices_parquet_filename not in os.listdir(test_settings.test_dataset_dir) or \
       distances_parquet_filename not in os.listdir(test_settings.test_dataset_dir):
        generate_test_files(indices_parquet_filename,
                            distances_parquet_filename,
                            indices_count,
                            k)
    
    merge_indices_and_distances(test_settings.test_dataset_dir, 
                                dummy_model_name,
                                dimensions,
                                test_settings.get_final_indices_filename(dummy_model_name, dimensions),
                                test_settings.get_final_distances_filename(dummy_model_name, dimensions),
                                k)