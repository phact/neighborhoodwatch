import os

import numpy as np
import pandas as pd
import pytest
import struct
import neighborhoodwatch.merge

def generate_test_files():
    indices_parquet_filename = 'indices0.parquet'
    distances_parquet_filename = 'distances0.parquet'

    num_rows = 10
    num_columns = 100  # Adjust based on your needs

    data = {f"{i}": np.random.randint(0, 100000, size=num_rows) for i in range(num_columns)}
    data["RowNum"] = [i for i in range(num_rows)]

    df_indices = pd.DataFrame(data)
    df_indices.to_parquet(indices_parquet_filename)

    dimensions = 1536  # Adjust based on your needs
    data = {f"embedding_{i}": np.random.rand(num_rows) for i in range(dimensions)}
    data["RowNum"] = [i for i in range(num_rows)]

    df = pd.DataFrame(data)
    df.to_parquet(distances_parquet_filename)

def test_merge():
    if 'distances0.parquet' not in os.listdir() or 'indices0.parquet' not in os.listdir():
        generate_test_files()
    neighborhoodwatch.merge.main()