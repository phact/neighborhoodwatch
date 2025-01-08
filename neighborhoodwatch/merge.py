import math

import cudf
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import glob
import re
import numpy as np

from tqdm import tqdm

from neighborhoodwatch.nw_utils import *


def get_file_count(data_dir, model_prefix, dimensions):
    str_pattern = r"" + data_dir + "/" + model_prefix + "_" + str(dimensions) + "_.+indices.+(\d+)\.parquet"
    pattern = re.compile(str_pattern)
    file_count = 0

    files = sorted(glob.glob(f"{data_dir}/{model_prefix}_{dimensions}*_indices*.parquet"))
    for filename in files:
        match = pattern.match(filename)
        if match:
            i = int(match.group(1))
            file_count = file_count + 1
    
    return file_count - 1


def read_ifvec_parquet_with_proper_schema(filename):
    ifvec_table = pq.read_table(filename)
    rownum_index = ifvec_table.schema.get_field_index('RowNum')
    if rownum_index != -1:
        ifvec_table.remove_column(rownum_index)

    return ifvec_table


def merge_indices_and_distances(data_dir,
                                model_prefix,
                                input_dimension,
                                final_indices_filename,
                                final_distances_filename,
                                k=100,
                                normalize_embed=False):
    file_count = get_file_count(data_dir, model_prefix, input_dimension)
    if file_count > 0:
        indices_table = read_ifvec_parquet_with_proper_schema(
            f"{data_dir}/{model_prefix}_{input_dimension}_final_indices_query_token100000_k100_0.parquet" if not normalize_embed else
            f"{data_dir}/{model_prefix}_{input_dimension}_normalized_indices0.parquet")
        distances_table = read_ifvec_parquet_with_proper_schema(
            f"{data_dir}/{model_prefix}_{input_dimension}_final_distances_query_token100000_k100_0.parquet" if not normalize_embed else
            f"{data_dir}/{model_prefix}_{input_dimension}_normalized_distances0.parquet")

        batch_size = min(10000000, len(indices_table))
        batch_count = math.ceil(len(indices_table) / batch_size)

        final_indices_writer = pq.ParquetWriter(final_indices_filename, indices_table.schema)
        final_distances_writer = pq.ParquetWriter(final_distances_filename, distances_table.schema)

        for start in tqdm(range(0, batch_count)):

            final_indices = pd.DataFrame()
            final_distances = pd.DataFrame()

            for i in tqdm(range(file_count)):
                indices_table = read_ifvec_parquet_with_proper_schema(
                    f"{data_dir}/{model_prefix}_{input_dimension}_final_indices_query_token100000_k100_{i}.parquet" if not normalize_embed else
                    f"{data_dir}/{model_prefix}_{input_dimension}_normalized_indices{i}.parquet")
                distances_table = read_ifvec_parquet_with_proper_schema(
                    f"{data_dir}/{model_prefix}_{input_dimension}_final_distances_query_token100000_k100_{i}.parquet" if not normalize_embed else
                    f"{data_dir}/{model_prefix}_{input_dimension}_normalized_distances{i}.parquet")

                if start != batch_count:
                    indices_batch = indices_table.slice(start, batch_size).to_pandas()
                    distances_batch = distances_table.slice(start, batch_size).to_pandas()
                else:
                    indices_batch = indices_table.slice(start, len(indices_table) - start * batch_size)
                    distances_batch = distances_table.slice(start, len(distances_table) - start * batch_size)

                if final_indices.empty & final_distances.empty:
                    final_indices = indices_batch
                    final_distances = distances_batch
                else:
                    # Concatenate the distances and indices along the column axis (axis=1)
                    concatenated_distances = pd.concat([final_distances, distances_batch], axis=1)
                    concatenated_indices = pd.concat([final_indices, indices_batch], axis=1)

                    # Convert the DataFrame to numpy arrays for argsort
                    concatenated_distances_np = concatenated_distances.values
                    concatenated_indices_np = concatenated_indices.values

                    # Get the sorted indices for each row in the concatenated distances DataFrame
                    sorted_indices_np = np.argsort(concatenated_distances_np, axis=1)

                    #rows = np.arange(sorted_indices_np.shape[0])[:, None]
                    #sorted_distances_np = concatenated_distances_np[rows, sorted_indices_np]
                    #sorted_indices_np = concatenated_indices_np[rows, sorted_indices_np]

                    # Using broadcasting to get the sorted distances and indices
                    sorted_distances_np = np.take_along_axis(concatenated_distances_np, sorted_indices_np, axis=1)
                    sorted_indices_np = np.take_along_axis(concatenated_indices_np, sorted_indices_np, axis=1)

                    # Convert the numpy arrays back to DataFrames only when necessary
                    sorted_distances = pd.DataFrame(sorted_distances_np,
                                                    index=concatenated_distances.index,
                                                    columns=concatenated_distances.columns)
                    sorted_indices = pd.DataFrame(sorted_indices_np,
                                                  index=concatenated_indices.index,
                                                  columns=concatenated_indices.columns)

                    # Select the top K distances and corresponding indices for each row
                    final_distances = sorted_distances.iloc[:, :k]
                    final_indices = sorted_indices.iloc[:, :k]

                    # Ensure the final distances are sorted in ascending order for each row
                    assert (final_distances.apply(lambda row: row.is_monotonic_increasing, axis=1).all())

            # .copy() is required for https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
            final_distances_copy = final_distances.copy()
            final_indices_copy = final_indices.copy()

            final_indices_writer.write_table(pa.Table.from_pandas(final_indices_copy))
            final_distances_writer.write_table(pa.Table.from_pandas(final_distances_copy))

        final_indices_writer.close()
        final_distances_writer.close()
