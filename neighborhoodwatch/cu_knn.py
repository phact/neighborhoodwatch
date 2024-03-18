import cudf
import pyarrow.parquet as pq
import cupy as cp
from tqdm import tqdm
import pandas as pd
import gc
import math
import numpy as np
from pylibraft.neighbors.brute_force import knn
import rmm
from rich import print as rprint
from rich.markdown import Markdown

from neighborhoodwatch.nw_utils import *


##
# The main data processing task in this script is based on
# pyarrow.Table API
#
## 


def stream_cudf_to_parquet(df, chunk_size, filename):
    """
    Stream a cuDF DataFrame to a single Parquet file in chunks.

    Parameters:
    - df: cuDF DataFrame
    - chunk_size: Number of rows for each chunk
    - filename: Output Parquet file name
    """
    writer = None

    num_chunks = (len(df) + chunk_size - 1) // chunk_size

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size
        df_chunk = df.iloc[start_idx:end_idx]

        table_chunk = cudf.DataFrame.to_arrow(df_chunk)

        if writer is None:
            writer = pq.ParquetWriter(filename, table_chunk.schema)

        writer.write_table(table_chunk)

    writer.close()


def tune_memory(table, batch_size, max_memory_threshold, rmm):
    num_rows = len(table)
    rprint(Markdown(f"Tuning memory settings (num rows: {num_rows}; initial batch size: {batch_size}) ..."))

    # Only tune memory if the dataset is large enough
    batch = table.slice(0, batch_size)
    df = cudf.DataFrame.from_arrow(batch)

    import nvidia_smi
    nvidia_smi.nvmlInit()
    # card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)

    total_gpu_memory = nvidia_smi.nvmlDeviceGetMemoryInfo(handle).total

    print("-- total memory:", total_gpu_memory)

    nvidia_smi.nvmlShutdown()

    # Measure GPU memory usage after converting to cuDF dataframe
    memory_used = df.memory_usage().sum()
    print(f"-- memory_used {memory_used}")
    factor = math.ceil((total_gpu_memory * .2) / memory_used)
    print(f"-- factor {factor}")
    batch_size *= factor  # or any other increment factor you find suitable

    while True:
        try:
            if num_rows < batch_size:
                print(
                    f"-- the calculated batch size {batch_size} is bigger than total rows {num_rows}. Use total rows as the target batch size!")
                batch_size = num_rows
                break

            rmm.reinitialize(pool_allocator=False)
            batch = table.slice(0, batch_size)
            df = cudf.DataFrame.from_arrow(batch)

            # Measure GPU memory usage after converting to cuDF dataframe
            memory_used = df.memory_usage().sum()

            if memory_used > max_memory_threshold * total_gpu_memory:
                # If the memory used goes beyond the threshold, break and set the batch size
                # to the last successful size.
                batch_size = int(0.8 * batch_size)
                print(f"-- found threshold {batch_size}")
                break
            else:
                print(f"-- memory used {memory_used}, ratio {memory_used / total_gpu_memory}, batch_size {batch_size}")
                batch_size *= 1.2

        except Exception as e:
            batch_size = int(0.8 * batch_size)
            print(f"-- exception {e}, max batch size {batch_size}")
            break

    return batch_size


def load_table(data_dir, filename, start, end):
    # If needed, add "thrift_string_size_limit=1000000000" option to avoid the error of exeeding the thrift string size limit
    return pq.read_table(get_full_filename(data_dir, filename)).slice(start, end)


def drop_columns(table, keep_columns):
    columns_to_drop = list(set(table.schema.names) - set(keep_columns))
    for col in columns_to_drop:
        if col in table.schema.names:
            col_index = table.schema.get_field_index(col)
            table = table.remove_column(col_index)
    return table


def get_embedding_count(table):
    column_names = table.schema.names
    matching_columns = [name for name in column_names if name.startswith('embedding_')]
    return len(matching_columns)


def prep_table(data_dir, filename, count, input_dimension):
    table = load_table(data_dir, filename, 0, count)
    total_dimensions = sum('embedding_' in c for c in table.column_names)
    assert total_dimensions >= input_dimension and total_dimensions % input_dimension == 0, \
        f"Input dimension {input_dimension} does not match the actual dimension {total_dimensions} in the table."

    # Not sure if we really need to keep these 2 columns here
    # in the "process_batches" function, we only select the float columns as per the following code:
    # - df_numeric = df.select_dtypes(['float32', 'float64'])

    # column_names = ['text', 'document_id_idx']
    column_names = []
    for i in range(total_dimensions):
        column_names.append(f'embedding_{i}')

    return drop_columns(table, column_names)


def compute_knn(data_dir,
                input_dimensions,
                query_filename,
                query_count,
                base_filename,
                base_count,
                final_indecies_filename,
                final_distances_filename,
                mem_tune=False,
                k=100,
                initial_batch_size=100000,
                max_memory_threshold=0.1,
                split=True):
    rmm.mr.set_current_device_resource(rmm.mr.PoolMemoryResource(rmm.mr.ManagedMemoryResource()))

    batch_size = initial_batch_size
    # batch_size = 543680

    print(f"-- prepare query source table for brute-force KNN computation.")
    query_table = prep_table(data_dir, query_filename, query_count, input_dimensions)
    print(f"-- prepare base source table for brute-force KNN computation.")
    base_table = prep_table(data_dir, base_filename, base_count, input_dimensions)

    if mem_tune:
        batch_size = tune_memory(base_table, batch_size, max_memory_threshold, rmm)

    batch_count = math.ceil(len(base_table) / batch_size)
    assert (len(base_table) % batch_size == 0) or k <= (
            len(base_table) % batch_size), f"Cannot generate k of {k} with only {len(base_table)} rows and batch_size of {batch_size}."

    process_batches(final_indecies_filename,
                    final_distances_filename,
                    base_table,
                    query_table,
                    batch_count,
                    batch_size,
                    k,
                    split)


def cleanup(*args):
    for arg in args:
        try:
            del arg
        except:
            pass
    gc.collect()
    rmm.reinitialize(pool_allocator=False)


def process_batches(final_indecies_filename,
                    final_distances_filename,
                    base_table,
                    query_table,
                    batch_count,
                    batch_size,
                    k,
                    split):
    for start in tqdm(range(0, batch_count)):
        batch_offset = start * batch_size
        batch_length = batch_size if start != batch_count - 1 else len(base_table) - batch_offset

        dataset_batch = base_table.slice(batch_offset, batch_length)
        df = cudf.DataFrame.from_arrow(dataset_batch)

        df_numeric = df.select_dtypes(['float32', 'float64'])
        cleanup(df)

        dataset = cp.from_dlpack(df_numeric.to_dlpack()).copy(order='C')

        # Split the DataFrame into parts (floor division)
        # TODO: pull out this variable
        # split_factor = 50
        split_factor = 1
        splits = split_factor * batch_count
        rows_per_split = len(query_table) // splits

        distances = cudf.DataFrame()
        indices = cudf.DataFrame()
        if split:
            for i in tqdm(range(splits)):
                offset = i * rows_per_split
                length = rows_per_split if i != splits - 1 else len(query_table) - offset  # To handle the last chunk

                query_batch = query_table.slice(offset, length)

                df1 = cudf.DataFrame.from_arrow(query_batch)
                df_numeric1 = df1.select_dtypes(['float32', 'float64'])

                cleanup(df1)
                query = cp.from_dlpack(df_numeric1.to_dlpack()).copy(order='C')

                assert (k <= len(dataset))

                cupydistances1, cupyindices1 = knn(dataset.astype(np.float32),
                                                   query.astype(np.float32),
                                                   k)

                distances1 = cudf.from_pandas(pd.DataFrame(cp.asarray(cupydistances1).get()))
                # add batch_offset to indices
                indices1 = cudf.from_pandas(pd.DataFrame((cp.asarray(cupyindices1) + batch_offset).get()))

                distances = cudf.concat([distances, distances1], ignore_index=True)
                indices = cudf.concat([indices, indices1], ignore_index=True)

            distances.columns = distances.columns.astype(str)
            indices.columns = indices.columns.astype(str)

        assert (len(distances) == len(query_table))
        assert (len(indices) == len(query_table))

        stream_cudf_to_parquet(distances, 100000, final_distances_filename)
        stream_cudf_to_parquet(indices, 100000, final_indecies_filename)

        cleanup(df_numeric, distances, indices, dataset)