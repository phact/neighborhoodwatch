import sys

import cudf
import pyarrow.parquet as pq
import cupy as cp
from tqdm import tqdm
import pandas as pd
import gc
import math
from pylibraft.neighbors.brute_force import knn
import rmm
from rich import print as rprint
from rich.markdown import Markdown


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
    rprint(Markdown("Tuning memory settings..."))
    batch = table.slice(0, batch_size)
    df = cudf.DataFrame.from_arrow(batch)

    import nvidia_smi
    nvidia_smi.nvmlInit()
    # card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)

    total_gpu_memory = nvidia_smi.nvmlDeviceGetMemoryInfo(handle).total

    print("Total memory:", total_gpu_memory)

    nvidia_smi.nvmlShutdown()

    # Measure GPU memory usage after converting to cuDF dataframe
    memory_used = df.memory_usage().sum()
    print(f"memory_used {memory_used}")
    factor = math.ceil((total_gpu_memory * .2) / memory_used)
    print(f"factor {factor}")
    batch_size *= factor  # or any other increment factor you find suitable
    while True:
        try:
            rmm.reinitialize(pool_allocator=False)
            batch = table.slice(0, batch_size)
            df = cudf.DataFrame.from_arrow(batch)

            # Measure GPU memory usage after converting to cuDF dataframe
            memory_used = df.memory_usage().sum()

            if memory_used > max_memory_threshold * total_gpu_memory:
                # If the memory used goes beyond the threshold, break and set the batch size
                # to the last successful size.
                batch_size = int(0.8 * batch_size)
                print(f"found threshold {batch_size}")
                break
            else:
                print(memory_used / total_gpu_memory)
                batch_size *= 1.2  # or any other increment factor you find suitable

        except Exception as e:
            batch_size = int(0.8 * batch_size)
            print(f"exception {e}, max {batch_size}")
            break
    return batch_size


def load_table(filename, start, end):
    return pq.read_table(filename).slice(start, end)


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


def prep_table(filename, count, n):
    table = load_table(filename, 0, count)
    assert get_embedding_count(table) == n
    assert len(table) == count, f"Expected {count} rows, got {len(table)} rows."
    column_names = ['text', 'document_id_idx']
    for i in range(n):
        column_names.append(f'embedding_{i}')
    return drop_columns(table, column_names)


def compute_knn(query_filename, query_count, sorted_data_filename, base_count, dimensions=1536, mem_tune=True, k=100,
                initial_batch_size=100000, max_memory_threshold=0.1, split=True):
    rmm.mr.set_current_device_resource(rmm.mr.PoolMemoryResource(rmm.mr.ManagedMemoryResource()))

    batch_size = initial_batch_size
    # batch_size = 543680

    n = dimensions
    query_table = prep_table(query_filename, query_count, n)
    table = prep_table(sorted_data_filename, base_count, n)

    if mem_tune:
        batch_size = tune_memory(table, batch_size, max_memory_threshold, rmm)

    batch_count = math.ceil(len(table) / batch_size)
    assert ((len(table) % batch_size == 0) or k <= (len(table) % batch_size)), f"Cannot generate k of {k} with only {len(table)} rows and batch_size of {batch_size}."

    process_batches(table, query_table, batch_count, batch_size, k, split)


def cleanup(*args):
    for arg in args:
        try:
            del arg
        except:
            pass
    gc.collect()
    rmm.reinitialize(pool_allocator=False)


def process_batches(table, query_table, batch_count, batch_size, k, split):
    for start in tqdm(range(0, batch_count)):
        batch_offset = start * batch_size
        batch_length = batch_size if start != batch_count - 1 else len(table) - batch_offset
        dataset_batch = table.slice(batch_offset, batch_length)

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

                cupydistances1, cupyindices1 = knn(dataset, query, k)

                distances1 = cudf.from_pandas(pd.DataFrame(cp.asarray(cupydistances1).get()))
                # add batch_offset to indices
                indices1 = cudf.from_pandas(pd.DataFrame((cp.asarray(cupyindices1) + batch_offset).get()))

                distances = cudf.concat([distances, distances1], ignore_index=True)
                indices = cudf.concat([indices, indices1], ignore_index=True)

            distances.columns = distances.columns.astype(str)
            indices.columns = indices.columns.astype(str)
        assert (len(distances) == len(query_table))
        assert (len(indices) == len(query_table))

        distances['RowNum'] = range(0, len(distances))
        indices['RowNum'] = range(0, len(indices))

        stream_cudf_to_parquet(distances, 100000, f'distances{start}.parquet')
        stream_cudf_to_parquet(indices, 100000, f'indices{start}.parquet')

        cleanup(df_numeric, distances, indices, dataset)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python cu_knn.py query_filename query_count sorted_data_filename base_count dimensions mem_tune k")
        sys.exit(1)

    query_filename = sys.argv[1]
    query_count = int(sys.argv[2])
    sorted_data_filename = sys.argv[3]
    base_count = int(sys.argv[4])
    dimensions = int(sys.argv[5])
    mem_tune = sys.argv[6] == 'True'
    k = int(sys.argv[7])

    compute_knn(query_filename, query_count, sorted_data_filename, base_count, dimensions, True, k)