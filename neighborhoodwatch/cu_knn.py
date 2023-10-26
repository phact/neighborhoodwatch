import sys

import cudf
import numpy as np
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
import pyarrow.dataset as ds


from neighborhoodwatch.parquet_to_ivec_fvec import dot_product


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


def tune_memory(dataset, batch_size, max_memory_threshold, rmm, column_names):
    rprint(Markdown("Tuning memory settings..."))
    indices = np.arange(0, batch_size)
    batch = drop_columns(dataset.take(indices), column_names)
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
            indices = np.arange(0, batch_size)
            batch = drop_columns(dataset.take(indices), column_names)
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
                print(f"memory used ratio {memory_used / total_gpu_memory}, batch_size {batch_size}")
                batch_size *= 1.2

        except Exception as e:
            batch_size = int(0.8 * batch_size)
            print(f"exception {e}, max {batch_size}")
            break
    return batch_size


def load_table(filename, start, end):
    return pq.read_table(filename).slice(start, end)


def load_dataset(filename, start, end):
    #indices = np.arange(start, end)
    #dataset = ds.dataset(filename, format="parquet").take(indices)
    dataset = ds.dataset(filename, format="parquet")
    return dataset


def drop_columns(table, keep_columns):
    columns_to_drop = list(set(table.schema.names) - set(keep_columns))
    for col in columns_to_drop:
        if col in table.schema.names:
            col_index = table.schema.get_field_index(col)
            table = table.remove_column(col_index)
    return table


def get_embedding_count(dataset):
    column_names = dataset.schema.names
    matching_columns = [name for name in column_names if name.startswith('embedding_')]
    return len(matching_columns)


def dataset_count(dataset):
    return dataset.scanner(batch_size=100000).count_rows()


def prep_dataset(filename, count, n):
    dataset = load_dataset(filename, 0, count)
    assert get_embedding_count(dataset) == n
    #ds_count = dataset_count(dataset)
    #assert ds_count == count, f"Expected {count} rows, got {ds_count} rows."
    column_names = ['text', 'document_id_idx']
    for i in range(n):
        column_names.append(f'embedding_{i}')
    return dataset, column_names


def compute_knn(query_filename, query_count, sorted_data_filename, base_count, dimensions=1536, mem_tune=True, k=100,
                initial_batch_size=100000, max_memory_threshold=0.1, split=True):
    rmm.mr.set_current_device_resource(rmm.mr.PoolMemoryResource(rmm.mr.ManagedMemoryResource()))

    batch_size = initial_batch_size
    # batch_size = 543680

    n = dimensions
    query_dataset, query_column_names = prep_dataset(query_filename, query_count, n)
    base_dataset, base_column_names = prep_dataset(sorted_data_filename, base_count, n)

    if mem_tune:
        batch_size = tune_memory(base_dataset, batch_size, max_memory_threshold, rmm, base_column_names)

    #base_length = dataset_count(base_dataset)
    #query_length = dataset_count(query_dataset)
    base_length = base_count
    query_length = query_count

    batch_count = math.ceil(base_length / batch_size)
    assert (base_length % batch_size == 0) or k <= (base_length % batch_size), f"Cannot generate k of {k} with only {base_length} rows and batch_size of {batch_size}."

    process_batches(base_dataset, base_length, query_dataset, query_length, batch_count, batch_size, k, split, query_column_names, base_column_names)


def cleanup(*args):
    for arg in args:
        try:
            del arg
        except:
            pass
    gc.collect()
    rmm.reinitialize(pool_allocator=False)


def process_batches(base_dataset, base_length, query_dataset, query_length, batch_count, batch_size, k, split, query_column_names, base_column_names):
    for start in tqdm(range(0, batch_count)):
        batch_offset = start * batch_size
        batch_length = batch_size if start != batch_count - 1 else base_length - batch_offset

        np_index = np.arange(batch_offset, batch_offset + batch_length)
        dataset_batch = drop_columns(base_dataset.take(np_index), base_column_names)

        df = cudf.DataFrame.from_arrow(dataset_batch)
        df_numeric = df.select_dtypes(['float32', 'float64'])

        cleanup(df)

        dataset = cp.from_dlpack(df_numeric.to_dlpack()).copy(order='C')

        # Split the DataFrame into parts (floor division)
        # TODO: pull out this variable
        # split_factor = 50
        split_factor = 1
        splits = split_factor * batch_count
        rows_per_split = query_length // splits

        distances = cudf.DataFrame()
        indices = cudf.DataFrame()
        if split:
            for i in tqdm(range(splits)):
                offset = i * rows_per_split
                length = rows_per_split if i != splits - 1 else query_length - offset  # To handle the last chunk

                np_index = np.arange(offset, offset + length)
                query_batch = drop_columns(query_dataset.take(np_index), query_column_names)

                df1 = cudf.DataFrame.from_arrow(query_batch)
                df_numeric1 = df1.select_dtypes(['float32', 'float64'])

                cleanup(df1)
                query = cp.from_dlpack(df_numeric1.to_dlpack()).copy(order='C')

                assert (k <= len(dataset))

                cupydistances1, cupyindices1 = knn(dataset, query, k)

                distances1 = cudf.from_pandas(pd.DataFrame(cp.asarray(cupydistances1).get()))
                # add batch_offset to indices
                indices1 = cudf.from_pandas(pd.DataFrame((cp.asarray(cupyindices1) + batch_offset).get()))

                #cosine_dist= cosine(dataset[775].get(),query[0].get())
                #distance = 1 - dot_product(dataset[775],query[0])
                #isClose = np.isclose(2*distance, distances1[0][0])
                #print(isClose)

                distances = cudf.concat([distances, distances1], ignore_index=True)
                indices = cudf.concat([indices, indices1], ignore_index=True)

            distances.columns = distances.columns.astype(str)
            indices.columns = indices.columns.astype(str)
        assert (len(distances) == query_length)
        assert (len(indices) == query_length)

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
