import sys

import cudf
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import pyarrow as pa
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
# The main data processing taks in this script is based on
# pyarrow.Dataset API
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


# GPU memory per card
def get_gpu_memory():
    import nvidia_smi
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    total_memory = nvidia_smi.nvmlDeviceGetMemoryInfo(handle).total
    nvidia_smi.nvmlShutdown()
    return total_memory


def get_df_for_batch(dataset, batch_size, column_names):
    num_rows = dataset_count(dataset)
    indices = np.arange(0, min(num_rows, batch_size))
    batch = drop_columns(dataset.take(indices), column_names)
    return cudf.DataFrame.from_arrow(batch)


def drop_columns(table, keep_columns):
    columns_to_drop = list(set(table.schema.names) - set(keep_columns))
    for col in columns_to_drop:
        if col in table.schema.names:
            col_index = table.schema.get_field_index(col)
            table = table.remove_column(col_index)
    return table


# dataset: pyarrow.dataset.Dataset
# rmm: RAPIDS memory manager
def tune_memory(dataset, batch_size, max_memory_threshold, rmm, column_names):
    num_rows = dataset_count(dataset)
    rprint(Markdown(f"Tuning memory settings (num rows: {num_rows}; initial batch size: {batch_size}) ..."))

    total_gpu_memory = get_gpu_memory()
    print("-- total memory per GPU:", total_gpu_memory)

    # Measure GPU memory usage after converting to cuDF dataframe
    df = get_df_for_batch(dataset, batch_size, column_names)
    memory_used = df.memory_usage().sum()
    print(f"-- dataframe memory_used {memory_used} for a batch size of {batch_size}")

    factor = math.ceil((total_gpu_memory * max_memory_threshold) / memory_used)
    print(f"-- batch processing adjustment factor (per GPU): {factor}")
    
    batch_size *= factor  # or any other increment factor you find suitable
    while True:
        try:
            if num_rows < batch_size:
                print(f"-- the calculated batch size {batch_size} is bigger than total rows {num_rows}. Use total rows as the target batch size!")
                batch_size = num_rows
                break 
    
            rmm.reinitialize(pool_allocator=False)

            # process a new batch with adjusted batch size
            df = get_df_for_batch(dataset, batch_size, column_names)
            memory_used = df.memory_usage().sum()

            print(f"-- mm_thresh{max_memory_threshold}")
            print(f"-- mem limit {max_memory_threshold*total_gpu_memory}")
            print(f"-- mem used {memory_used}")

            if memory_used > max_memory_threshold * total_gpu_memory:
                # If the memory used goes beyond the threshold, break and set the batch size
                # to the last successful size.
                batch_size = int(0.8 * batch_size)
                print(f"-- found threshold {batch_size}")
                break
            else:
                print(f"-- memory used ratio {memory_used / total_gpu_memory}, batch_size {batch_size}")
                batch_size *= 1.2

        except Exception as e:
            batch_size = int(0.8 * batch_size)
            print(f"-- exception {e}, max {batch_size}")
            break
    
    return batch_size


def load_dataset(data_dir, filename):
    dataset = ds.dataset(get_full_filename(data_dir, filename), format="parquet")
    return dataset


def slice_dataset(dataset, start, end):
    indices = np.arange(start, end)
    return ds.dataset(dataset.take(indices))


def get_embedding_count(table):
    column_names = table.schema.names
    matching_columns = [name for name in column_names if name.startswith('embedding_')]
    return len(matching_columns)


def dataset_count(dataset):
    return dataset.scanner(batch_size=100000).count_rows()


def get_dataset_columns(dataset, n):
    all_columns = dataset.schema.names

    if 'text' in all_columns:
        column_names = ['text']
    if 'question' in all_columns:
        column_names = ['question']

    for i in range(n):
        column_names.append(f'embedding_{i}')

    return column_names


def compute_knn_ds(data_dir,
                   model_name,
                   dimensions,
                   query_filename,
                   query_count,
                   base_filename,
                   base_count,
                   mem_tune=True,
                   k=100,
                   initial_batch_size=200000,
                   max_memory_threshold=0.2,
                   split=True):
    rmm.mr.set_current_device_resource(rmm.mr.PoolMemoryResource(rmm.mr.ManagedMemoryResource()))

    model_prefix = get_model_prefix(model_name)
    batch_size = initial_batch_size

    query_dataset = load_dataset(data_dir, query_filename)
    query_dataset = slice_dataset(query_dataset, 0, query_count)
    query_column_names = get_dataset_columns(query_dataset, dimensions)
    
    base_dataset = load_dataset(data_dir, base_filename)
    base_dataset = slice_dataset(base_dataset, 0, base_count)
    base_column_names = get_dataset_columns(base_dataset, dimensions)

    empty_schema = pa.schema([])
    empty_table = pa.table({}, schema=empty_schema)
    empty_dataset = ds.dataset([empty_table])
    
    if mem_tune:
        batch_size = tune_memory(base_dataset, batch_size, max_memory_threshold, rmm, base_column_names)

    ## Not used (pyarrow.dataset.Dataset.to_batches() used instead)
    # batch_count = math.ceil(len(base_dataset) / batch_size)
    assert (base_count % batch_size == 0) or k <= (base_count % batch_size), f"Cannot generate k of {k} with only {base_count} rows and batch_size of {batch_size}."
    
    process_dataset_batches(data_dir,
                            model_prefix,
                            dimensions,
                            base_dataset, 
                            base_column_names,
                            query_dataset,
                            query_column_names, 
                            batch_size, 
                            query_count,
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


def process_dataset_batches(data_dir,
                            model_prefix,
                            output_dimension,
                            base_dataset, 
                            base_column_names,
                            query_dataset, 
                            query_column_names,
                            batch_size,
                            query_count,
                            k, 
                            split):
    i = 0
    for base_batch in tqdm(base_dataset.to_batches(batch_size=batch_size, columns=base_column_names)):
        batch_offset = batch_size * i
        
        base_df = cudf.DataFrame.from_arrow(pa.Table.from_batches([base_batch]))
        base_df_numeric = base_df.select_dtypes(['float32', 'float64'])
        cleanup(base_df)
        
        base_ds = cp.from_dlpack(base_df_numeric.to_dlpack()).copy(order='C')

        distances = cudf.DataFrame()
        indices = cudf.DataFrame()

        if split:
            assert (k <= len(base_ds))

            for query_batch in query_dataset.to_batches(batch_size=batch_size, columns=query_column_names):
                query_df = cudf.DataFrame.from_arrow(pa.Table.from_batches([query_batch]))
                query_df_numeric = query_df.select_dtypes(['float32', 'float64'])
                cleanup(query_df)

                query_ds = cp.from_dlpack(query_df_numeric.to_dlpack()).copy(order='C')

                cupydistances, cupyindices = knn(base_ds, query_ds, k)
                distances_q = cudf.from_pandas(pd.DataFrame(cp.asarray(cupydistances).get()))
                # add batch_offset to indices
                indices_q = cudf.from_pandas(pd.DataFrame((cp.asarray(cupyindices) + batch_offset).get()))

                # cosine_dist= cosine(dataset[775].get(),query[0].get())
                # distance = 1 - dot_product(dataset[775],query[0])
                # isClose = np.isclose(2*distance, distances1[0][0])
                # print(isClose)

                distances = cudf.concat([distances, distances_q], ignore_index=True)
                indices = cudf.concat([indices, indices_q], ignore_index=True)

            distances.columns = distances.columns.astype(str)
            indices.columns = indices.columns.astype(str)

        assert (len(distances) == query_count)
        assert (len(indices) == query_count)
 
        stream_cudf_to_parquet(distances, 100000, f'{data_dir}/{model_prefix}_{output_dimension}_distances{i}.parquet')
        stream_cudf_to_parquet(indices, 100000, f'{data_dir}/{model_prefix}_{output_dimension}_indices{i}.parquet')

        cleanup(base_ds, base_df_numeric, query_ds, query_df_numeric, 
                distances, indices, distances_q, indices_q, 
                base_batch, query_batch)
        
        i += 1


if __name__ == "__main__":
    if len(sys.argv) != 8:
        print("Usage: python cu_knn.py model_name query_filename query_count base_filename base_count dimensions mem_tune k")
        sys.exit(1)

    model_name = sys.argv[1]
    query_filename = sys.argv[2]
    query_count = int(sys.argv[3])
    base_filename = sys.argv[4]
    base_count = int(sys.argv[5])
    dimensions = int(sys.argv[6])
    mem_tune = sys.argv[7] == 'True'
    k = int(sys.argv[8])

    compute_knn_ds('.',
                   model_name,
                   dimensions,
                   query_filename,
                   query_count,
                   base_filename,
                   base_count,
                   True,
                   k)