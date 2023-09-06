import cudf
import pyarrow as pa
import pyarrow.parquet as pq
import cupy as cp
from cuml.neighbors import KNeighborsClassifier
from cuml.datasets import make_blobs
from cuml.neighbors import NearestNeighbors
import pandas as pd
from pylibraft.neighbors.brute_force import knn
import pylibraft
import gc
import math
import time


import rmm
rmm.mr.set_current_device_resource(rmm.mr.PoolMemoryResource(rmm.mr.ManagedMemoryResource()))

import nvidia_smi

nvidia_smi.nvmlInit()

# card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate
handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)

total_gpu_memory = nvidia_smi.nvmlDeviceGetMemoryInfo(handle).total

print("Total memory:", total_gpu_memory)

nvidia_smi.nvmlShutdown()

import rmm
rmm.mr.set_current_device_resource(rmm.mr.PoolMemoryResource(rmm.mr.ManagedMemoryResource()))


#batch_size = 10000
batch_size = 543680
max_memory_threshold = 0.1

table = pq.read_table('split.parquet')

batch = table.slice(0, batch_size)
df = cudf.DataFrame.from_arrow(batch)

# Measure GPU memory usage after converting to cuDF dataframe
memory_used = df.memory_usage().sum()
print(f"memory_used {memory_used}")

factor = math.ceil((total_gpu_memory*.2)/memory_used)
print(f"factor {factor}")
batch_size *= factor # or any other increment factor you find suitable

while True:
    try:
        print(f"batch_size {batch_size}")
        rmm.reinitialize(pool_allocator=False)  
        batch = table.slice(0, batch_size)
        df = cudf.DataFrame.from_arrow(batch)
        
        # Measure GPU memory usage after converting to cuDF dataframe
        memory_used = df.memory_usage().sum()

        if memory_used > max_memory_threshold * total_gpu_memory:
            # If the memory used goes beyond the threshold, break and set the batch size
            # to the last successful size.
            batch_size  = int(0.8 * batch_size)
            print(f"found threshold {batch_size}")
            break
        else:
            print( memory_used / total_gpu_memory)
            batch_size *= 1.2  # or any other increment factor you find suitable

    except Exception as e:
        batch_size = int(0.8 * batch_size)
        print(f"exception {e}, max {batch_size}")
        break


#before batching clean up the table

print(f'length table: {len(table)}')

n = len(table.column('embedding')[0])  # Determine the length of one of the embedding lists
print(f'length {n}')

# Expand the embedding column into multiple columns
arrays = []
column_names = ['seq_no', 'content']
for i in range(n):
    column_names.append(f'embedding_{i}')

columns_to_drop = list(set(table.schema.names) - set(column_names))

print(f'columns_to_drop {columns_to_drop}')

for col in columns_to_drop:
    if col in table.schema.names:  # Check if the column exists in the table
        col_index = table.schema.get_field_index(col)
        table = table.remove_column(col_index)


print(f'batch_size {batch_size}')

batch_count=math.ceil(len(table)/batch_size)
print(f'batch_count {batch_count}')

#TODO: remove /10
#batch_size = int(batch_size/10)

for start in range(0, batch_count):
    if start !=batch_count:
      batch = table.slice(start, batch_size)
    else:
      batch = table.slice(start, len(table) - start*batch_size)
    df = cudf.DataFrame.from_arrow(batch)
    print(f'length df: {len(df)}')
    print(f'start: {start}')
    print(f'batch_size: {batch_size}')

    k=100
    split = True

#    # Check if 'seq_no' column does not exist
#    if 'seq_no' not in df.columns:
#        # Sort by partition_key
#        df = df.sort_values(by='partition_key')
#
#        # Add sequence number
#        df['seq_no'] = range(start, start + batch_size)


#    n = len(df['embedding'][0])  # Get the length of one of the embedding lists to determine the number of columns
#    print(f'length {n}')

#    column_names = []
#    column_names.append('seq_no')
#    for i in range(n):
#        column_names.append(f'embedding_{i}')
    
#    # Determine columns to drop
#    columns_to_drop = list(set(df.columns) - set(column_names))

#    # Drop the columns
#    df.drop(columns=columns_to_drop, inplace=True)

#    gc.collect()
#    rmm.reinitialize(pool_allocator=False)  

#    print("df.columns")
#    print(df.columns)

    df_numeric = df.select_dtypes(['float32', 'float64'])

    del df
    gc.collect()
    rmm.reinitialize(pool_allocator=False)  

    #sample_df = df_numeric.sample(frac=1)
    print(f"starting fit for k ={k}")
    nn = NearestNeighbors(n_neighbors=k, algorithm='brute')
    nn.fit(df_numeric)
    print("done fit")

    print("df_numeric")
    print(df_numeric)

    # Split the DataFrame into parts (floor division)
    splits=50
    mid_idx = len(df_numeric) // splits

    distances = cudf.DataFrame()
    indices = cudf.DataFrame()
    if (split):
        for i in range(splits):
          print(f"i {i} of {splits}")

          start_idx = i * mid_idx
          end_idx = (i + 1) * mid_idx if i != splits - 1 else len(df_numeric)  # To handle the last chunk
          df1 = df_numeric.iloc[start_idx:end_idx]

          num_rows, num_columns = df1.shape
          print(f"df1 Number of rows: {num_rows}")
          print(f"df1 Number of columns: {num_columns}")

          distances1, indices1 = nn.kneighbors(df1, two_pass_precision=True)
          distances = cudf.concat([distances, distances1], ignore_index=True)
          indices = cudf.concat([indices, indices1], ignore_index=True)

        distances.columns = distances.columns.astype(str)
        indices.columns = indices.columns.astype(str)
    print("done distances")



    print("distances, indices, batch_size")
    print(len(distances))
    print(len(indices))
    print(batch_size)
    assert(len(distances) == batch_size)
    assert(len(indices) == batch_size)
    print("start*batch_size")
    distances['RowNum'] = range(start*batch_size, start*batch_size + len(distances))
    indices['RowNum'] = range(start*batch_size, start*batch_size + len(indices))


    print(f"distances: \n{distances.iloc[127:128]}")
    print(f"indices: \n{indices.iloc[127:128]}")
    print(f"embeddings: \n{df_numeric.iloc[127:128]}")


    # Extract indices from the first row (excluding the 'RowNum' column)
    selected_indices = indices.iloc[0].drop("RowNum").values
    selected_indices = indices.iloc[127].drop("RowNum").values
    selected_indices = indices.iloc[7].drop("RowNum").values

    print(selected_indices)
    # Fetch the rows from the embeddings DataFrame
    selected_rows = df_numeric.loc[selected_indices]

    print(selected_rows)

    # Save distances
    distances.to_parquet(f'distances{start}.parquet')

    # Save indices
    indices.to_parquet(f'indices{start}.parquet')

    del df_numeric
    del distances
    del indices 
    del distances1
    del indices1
    gc.collect()
    rmm.reinitialize(pool_allocator=False)  


