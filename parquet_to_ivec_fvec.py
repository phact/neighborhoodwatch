import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# Reading Parquet into a DataFrame
def read_parquet_to_dataframe(filename):
    table = pq.read_table(filename)
    return table.to_pandas()

# Writing DataFrame to fvec
def write_fvec_from_dataframe(filename, df):
    with open(filename, 'wb') as f:
        for index, row in df.iterrows():
            vec = row.values.astype(np.float32)
            dim = len(vec)
            f.write(dim.to_bytes(4, 'little'))
            f.write(vec.tobytes())

# Writing DataFrame to ivec
def write_ivec_from_dataframe(filename, df):
    with open(filename, 'wb') as f:
        for index, row in df.iterrows():
            vec = row.values.astype(np.int32)
            dim = len(vec)
            f.write(dim.to_bytes(4, 'little'))
            f.write(vec.tobytes())

df = read_parquet_to_dataframe('final_distances.parquet')
write_fvec_from_dataframe('pages_ada_002_distances.fvec', df)

df = read_parquet_to_dataframe('pages_ada_002_final_indices.parquet')
write_ivec_from_dataframe('indices.ivec', df)


#table = pq.read_table('split.parquet')
table = pq.read_table('pages_ada_002_sorted.parquet')

#n = len(table.column('embedding')[0])  # Determine the length of one of the embedding lists
n = 1536
print(f'length {n}')
arrays = []

#column_names = ['seq_no', 'content']
column_names = []
for i in range(n):
    column_names.append(f'embedding_{i}')

columns_to_drop = list(set(table.schema.names) - set(column_names))

print(f'columns_to_drop {columns_to_drop}')

for col in columns_to_drop:
    if col in table.schema.names:  # Check if the column exists in the table
        col_index = table.schema.get_field_index(col)
        table = table.remove_column(col_index)

df = table.to_pandas()

write_fvec_from_dataframe('pages_ada_002_base_vectors.fvec', df)
