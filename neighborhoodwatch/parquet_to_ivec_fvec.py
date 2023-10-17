import sys
import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm
from rich import print as rprint
from rich.markdown import Markdown
import struct


# Reading Parquet into a DataFrame
def read_parquet_to_dataframe(filename):
    table = pq.read_table(filename)
    return table.to_pandas()

def count_vectors(filename):
    with open(filename, 'rb') as f:
        count = 0
        while True:
            dim = f.read(4)
            if not dim:
                break
            dim = struct.unpack('i', dim)[0]
            f.read(4 * dim)
            count += 1
        return count


def write_ivec_fvec_from_dataframe(filename, df, type_char):
    with open(filename, 'wb') as f:
        for index, row in tqdm(df.iterrows()):
            vec = row.values.astype(np.int32)
            if type_char == 'f':
                vec = row.values.astype(np.float32)
            dim = len(vec)
            f.write(dim.to_bytes(4, 'little'))
            f.write(vec.tobytes())


# Generate query_vector.fvec file
def generate_query_vectors_fvec(input_parquet, base_count, query_count):
    table = pq.read_table(input_parquet)
    table = table.slice(0, query_count)
    n = 1536

    column_names = []
    for i in range(n):
        column_names.append(f'embedding_{i}')
    columns_to_drop = list(set(table.schema.names) - set(column_names))

    for col in columns_to_drop:
        if col in table.schema.names:  # Check if the column exists in the table
            col_index = table.schema.get_field_index(col)
            table = table.remove_column(col_index)
    df = table.to_pandas()
    output_fvec = f'ada_002_{base_count}_query_vectors_{query_count}.fvec'
    write_ivec_fvec_from_dataframe(output_fvec, df, 'f')
    return output_fvec


# Generate base_vectors.fvec file
def generate_base_vectors_fvec(input_parquet, base_count):
    table = pq.read_table(input_parquet)
    table = table.slice(0, base_count)
    n = 1536

    column_names = []
    for i in range(n):
        column_names.append(f'embedding_{i}')
    columns_to_drop = list(set(table.schema.names) - set(column_names))
    for col in columns_to_drop:
        if col in table.schema.names:  # Check if the column exists in the table
            col_index = table.schema.get_field_index(col)
            table = table.remove_column(col_index)
    df = table.to_pandas()
    output_fvec = f'ada_002_{base_count}_base_vectors.fvec'
    write_ivec_fvec_from_dataframe(output_fvec, df, 'f')
    return output_fvec


# Generate indices.ivec file
def generate_indices_ivec(input_parquet, base_count, query_count):
    df = read_parquet_to_dataframe(input_parquet)
    output_ivec = f'ada_002_{base_count}_indices_query_{query_count}.ivec'
    write_ivec_fvec_from_dataframe(output_ivec, df, 'i')
    return output_ivec


def generate_files(indices_parquet, base_vectors_parquet, base_count, query_count):
    indices_ivec = generate_indices_ivec(indices_parquet, base_count, query_count)
    query_vector_fvec = generate_query_vectors_fvec(base_vectors_parquet, base_count, query_count)
    base_vector_fvec = generate_base_vectors_fvec(base_vectors_parquet, base_count)
    rprint(Markdown('',f"Generated files: ",''))
    # print counts
    rprint(Markdown(f"*`{indices_ivec}` - indices count*: `{count_vectors(indices_ivec)}`"))
    rprint(Markdown(f"*`{query_vector_fvec}` - query vector count*: `{count_vectors(query_vector_fvec)}`"))
    rprint(Markdown(f"*`{base_vector_fvec}` - base vector count*: `{count_vectors(base_vector_fvec)}`"))


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python parquet_to_ivec_fvec.py indices.parquet base_vectors.parquet base_count query_count")
        sys.exit(1)
    indices_parquet = sys.argv[1]
    base_vectors_parquet = sys.argv[2]
    base_count = sys.argv[3]
    query_count = sys.argv[4]
    generate_files(indices_parquet, base_vectors_parquet, base_count, query_count)
