import sys
import numpy as np
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


# Generate query_vector.fvec file
def generate_query_vectors_fvec(input_parquet, base_count, query_count):
    table = pq.read_table(input_parquet)
    table = table.slice(0, query_count)
    n = 1536
    print(f'length {n}')
    arrays = []
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
    output_fvec = f'pages_ada_002_{base_count}_query_vectors_{query_count}.fvec'
    write_fvec_from_dataframe(output_fvec, df)
    return output_fvec


# Generate indices.ivec file
def generate_indices_ivec(input_parquet, base_count, query_count):
    df = read_parquet_to_dataframe(input_parquet)
    output_ivec = f'pages_ada_002_{base_count}_indices_query_{query_count}.ivec'
    write_ivec_from_dataframe(output_ivec, df)
    return output_ivec


# Generate base_vectors.fvec file
def generate_base_vectors_fvec(input_parquet, base_count, query_count):
    table = pq.read_table(input_parquet)
    table = table.slice(0, base_count)
    n = 1536
    print(f'length {n}')
    arrays = []
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
    output_fvec = f'pages_ada_002_{base_count}_base_vectors.fvec'
    write_fvec_from_dataframe(output_fvec, df)
    return output_fvec


def main(indices_parquet, base_vectors_parquet, base_count, query_count):
    generate_indices_ivec(indices_parquet, base_count, query_count)
    generate_query_vectors_fvec(base_vectors_parquet, base_count, query_count)
    generate_base_vectors_fvec(base_vectors_parquet, base_count, query_count)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python parquet_to_ivec_fvec.py indices.parquet base_vectors.parquet base_count query_count")
        sys.exit(1)
    indices_parquet = sys.argv[1]
    base_vectors_parquet = sys.argv[2]
    base_count = sys.argv[3]
    query_count = sys.argv[4]
    main(indices_parquet, base_vectors_parquet, base_count, query_count)