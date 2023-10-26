import sys
import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm
from rich import print as rprint
from rich.markdown import Markdown
import struct
import os


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

def get_first_vector(filename):
    return get_nth_vector(filename, 0)


def get_nth_vector(filename, n):
    format_char = 'f'
    if filename.endswith("ivec"):
        format_char = 'i'
    with open(filename, 'rb') as f:
        dimension = struct.unpack('i', f.read(4))[0]
        f.seek(4 * n * (1+dimension), 1)
        if (os.path.getsize(filename) < f.tell() + 4 * dimension):
            print("file size is less than expected")
        #f.seek(4 * n * (dimension), 1)
        assert os.path.getsize(filename) >= f.tell() + 4 * dimension
        vector = struct.unpack(format_char * dimension, f.read(4 * dimension))
        if format_char == 'f':
            if ("distances" not in filename):
                if not np.count_nonzero(vector) == 0:
                    if not np.isclose(np.linalg.norm(vector), 1):
                        assert np.isclose(np.linalg.norm(vector), 1), f"Vector {n} in file {filename} is not normalized: {vector}"
                else:
                    print(f"Vector {n} in file {filename} is the zero vector: {vector}")

    return vector


def write_ivec_fvec_from_dataframe(filename, df, type_char, num_columns):
    with open(filename, 'wb') as f:
        for index, row in tqdm(df.iterrows()):
            # potentially remove rownum field
            if len(row.values) == num_columns + 1:
                row = row[:-1]
            assert len(row.values) == num_columns, f"Expected {num_columns} values, got {len(row.values)}"
            vec = row.values.astype(np.int32)
            if type_char == 'f':
                vec = row.values.astype(np.float32)
            dim = len(vec)
            f.write(dim.to_bytes(4, 'little'))
            f.write(vec.tobytes())


# Generate query_vector.fvec file
def generate_query_vectors_fvec(input_parquet, base_count, query_count, n):
    table = pq.read_table(input_parquet)
    table = table.slice(0, query_count)

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
    write_ivec_fvec_from_dataframe(output_fvec, df, 'f', n)
    return output_fvec


# Generate base_vectors.fvec file
def generate_base_vectors_fvec(input_parquet, base_count, k):
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
    write_ivec_fvec_from_dataframe(output_fvec, df, 'f', k)
    return output_fvec

def generate_distances_fvec(input_parquet, count, k):
    df = read_parquet_to_dataframe(input_parquet)
    output_fvec = f'ada_002_{count}_distances_count.fvec'
    write_ivec_fvec_from_dataframe(output_fvec, df, 'f', k)
    return output_fvec


# Generate indices.ivec file
def generate_indices_ivec(input_parquet, base_count, query_count, k):
    df = read_parquet_to_dataframe(input_parquet)
    output_ivec = f'ada_002_{base_count}_indices_query_{query_count}.ivec'
    write_ivec_fvec_from_dataframe(output_ivec, df, 'i', k)
    return output_ivec


def generate_files(indices_parquet, base_vectors_parquet, query_vectors_parquet, final_distances_parquet, base_count, query_count, k, n):
    indices_ivec = generate_indices_ivec(indices_parquet, base_count, query_count, k)
    query_vector_fvec = generate_query_vectors_fvec(query_vectors_parquet, base_count, query_count, n)
    base_vector_fvec = generate_base_vectors_fvec(base_vectors_parquet, base_count, n)
    distances_fvec = generate_distances_fvec(final_distances_parquet, query_count, k)

    rprint(Markdown("Generated files: "), '')
    # print counts
    rprint(Markdown(f"*`{indices_ivec}` - indices count*: `{count_vectors(indices_ivec)}` k*: `{len(get_first_vector(indices_ivec))}`"))
    rprint(Markdown(f"*`{query_vector_fvec}` - query vector count*: `{count_vectors(query_vector_fvec)}` dimensions*: `{len(get_first_vector(query_vector_fvec))}`"))
    rprint(Markdown(f"*`{base_vector_fvec}` - base vector count*: `{count_vectors(base_vector_fvec)}` dimensions*: `{len(get_first_vector(base_vector_fvec))}`"))
    rprint(Markdown(f"*`{distances_fvec}` - distances count*: `{count_vectors(distances_fvec)}` k*: `{len(get_first_vector(distances_fvec))}`"))

    return query_vector_fvec, indices_ivec, distances_fvec, base_vector_fvec

def validate_files(query_vector_fvec, indices_ivec, distances_fvec, base_vector_fvec):

    for n in range(count_vectors(query_vector_fvec)):
        nth_query_vector = get_nth_vector(query_vector_fvec, n)
        first_indexes = get_nth_vector(indices_ivec, n)
        distance_vector = get_nth_vector(distances_fvec, n)

        if np.count_nonzero(nth_query_vector) == 0:
            continue
        col = 0
        for index in first_indexes:
            base_vector = get_nth_vector(base_vector_fvec, index)
            similarity = dot_product(nth_query_vector, base_vector)
            distance = distance_vector[col]
            if not np.isclose((1-similarity), distance/2):
                print(f"Expected 1 - similarity {1-similarity} to equal distance {distance} for query vector {n} and base vector {index}")
                print(f"Difference {1-similarity - distance/2}")
                #print(base_vector)
                #print(nth_query_vector)
            col += 1


def dot_product(A, B):
    return sum(a * b for a, b in zip(A, B))

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python parquet_to_ivec_fvec.py indices.parquet base_vectors.parquet base_count query_count")
        sys.exit(1)
    indices_parquet = sys.argv[1]
    base_vectors_parquet = sys.argv[2]
    base_count = sys.argv[3]
    query_count = sys.argv[4]
    generate_files(indices_parquet, base_vectors_parquet, base_count, query_count)
