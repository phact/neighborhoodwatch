import sys

import h5py
import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm
from rich import print as rprint
from rich.markdown import Markdown
import struct
import os


##
# Convert the Parquet file into the specified target format.
# - 'ivec', 'fvec', and 'hdf5' format are currently supported
##


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
def read_and_extract(input_parquet, base_count, dimensions):
    table = pq.read_table(input_parquet)
    table = table.slice(0, base_count)

    column_names = []
    for i in range(dimensions):
        column_names.append(f'embedding_{i}')
    columns_to_drop = list(set(table.schema.names) - set(column_names))
    for col in columns_to_drop:
        if col in table.schema.names:  # Check if the column exists in the table
            col_index = table.schema.get_field_index(col)
            table = table.remove_column(col_index)
    df = table.to_pandas()
    return df


def generate_query_vectors_fvec(data_dir, input_parquet, base_count, query_count, dimensions, model_prefix):
    df = read_and_extract(input_parquet, base_count, dimensions)
    output_fvec = f'{data_dir}/{model_prefix}_{base_count}_query_vectors_{query_count}.fvec'
    write_ivec_fvec_from_dataframe(output_fvec, df, 'f', dimensions)
    return output_fvec


# Generate base_vectors.fvec file
def generate_base_vectors_fvec(data_dir, input_parquet, base_count, k, dimensions, model_prefix):
    df = read_and_extract(input_parquet, base_count, dimensions)
    output_fvec = f'{data_dir}/{model_prefix}_{base_count}_base_vectors.fvec'
    write_ivec_fvec_from_dataframe(output_fvec, df, 'f', dimensions)
    return output_fvec


def generate_distances_fvec(data_dir, input_parquet,base_count, count, k, model_prefix):
    df = read_parquet_to_dataframe(input_parquet)
    output_fvec = f'{data_dir}/{model_prefix}_{base_count}_distances_{count}.fvec'
    write_ivec_fvec_from_dataframe(output_fvec, df, 'f', k)
    return output_fvec


# Generate indices.ivec file
def generate_indices_ivec(data_dir, input_parquet, base_count, query_count, k, model_prefix):
    df = read_parquet_to_dataframe(input_parquet)
    output_ivec = f'{data_dir}/{model_prefix}_{base_count}_indices_query_{query_count}.ivec'
    write_ivec_fvec_from_dataframe(output_ivec, df, 'i', k)
    return output_ivec


def generate_ivec_fvec_files(data_dir,
                             indices_parquet, 
                             base_vectors_parquet, 
                             query_vectors_parquet, 
                             final_distances_parquet, 
                             base_count, 
                             query_count, 
                             k, 
                             dimensions, 
                             model_name):
    if model_name:
        model_prefix = model_name.replace("/", "_")
    else:
        model_prefix = "ada_002"
    indices_ivec = generate_indices_ivec(data_dir, indices_parquet, base_count, query_count, k, model_prefix)
    query_vector_fvec = generate_query_vectors_fvec(data_dir, query_vectors_parquet, base_count, query_count, dimensions, model_prefix)
    base_vector_fvec = generate_base_vectors_fvec(data_dir, base_vectors_parquet, base_count, k, dimensions, model_prefix)
    distances_fvec = generate_distances_fvec(data_dir, final_distances_parquet, base_count, query_count, k, model_prefix)

    rprint(Markdown("Generated files: "), '')
    # print counts
    rprint(Markdown(f"*`{indices_ivec}` - indices count*: `{count_vectors(indices_ivec)}` k*: `{len(get_first_vector(indices_ivec))}`"))
    rprint(Markdown(f"*`{query_vector_fvec}` - query vector count*: `{count_vectors(query_vector_fvec)}` dimensions*: `{len(get_first_vector(query_vector_fvec))}`"))
    rprint(Markdown(f"*`{base_vector_fvec}` - base vector count*: `{count_vectors(base_vector_fvec)}` dimensions*: `{len(get_first_vector(base_vector_fvec))}`"))
    rprint(Markdown(f"*`{distances_fvec}` - distances count*: `{count_vectors(distances_fvec)}` k*: `{len(get_first_vector(distances_fvec))}`"))

    return query_vector_fvec, indices_ivec, distances_fvec, base_vector_fvec


def generate_hdf5_file(data_dir,
                       indices_parquet, 
                       base_vectors_parquet, 
                       query_vectors_parquet, 
                       final_distances_parquet, 
                       base_count, 
                       query_count, 
                       k, 
                       dimensions, 
                       model_name):
    if model_name:
        model_prefix = model_name.replace("/", "_")
    else:
        model_prefix = "ada_002"

    filename = f'{data_dir}/{model_prefix}_base_{base_count}_query_{query_count}.hdf5'

    df = read_and_extract(base_vectors_parquet, base_count, dimensions)
    write_hdf5(df, filename, 'train')

    df = read_and_extract(query_vectors_parquet, base_count, dimensions)
    write_hdf5(df, filename, 'test')

    df = read_parquet_to_dataframe(final_distances_parquet)
    write_hdf5(df, filename, 'distances')

    df = read_parquet_to_dataframe(indices_parquet)
    write_hdf5(df, filename, 'neighbors')


def write_hdf5(df, filename, datasetname):
    data = df.values
    with h5py.File(filename, 'a') as f:
        if datasetname in f:
            print(f"Dataset '{datasetname}' already exists in file '{filename}'")
        else:
            f.create_dataset(datasetname, data=data)


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
        print("Usage: python parquet_to_dest_format.py indices.parquet base_vectors.parquet base_count query_count")
        sys.exit(1)
    indices_parquet = sys.argv[1]
    base_vectors_parquet = sys.argv[2]
    base_count = sys.argv[3]
    query_count = sys.argv[4]
    generate_files(indices_parquet, base_vectors_parquet, base_count, query_count)
