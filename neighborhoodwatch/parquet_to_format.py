import sys
import cupy as cp
import cupy as cp
import h5py
import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm
from rich import print as rprint
from rich.markdown import Markdown
import struct
import os
from neighborhoodwatch.nw_utils import *

from cuvs.distance import pairwise_distance
from cuvs.neighbors import brute_force
import torch
torch.device("cuda")


##
# Convert the Parquet file into the specified target format.
# - 'ivec', 'fvec', and 'hdf5' format are currently supported
##


# Reading Parquet into a DataFrame
def read_parquet_to_dataframe(data_dir, filename):
    full_filename = get_full_filename(data_dir, filename)
    table = pq.read_table(full_filename)
    return table.to_pandas()


def count_vectors(data_dir, filename):
    full_filename = get_full_filename(data_dir, filename)
    with open(full_filename, 'rb') as f:
        count = 0
        while True:
            dim = f.read(4)
            if not dim:
                break
            dim = struct.unpack('i', dim)[0]
            f.read(4 * dim)
            count += 1
        return count


def get_first_vector(data_dir, filename):
    return get_nth_vector(data_dir, filename, 0)


def get_nth_vector(data_dir, filename, n):
    full_filename = get_full_filename(data_dir, filename)
    format_char = 'f'
    if full_filename.endswith("ivec"):
        format_char = 'i'
    with open(full_filename, 'rb') as f:
        dimension = struct.unpack('i', f.read(4))[0]
        f.seek(4 * n * (1+dimension), 1)
        if (os.path.getsize(full_filename) < f.tell() + 4 * dimension):
            print("file size is less than expected")
        #f.seek(4 * n * (dimension), 1)
        assert os.path.getsize(full_filename) >= f.tell() + 4 * dimension
        vector = struct.unpack(format_char * dimension, f.read(4 * dimension))

    return vector


def write_ivec_fvec_from_dataframe(data_dir, filename, df, type_char, num_columns, ignore_dimension_check=False):
    full_filename = get_full_filename(data_dir, filename)
    with open(full_filename, 'wb') as f:
        for index, row in tqdm(df.iterrows()):
            # potentially remove rownum field
            if len(row.values) == num_columns + 1:
                row = row[:-1]
            if not ignore_dimension_check:
                assert len(row.values) == num_columns, f"Expected {num_columns} values, got {len(row.values)} [filename: {filename}]"
            vec = row.values.astype(np.int32)
            if type_char == 'f':
                vec = row.values.astype(np.float32)
            dim = len(vec)
            f.write(dim.to_bytes(4, 'little'))
            f.write(vec.tobytes())


def read_and_extract(data_dir, input_parquet, rowcount, dimensions, column_names=None):
def read_and_extract(data_dir, input_parquet, rowcount, dimensions, column_names=None):
    full_filename = get_full_filename(data_dir, input_parquet)
    table = pq.read_table(full_filename)
    table = table.slice(0, rowcount)
    table = table.slice(0, rowcount)

    if column_names is None:
        column_names = []
        for i in range(dimensions):
            column_names.append(f'embedding_{i}')
        columns_to_drop = list(set(table.schema.names) - set(column_names))
        for col in columns_to_drop:
            if col in table.schema.names:  # Check if the column exists in the table
                col_index = table.schema.get_field_index(col)
                table = table.remove_column(col_index)

    if column_names is None:
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


def generate_query_vectors_fvec(data_dir,
                                input_parquet,
                                query_count,
                                dimensions,
                                query_vectors_fvec_file,
                                output_hdf5=True,
                                hdf5_file=None):
    df = read_and_extract(data_dir, input_parquet, query_count, dimensions)

    if not os.path.exists(query_vectors_fvec_file):
        write_ivec_fvec_from_dataframe(data_dir, query_vectors_fvec_file, df, 'f', dimensions)
    else:
        print(f"File {query_vectors_fvec_file} already exists")

    if output_hdf5:
        # no-op if hdf5 file exists and the 'test' group also exists
        write_hdf5(data_dir, df, hdf5_file, 'test')


def generate_base_vectors_fvec(data_dir,
                               input_parquet,
                               base_count,
                               dimensions,
                               column_names,
                               base_vectors_fvec_file,
                               output_hdf5=True,
                               hdf5_file=None):
    df = read_and_extract(data_dir, input_parquet, base_count, dimensions, column_names)

    if not os.path.exists(base_vectors_fvec_file):
        write_ivec_fvec_from_dataframe(data_dir, base_vectors_fvec_file, df, 'f', dimensions)
    else:
        print(f"File {base_vectors_fvec_file} already exists")

    if output_hdf5:
        # no-op if hdf5 file exists and the 'train' group also exists
        write_hdf5(data_dir, df, hdf5_file, 'train')


def generate_distances_fvec(data_dir,
                            input_parquet,
                            k,
                            distances_fvec_file,
                            output_hdf5=True,
                            hdf5_file=None):
    df = read_parquet_to_dataframe(data_dir, input_parquet)

    if not os.path.exists(distances_fvec_file):
        write_ivec_fvec_from_dataframe(data_dir, distances_fvec_file, df, 'f', k)
    else:
        print(f"File {distances_fvec_file} already exists")

    if output_hdf5:
        # no-op if hdf5 file exists and the 'test' group also exists
        write_hdf5(data_dir, df, hdf5_file, 'distances')


def generate_indices_ivec(data_dir,
                          input_parquet,
                          k,
                          distances_fvec_file,
                          output_hdf5=True,
                          hdf5_file=None):
    df = read_parquet_to_dataframe(data_dir, input_parquet)

    if not os.path.exists(distances_fvec_file):
        write_ivec_fvec_from_dataframe(data_dir, distances_fvec_file, df, 'i', k)
    else:
        print(f"File {distances_fvec_file} already exists")

    if output_hdf5:
        # no-op if hdf5 file exists and the 'test' group also exists
        write_hdf5(data_dir, df, hdf5_file, 'neighbors')


def generate_output_files(data_dir,
                          model_prefix,
                          dimensions,
                          base_vectors_parquet,
                          query_vectors_parquet,
                          final_indices_parquet,
                          final_distances_parquet,
                          base_count,
                          query_count,
                          k,
                          output_hdf5=True,
                          column_names=None,
                          output_dtype=None):
    rprint(Markdown(f"Generated files (output_hdf5: {output_hdf5}): "), '')

    hdf5_filename = f'{model_prefix}_{dimensions}_base_{base_count}_query_{query_count}.hdf5'
    hdf5_filename = get_full_filename(data_dir, hdf5_filename)

    if output_dtype is not None:
        query_vector_fvec_base = f'{data_dir}/{model_prefix}_{dimensions}_{output_dtype}_query_vectors_{query_count}'
        base_vector_fvec_base = f'{data_dir}/{model_prefix}_{dimensions}_{output_dtype}_base_vectors_{base_count}'
        indices_ivec_base = f'{data_dir}/{model_prefix}_{dimensions}_{output_dtype}_indices_b{base_count}_q{query_count}_k{k}'
        distances_fvec_base = f'{data_dir}/{model_prefix}_{dimensions}_{output_dtype}_distances_b{base_count}_q{query_count}_k{k}'
    else:
        query_vector_fvec_base = f'{data_dir}/{model_prefix}_{dimensions}_query_vectors_{query_count}'
        base_vector_fvec_base = f'{data_dir}/{model_prefix}_{dimensions}_base_vectors_{base_count}'
        indices_ivec_base = f'{data_dir}/{model_prefix}_{dimensions}_indices_b{base_count}_q{query_count}_k{k}'
        distances_fvec_base = f'{data_dir}/{model_prefix}_{dimensions}_distances_b{base_count}_q{query_count}_k{k}'

    query_vector_fvec_file = f'{query_vector_fvec_base}.fvec'
    generate_query_vectors_fvec(data_dir,
                                query_vectors_parquet,
                                query_count,
                                dimensions,
                                query_vector_fvec_file,
                                output_hdf5,
                                hdf5_filename)
    rprint(Markdown(f"*`{query_vector_fvec_file}`* - "
                    f"query vector count: `{count_vectors(data_dir, query_vector_fvec_file)}`, "
                    f"dimensions: `{len(get_first_vector(data_dir, query_vector_fvec_file))}`"))

    base_vector_fvec_file = f'{base_vector_fvec_base}.fvec'
    generate_base_vectors_fvec(data_dir,
                               base_vectors_parquet,
                               base_count,
                               dimensions,
                               column_names,
                               base_vector_fvec_file,
                               output_hdf5,
                               hdf5_filename)
    rprint(Markdown(f"*`{base_vector_fvec_file}`* - "
                    f"base vector count: `{count_vectors(data_dir, base_vector_fvec_file)}`, "
                    f"dimensions: `{len(get_first_vector(data_dir, base_vector_fvec_file))}`"))

    indices_ivec_file = f'{indices_ivec_base}.ivec'
    generate_indices_ivec(data_dir,
                          final_indices_parquet,
                          k,
                          indices_ivec_base,
                          output_hdf5,
                          hdf5_filename)
    rprint(Markdown(f"*`{indices_ivec_file}`* - "
                    f"indices count: `{count_vectors(data_dir, indices_ivec_file)}`, "
                    f"k: `{len(get_first_vector(data_dir, indices_ivec_file))}`"))

    distances_fvec_file = f'{distances_fvec_base}.fvec'
    generate_distances_fvec(data_dir,
                            final_distances_parquet,
                            k,
                            distances_fvec_file,
                            output_hdf5,
                            hdf5_filename)
    rprint(Markdown(f"*`{distances_fvec_file}`* - "
                    f"distances count: `{count_vectors(data_dir, distances_fvec_file)}`, "
                    f"k: `{len(get_first_vector(data_dir, distances_fvec_file))}`"))


def write_hdf5(data_dir, df, filename, group):
    data = df.values
    full_filename = get_full_filename(data_dir, filename)
    with h5py.File(full_filename, 'a') as f:
        if group in f:
            print(f"Group '{group}' already exists in file '{full_filename}'")
        else:
            f.create_dataset(group, data=data)


def validate_files(data_dir, query_vector_fvec, base_vector_fvec, indices_ivec, distances_fvec, columns=None,
                   input_parquet=None):
    zero_query_vector_count = 0
    total_query_vector_count = count_vectors(data_dir, query_vector_fvec)
    total_mismatch_count = 0

    for n in range(total_query_vector_count):
        nth_query_vector = get_nth_vector(data_dir, query_vector_fvec, n)
        first_indexes = get_nth_vector(data_dir, indices_ivec, n)
        distance_vector = get_nth_vector(data_dir, distances_fvec, n)

        if np.count_nonzero(nth_query_vector) == 0:
            print(f"Skipping zero query vector {n}/{total_query_vector_count}")
            zero_query_vector_count += 1
            continue

        col = 0
        last_distance = 0
        last_similarity = 1
        last_distance = 0
        last_similarity = 1
        for index in first_indexes:
            base_vector = get_nth_vector(data_dir, base_vector_fvec, index)
            similarity = dot_product(nth_query_vector, base_vector)
            distance = 1 - distance_vector[col]
            assert distance >= last_distance, f"Expected distance {distance} to be greater than last distance {last_distance}"
            last_distance = distance
            # assert similarity <= last_similarity, f"Expected similarity {similarity} to be less than last similarity {last_similarity}"
            last_similarity = similarity
            if not np.isclose((1 - similarity), distance, atol=1e-04):
                # start looking at cuvs
                output = cp.asarray(pairwise_distance(
                    tuple_to_cuda_interface_array(nth_query_vector),
                    tuple_to_cuda_interface_array(base_vector),
                    metric="cosine"
                ))[0][0]

                full_base = read_and_extract(
                    data_dir,
                    input_parquet,
                    count_vectors(data_dir, base_vector_fvec),
                    len(nth_query_vector),
                    columns
                )
                full_base_cp = cp.array(full_base.to_numpy(), order='C')
                full_index = brute_force.build(full_base_cp, metric="cosine")
                full_distances, full_neighbors = brute_force.search(full_index,
                                                                    tuple_to_cuda_interface_array(nth_query_vector),
                                                                    100000)
                this_index = cp.asarray(full_neighbors)[0].tolist().index(index)
                full_brute_force_distance = cp.asarray(full_distances)[0][this_index]

                search_index = brute_force.build(tuple_to_cuda_interface_array(base_vector), metric="cosine")
                distances, neighbors = brute_force.search(search_index, tuple_to_cuda_interface_array(nth_query_vector),
                                                          1)
                brute_force_distance = cp.asarray(distances)[0][0]

                this_index = first_indexes.index(index)

                distance_tensor = torch.matmul(torch.tensor(nth_query_vector, dtype=torch.float32),
                                               torch.tensor(base_vector, dtype=torch.float32))
                distances_tensor, indices_tensor = torch.topk(distance_tensor, 1, largest=True)

                total_mismatch_count += 1
                print(
                    f"torch {1 - distances_tensor.item()} full {full_brute_force_distance} brute {brute_force_distance} distance {distance} similarity {1 - similarity}")
                # print(f"torch {1-distances_tensor.item()} brute {brute_force_distance} distance {distance} similarity {1-similarity}")
                print(
                    f"Expected 1 - similarity: {1 - similarity} to equal distance: {distance} for query vector {n} and base vector {index}")
                print(f"distance vs similarity diff: {distance - (1 - similarity)}")
                print(f"cuvs vs similarity diff: {output - (1 - similarity)}")
                print(f"brute vs similarity diff: {brute_force_distance - (1 - similarity)}")
                print(f"full brute vs similarity diff: {full_brute_force_distance - (1 - similarity)}")
                print(f"torch vs similarity diff: {distances_tensor.item() - similarity}")
                print(base_vector)
                print(nth_query_vector)
            col += 1

    print(f"Total mismatch count: {total_mismatch_count}")


def dot_product(A, B):
    return sum(a * b for a, b in zip(A, B))


def tuple_to_cuda_interface_array(tuple_data, dtype=np.float32):
    """
    Converts a tuple of length 128 into a CuPy array (GPU compatible) with shape (1, 128).

    Parameters:
    tuple_data (tuple): A tuple of length 128.
    dtype (type): The data type of the output array (e.g., np.float32, np.int8, np.uint8).

    Returns:
    cp.ndarray: A CuPy array (GPU memory) that is CUDA array interface compliant.
    """
    if len(tuple_data) != 128:
        raise ValueError("The input tuple must have exactly 128 elements.")

    # Convert the tuple to a numpy array with the given dtype
    array_data = np.array(tuple_data, dtype=dtype)

    # Reshape to (1, 128) to match the required (n_samples, dim) shape
    array_data = array_data.reshape(1, 128)

    # Convert the numpy array to a CuPy array (transfers to GPU memory)
    cuda_array = cp.asarray(array_data)

    return cuda_array