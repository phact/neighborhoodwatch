import sys
import cupy as cp
import h5py
import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm
from rich import print as rprint
from rich.markdown import Markdown
import struct
import os
from cuvs.distance import pairwise_distance
from cuvs.neighbors import brute_force
import torch

torch.device("cuda")


from neighborhoodwatch.nw_utils import *


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
    full_filename = get_full_filename(data_dir, input_parquet)
    table = pq.read_table(full_filename)
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

    df = table.to_pandas()
    return df


def generate_query_vectors_fvec(data_dir,
                                input_parquet,
                                query_count,
                                model_prefix,
                                dimensions,
                                normalize_embed,
                                column_names=None,
                                output_dtype=None):
    df = read_and_extract(data_dir, input_parquet, query_count, dimensions, column_names)

    if output_dtype is not None:
        output_fvec_base = f'{data_dir}/{model_prefix}_{dimensions}_{output_dtype}_query_vectors_{query_count}'
    else:
        output_fvec_base = f'{data_dir}/{model_prefix}_{dimensions}_query_vectors_{query_count}'

    if not normalize_embed:
        output_fvec = f'{output_fvec_base}.fvec'
    else:
        output_fvec = f'{output_fvec_base}_normalized.fvec'

    if not os.path.exists(output_fvec):
        write_ivec_fvec_from_dataframe(data_dir,
                                       output_fvec,
                                       df,
                                       'f',
                                       dimensions,
                                       model_prefix == 'voyage-3-large')
    else:
        print(f"File {output_fvec} already exists")

    return output_fvec, df


# Generate base_vectors.fvec file
def generate_base_vectors_fvec(data_dir,
                               input_parquet,
                               base_count,
                               model_prefix,
                               dimensions,
                               normalize_embed,
                               column_names=None,
                               output_dtype=None,):
    df = read_and_extract(data_dir, input_parquet, base_count, dimensions, column_names)

    if output_dtype is not None:
        output_fvec_base = f'{data_dir}/{model_prefix}_{dimensions}_{output_dtype}_base_vectors_{base_count}'
    else:
        output_fvec_base = f'{data_dir}/{model_prefix}_{dimensions}_base_vectors_{base_count}'

    if not normalize_embed:
        output_fvec = f'{output_fvec_base}.fvec'
    else:
        output_fvec = f'{output_fvec_base}_normalized.fvec'

    if not os.path.exists(output_fvec):
        write_ivec_fvec_from_dataframe(data_dir,
                                       output_fvec,
                                       df,
                                       'f',
                                       dimensions,
                                       model_prefix == 'voyage-3-large')
    else:
        print(f"File {output_fvec} already exists")

    return output_fvec, df


def generate_distances_fvec(data_dir,
                            input_parquet,
                            base_count,
                            query_count,
                            k,
                            model_prefix,
                            dimensions,
                            normalize_embed,
                            output_dtype=None):
    df = read_parquet_to_dataframe(data_dir, input_parquet)

    if output_dtype is not None:
        output_fvec_base = f'{data_dir}/{model_prefix}_{dimensions}_{output_dtype}_distances_b{base_count}_q{query_count}_k{k}'
    else:
        output_fvec_base = f'{data_dir}/{model_prefix}_{dimensions}_distances_b{base_count}_q{query_count}_k{k}'

    if not normalize_embed:
        output_fvec = f'{output_fvec_base}.fvec'
    else:
        output_fvec = f'{output_fvec_base}_normalized.fvec'

    if not os.path.exists(output_fvec):
        write_ivec_fvec_from_dataframe(data_dir,
                                       output_fvec,
                                       df,
                                       'f',
                                       k,
                                       model_prefix == 'voyage-3-large')
    else:
        print(f"File {output_fvec} already exists")

    return output_fvec


# Generate indices.ivec file
def generate_indices_ivec(data_dir,
                          input_parquet,
                          base_count,
                          query_count,
                          k,
                          model_prefix,
                          dimensions,
                          normalize_embed,
                          output_dtype=None):
    df = read_parquet_to_dataframe(data_dir, input_parquet)

    if output_dtype is not None:
        output_ivec_base = f'{data_dir}/{model_prefix}_{dimensions}_{output_dtype}_indices_b{base_count}_q{query_count}_k{k}'
    else:
        output_ivec_base = f'{data_dir}/{model_prefix}_{dimensions}_indices_b{base_count}_q{query_count}_k{k}'

    if not normalize_embed:
        output_ivec = f'{output_ivec_base}.ivec'
    else:
        output_ivec = f'{output_ivec_base}_normalized.ivec'

    if not os.path.exists(output_ivec):
        write_ivec_fvec_from_dataframe(data_dir,
                                       output_ivec,
                                       df,
                                       'i',
                                       k,
                                       model_prefix == 'voyage-3-large')
    else:
        print(f"File {output_ivec} already exists")

    return output_ivec


def generate_ivec_fvec_files(data_dir,
                             model_name,
                             dimensions,
                             base_vectors_parquet,
                             query_vectors_parquet,
                             final_indices_parquet,
                             final_distances_parquet,
                             base_count,
                             query_count,
                             k,
                             normalize_embed,
                             column_names=None,
                             output_dtype=None):
    model_prefix = get_model_prefix(model_name)

    rprint(Markdown("Generated files: "), '')

    query_vector_fvec, query_df = generate_query_vectors_fvec(data_dir,
                                                              query_vectors_parquet,
                                                              query_count,
                                                              model_prefix,
                                                              dimensions,
                                                              normalize_embed,
                                                              column_names,
                                                              output_dtype)
    rprint(Markdown(f"*`{query_vector_fvec}`* - "
                    f"query vector count: `{count_vectors(data_dir, query_vector_fvec)}`, "
                    f"dimensions: `{len(get_first_vector(data_dir, query_vector_fvec))}`,"
                    f"output_dtype: `{output_dtype}`"))

    base_vector_fvec, base_df = generate_base_vectors_fvec(data_dir,
                                                           base_vectors_parquet,
                                                           base_count,
                                                           model_prefix,
                                                           dimensions,
                                                           normalize_embed,
                                                           column_names,
                                                           output_dtype)
    rprint(Markdown(f"*`{base_vector_fvec}`* - "
                    f"base vector count: `{count_vectors(data_dir, base_vector_fvec)}`, "
                    f"dimensions: `{len(get_first_vector(data_dir, base_vector_fvec))}`,"
                    f"output_dtype: `{output_dtype}`"))

    indices_ivec = generate_indices_ivec(data_dir,
                                         final_indices_parquet,
                                         base_count,
                                         query_count,
                                         k,
                                         model_prefix,
                                         dimensions,
                                         normalize_embed,
                                         output_dtype)
    rprint(Markdown(f"*`{indices_ivec}`* - "
                    f"indices count: `{count_vectors(data_dir, indices_ivec)}`, "
                    f"k: `{len(get_first_vector(data_dir, indices_ivec))}`"))
    
    distances_fvec = generate_distances_fvec(data_dir,
                                             final_distances_parquet,
                                             base_count,
                                             query_count,
                                             k,
                                             model_prefix,
                                             dimensions,
                                             normalize_embed,
                                             output_dtype)
    rprint(Markdown(f"*`{distances_fvec}`* - "
                    f"distances count: `{count_vectors(data_dir, distances_fvec)}`, "
                    f"k: `{len(get_first_vector(data_dir, distances_fvec))}`"))
    
    return query_vector_fvec, query_df, base_vector_fvec, base_df, indices_ivec, distances_fvec


def generate_hdf5_file(data_dir,
                       model_prefix,
                       dimensions,
                       base_df_hdf5,
                       query_df_hdf5,
                       final_indices_parquet,
                       final_distances_parquet,
                       base_count, 
                       query_count, 
                       k,
                       normalize_embed,
                       output_dtype=None):
    if output_dtype is not None:
        hdf5_filename_base = f'{model_prefix}_{dimensions}_{output_dtype}_base_{base_count}_query_{query_count}_k{k}'
    else:
        hdf5_filename_base = f'{model_prefix}_{dimensions}_base_{base_count}_query_{query_count}_k{k}'

    if not normalize_embed:
        hdf5_filename = get_full_filename(data_dir, f'{hdf5_filename_base}.hdf5')
    else:
        hdf5_filename = get_full_filename(data_dir, f'{hdf5_filename_base}_normalized.hdf5')

    rprint(Markdown(f"Generated file: {hdf5_filename}"), '')

    write_hdf5(data_dir, base_df_hdf5, hdf5_filename, 'train')

    write_hdf5(data_dir, query_df_hdf5, hdf5_filename, 'test')

    df = read_parquet_to_dataframe(data_dir, final_distances_parquet)
    write_hdf5(data_dir, df, hdf5_filename, 'distances')

    df = read_parquet_to_dataframe(data_dir, final_indices_parquet)
    write_hdf5(data_dir, df, hdf5_filename, 'neighbors')


def write_hdf5(data_dir, df, filename, datasetname):
    data = df.to_numpy()
    full_filename = get_full_filename(data_dir, filename)
    with h5py.File(full_filename, 'a') as f:
        if datasetname in f:
            print(f"Dataset '{datasetname}' already exists in file '{full_filename}'")
        else:
            rprint(Markdown(f"writing to dataset '{datasetname}' - shape: {df.shape}"))
            f.create_dataset(datasetname, data=data)


def validate_files(data_dir, query_vector_fvec, base_vector_fvec, indices_ivec, distances_fvec, columns=None, input_parquet=None):
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
        for index in first_indexes:
            base_vector = get_nth_vector(data_dir, base_vector_fvec, index)
            similarity = dot_product(nth_query_vector, base_vector)
            distance = distance_vector[col]
            assert distance >= last_distance, f"Expected distance {distance} to be greater than last distance {last_distance}"
            last_distance = distance
            assert similarity-last_similarity <= 0 or np.isclose(similarity-last_similarity, 0, atol=1e-04), f"Expected similarity {similarity} to be greater than last similarity {last_similarity}\nDifference: {similarity - last_similarity}"
            last_similarity = similarity
            if not np.isclose((1-similarity), distance, atol=1e-04):
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
                full_distances, full_neighbors = brute_force.search(full_index, tuple_to_cuda_interface_array(nth_query_vector), 100000)
                this_index = cp.asarray(full_neighbors)[0].tolist().index(index)
                full_brute_force_distance = cp.asarray(full_distances)[0][this_index]

                search_index = brute_force.build(tuple_to_cuda_interface_array(base_vector), metric="cosine")
                distances, neighbors = brute_force.search(search_index, tuple_to_cuda_interface_array(nth_query_vector), 1)
                brute_force_distance = cp.asarray(distances)[0][0]

                this_index = first_indexes.index(index)


                distance_tensor = torch.matmul(torch.tensor(nth_query_vector, dtype=torch.float32), torch.tensor(base_vector, dtype=torch.float32))
                distances_tensor, indices_tensor = torch.topk(distance_tensor, 1, largest=True)

                total_mismatch_count += 1
                print(f"torch {1-distances_tensor.item()} full {full_brute_force_distance} brute {brute_force_distance} distance {distance} similarity {1-similarity}")
                #print(f"torch {1-distances_tensor.item()} brute {brute_force_distance} distance {distance} similarity {1-similarity}")
                print(f"Expected 1 - similarity: {1-similarity} to equal distance: {distance} for query vector {n} and base vector {index}")
                print(f"distance vs similarity diff: {distance - (1-similarity)}")
                print(f"cuvs vs similarity diff: {output - (1-similarity)}")
                print(f"brute vs similarity diff: {brute_force_distance - (1-similarity)}")
                print(f"full brute vs similarity diff: {full_brute_force_distance - (1-similarity)}")
                print(f"torch vs similarity diff: {distances_tensor.item()- similarity}")
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