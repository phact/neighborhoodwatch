import argparse

from neighborhoodwatch.generate_dataset import generate_query_dataset, generate_base_dataset
from neighborhoodwatch.parquet_to_format import generate_ivec_fvec_files, generate_hdf5_file, validate_files
from neighborhoodwatch.merge import merge_indices_and_distances
from neighborhoodwatch.cu_knn import compute_knn
from neighborhoodwatch.cu_knn_ds import compute_knn_ds
from neighborhoodwatch.nw_utils import *

import os
import sys
from rich import print as rprint
from rich.markdown import Markdown
import time


def cleanup_partial_parquet():
    for filename in os.listdir():
        if filename.startswith("distances") or filename.startswith("indices") or filename.startswith("final"):
            os.remove(filename)


class KeepLineBreaksFormatter(argparse.RawTextHelpFormatter):
    pass


def main():
    start_time = time.time()

    parser = argparse.ArgumentParser(
        description="""nw (neighborhood watch) uses GPU acceleration to generate ground truth KNN datasets""",
        epilog="""
Some example commands:\n
    nw 1000 10000 -k 100 -m 'textembedding-gecko' --disable-memory-tuning
    nw 1000 10000 -k 100 -m 'intfloat/e5-large-v2' --disable-memory-tuning
    nw 1000 10000 -k 100 -m 'intfloat/e5-small-v2' --disable-memory-tuning
    nw 1000 10000 -k 100 -m 'intfloat/e5-base-v2' --disable-memory-tuning
        """, formatter_class=KeepLineBreaksFormatter)
    parser.add_argument('query_count', type=int, help="number of query vectors to generate")
    parser.add_argument('base_count', type=int, help="number of base vectors to generate")
    parser.add_argument('-m', '--model_name', type=str,
                        help='model name to use for generating embeddings, i.e. text-embedding-ada-002, textembedding-gecko, or intfloat/e5-large-v2')
    parser.add_argument('-ods', '--output_dimension_size', type=int,
                        help="Output dimension size; can be different from model's default dimension size!")
    parser.add_argument('-odt', '--output_dtype', type=str, default='float',
                        help="Output dtype; currently only valid for VoyageAI model!")
    parser.add_argument('-k', '--k', type=int, default=100, help='number of neighbors to compute per query vector')
    parser.add_argument('--data-dir', type=str, default='knn_dataset',
                        help='Directory to store the generated data (default: knn_dataset)')
    parser.add_argument('--norm-embed', action=argparse.BooleanOptionalAction, default=False,
                        help='Normalize the returned model embeddings (default: False)')
    parser.add_argument('--use-dataset-api', action=argparse.BooleanOptionalAction, default=False,
                        help='Use \'pyarrow.dataset\' API to read the dataset (default: True). Recommended for large datasets.')
    parser.add_argument('--gen-hdf5', action=argparse.BooleanOptionalAction, default=True,
                        help='Generate hdf5 files (default: True)')
    parser.add_argument('--post-validation', action=argparse.BooleanOptionalAction, default=False,
                        help='Validate the generated files (default: False)')
    parser.add_argument('--enable-memory-tuning', action='store_true', help='Enable memory tuning')
    parser.add_argument('--disable-memory-tuning', action='store_false',
                        help='Disable memory tuning (useful for very small datasets)')

    args = parser.parse_args()

    if not check_dataset_exists_remote():
        rprint(Markdown(f"The specified wikipedia dataset configuration/subset does not exist: {BASE_CONFIG}"))
        sys.exit(1)

    rprint('', Markdown(f"""**Neighborhood Watch** is generating brute force neighbors based on the wikipedia dataset with the following specification:\n
--- dataset/model specification ---\n
* source dataset version: `{BASE_DATASET}-{BASE_CONFIG}`\n
* query count: `{args.query_count}`\n
* base vector count: `{args.base_count}`\n
* model name: `{args.model_name}`\n
* output dimension size: `{args.output_dimension_size}`\n
* output dtype: `{args.output_dtype}` (currently only relevant for VoyageAI models)\n
* K: `{args.k}`\n
* normalize embeddings: `{args.norm_embed}`\n
--- behavior specification ---\n
* use dataset API: `{args.use_dataset_api}`\n
* generated hdf5 file: `{args.gen_hdf5}`\n
* post validation: `{args.post_validation}`\n
* enable memory tuning: `{args.enable_memory_tuning}`
"""))
    rprint('', Markdown("---"))

    model_prefix = get_model_prefix(args.model_name)
    data_dir = f"{args.data_dir}/{model_prefix}/q{args.query_count}_b{args.base_count}_k{args.k}"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    output_dimension = get_embedding_size(args.model_name, args.output_dimension_size)
    output_dtype = None
    if args.model_name.startswith('voyage'):
        output_dtype = args.output_dtype
        assert output_dtype in ['float', 'int8', 'uint8', 'binary', 'ubinary']

    rprint(Markdown("**Generating query dataset ......** "), '')
    section_time = time.time()
    query_filename = generate_query_dataset(data_dir,
                                            args.model_name,
                                            args.query_count,
                                            output_dimension,
                                            args.norm_embed,
                                            output_dtype)

    rprint(Markdown(
        f"(**Duration**: `{time.time() - section_time:.2f} seconds out of {time.time() - start_time:.2f} seconds`)"))
    rprint(Markdown("---"), '')

    rprint(Markdown("**Generating base dataset ......** "), '')
    section_time = time.time()
    base_filename = generate_base_dataset(data_dir,
                                          args.model_name,
                                          query_filename,
                                          args.base_count,
                                          output_dimension,
                                          args.norm_embed,
                                          output_dtype)
    rprint(Markdown(
        f"(**Duration**: `{time.time() - section_time:.2f} seconds out of {time.time() - start_time:.2f} seconds`)"))
    rprint(Markdown("---"), '')

    cleanup_partial_parquet()

    rprint(Markdown("**Computing knn ......** "), '')

    if output_dtype is not None:
        final_indecies_filename_base = f"{model_prefix}_{output_dimension}_{output_dtype}_final_indices_query{args.query_count}_k{args.k}"
        final_distances_filename_base = f"{model_prefix}_{output_dimension}_{output_dtype}_final_distances_query{args.query_count}_k{args.k}"
    else:
        final_indecies_filename_base = f"{model_prefix}_{output_dimension}_final_indices_query{args.query_count}_k{args.k}"
        final_distances_filename_base = f"{model_prefix}_{output_dimension}_final_distances_query{args.query_count}_k{args.k}"

    if not args.norm_embed:
        final_indecies_filename = get_full_filename(data_dir, f"{final_indecies_filename_base}.parquet")
        final_distances_filename = get_full_filename(data_dir, f"{final_distances_filename_base}.parquet")
    else:
        final_indecies_filename = get_full_filename(data_dir, f"{final_indecies_filename_base}_normalized.parquet")
        final_distances_filename = get_full_filename(data_dir, f"{final_distances_filename_base}_normalized.parquet")

    section_time = time.time()
    if args.use_dataset_api:
        compute_knn_ds(data_dir,
                       output_dimension,
                       query_filename,
                       args.query_count,
                       base_filename,
                       args.base_count,
                       final_indecies_filename,
                       final_distances_filename,
                       args.enable_memory_tuning,
                       args.k)
    else:
        compute_knn(data_dir,
                    output_dimension,
                    query_filename,
                    args.query_count,
                    base_filename,
                    args.base_count,
                    final_indecies_filename,
                    final_distances_filename,
                    args.enable_memory_tuning,
                    args.k,
                    ignore_dimension_check=(model_prefix == 'voyage-3-large'))
    rprint(Markdown(
        f"(**Duration**: `{time.time() - section_time:.2f} seconds out of {time.time() - start_time:.2f} seconds`)"))
    rprint(Markdown("---"), '')

    rprint(Markdown("**Merging indices and distances ......** "), '')
    section_time = time.time()
    merge_indices_and_distances(data_dir,
                                model_prefix,
                                output_dimension,
                                final_indecies_filename,
                                final_distances_filename,
                                args.k,
                                output_dtype)
    rprint(Markdown(
        f"(**Duration**: `{time.time() - section_time:.2f} seconds out of {time.time() - start_time:.2f} seconds`)"))
    rprint(Markdown("---"), '')

    rprint(Markdown("**Generating ivec's and fvec's ......** "), '')
    section_time = time.time()
    query_vector_fvec, query_df_hdf5, base_vector_fvec, base_df_hdf5, indices_ivec, distances_fvec = \
        generate_ivec_fvec_files(data_dir,
                                 args.model_name,
                                 output_dimension,
                                 base_filename,
                                 query_filename,
                                 final_indecies_filename,
                                 final_distances_filename,
                                 args.base_count,
                                 args.query_count,
                                 args.k,
                                 args.norm_embed,
                                 column_names=None,
                                 output_dtype=output_dtype)
    rprint(Markdown(
        f"(**Duration**: `{time.time() - section_time:.2f} seconds out of {time.time() - start_time:.2f} seconds`)"))
    rprint(Markdown("---"), '')

    if args.gen_hdf5:
        rprint(Markdown("**Generating hdf5 ......** "), '')
        section_time = time.time()
        generate_hdf5_file(data_dir,
                           model_prefix,
                           output_dimension,
                           base_df_hdf5,
                           query_df_hdf5,
                           final_indecies_filename,
                           final_distances_filename,
                           args.base_count,
                           args.query_count,
                           args.k,
                           args.norm_embed,
                           output_dtype)
        rprint(Markdown(
            f"(**Duration**: `{time.time() - section_time:.2f} seconds out of {time.time() - start_time:.2f} seconds`)"))
        rprint(Markdown("---"), '')

    if args.post_validation:
        yes_no_str = input(
            "Dataset validation is enabled and it may take a very long time to finish. Do you want to continue? (y/n/yes/no): ")
        if yes_no_str == 'y' or yes_no_str == 'yes':
            rprint(Markdown("**Validating ivec's and fvec's ......** "), '')
            section_time = time.time()
            validate_files(data_dir,
                           query_vector_fvec,
                           base_vector_fvec,
                           indices_ivec,
                           distances_fvec)
            rprint(Markdown(
                f"(**Duration**: `{time.time() - section_time:.2f} seconds out of {time.time() - start_time:.2f} seconds`)"))
            rprint(Markdown("---"), '')


if __name__ == "__main__":
    main()