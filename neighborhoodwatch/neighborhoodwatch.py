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
    parser.add_argument('-rd', '--reduced_dimension_size', type=int,
                        help='Reduced (output) dimension size. Only supported in models (e.g. OpenAI text-embedding-3-xxx) that have this feature. Ignored otherwise!')
    parser.add_argument('-k', '--k', type=int, default=100, help='number of neighbors to compute per query vector')
    parser.add_argument('--data_dir', type=str, default='knn_dataset',
                        help='Directory to store the generated data (default: knn_dataset)')
    parser.add_argument('--skip-zero-vec', action=argparse.BooleanOptionalAction, default=True,
                        help='Skip generating zero vectors when failing to retrieve the embedding (default: True)')
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
* reduced (output) dimension size: `{args.reduced_dimension_size} (Only relevant with OpenAI latest embedding models: text-embedding-3-small/large`\n
--- behavior specification ---\n
* skip zero vector: `{args.skip_zero_vec}`\n
* use dataset API: `{args.use_dataset_api}`\n
* generated hdf5 file: `{args.gen_hdf5}`\n
* post validation: `{args.post_validation}`\n
* enable memory tuning: `{args.enable_memory_tuning}`
"""))
    rprint('', Markdown("---"))

    try:
        reduced_dimension = get_embedding_size(args.model_name, args.reduced_dimension_size)
    except:
        rprint(Markdown(
            f"Unsupported model name ({args.model_name}) or can't determine the dimension size for it. "
            f"Please double check the input model name!"))
        sys.exit(2)

    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

    rprint(Markdown("**Generating query dataset ......** "), '')
    section_time = time.time()
    query_filename = generate_query_dataset(args.data_dir,
                                            args.query_count,
                                            args.model_name,
                                            reduced_dimension,
                                            args.skip_zero_vec)

    rprint(Markdown(
        f"(**Duration**: `{time.time() - section_time:.2f} seconds out of {time.time() - start_time:.2f} seconds`)"))
    rprint(Markdown("---"), '')

    rprint(Markdown("**Generating base dataset ......** "), '')
    section_time = time.time()
    base_filename = generate_base_dataset(args.data_dir,
                                          query_filename,
                                          args.base_count,
                                          args.model_name,
                                          reduced_dimension,
                                          args.skip_zero_vec)
    rprint(Markdown(
        f"(**Duration**: `{time.time() - section_time:.2f} seconds out of {time.time() - start_time:.2f} seconds`)"))
    rprint(Markdown("---"), '')

    cleanup_partial_parquet()

    rprint(Markdown("**Computing knn ......** "), '')
    section_time = time.time()
    if args.use_dataset_api:
        compute_knn_ds(args.data_dir,
                       args.model_name,
                       reduced_dimension,
                       query_filename,
                       args.query_count,
                       base_filename,
                       args.base_count,
                       args.enable_memory_tuning,
                       args.k)
    else:
        compute_knn(args.data_dir,
                    args.model_name,
                    reduced_dimension,
                    query_filename,
                    args.query_count,
                    base_filename,
                    args.base_count,
                    args.enable_memory_tuning,
                    args.k)
    rprint(Markdown(
        f"(**Duration**: `{time.time() - section_time:.2f} seconds out of {time.time() - start_time:.2f} seconds`)"))
    rprint(Markdown("---"), '')

    rprint(Markdown("**Merging indices and distances ......** "), '')
    section_time = time.time()
    merge_indices_and_distances(args.data_dir,
                                args.model_name,
                                reduced_dimension,
                                args.k)
    rprint(Markdown(
        f"(**Duration**: `{time.time() - section_time:.2f} seconds out of {time.time() - start_time:.2f} seconds`)"))
    rprint(Markdown("---"), '')

    model_prefix = get_model_prefix(args.model_name)

    rprint(Markdown("**Generating ivec's and fvec's ......** "), '')
    section_time = time.time()
    query_vector_fvec, base_vector_fvec, indices_ivec, distances_fvec = \
        generate_ivec_fvec_files(args.data_dir,
                                 args.model_name,
                                 reduced_dimension,
                                 base_filename,
                                 query_filename,
                                 f"{model_prefix}_{reduced_dimension}_final_indices_k{args.k}.parquet",
                                 f"{model_prefix}_{reduced_dimension}_final_distances_k{args.k}.parquet",
                                 args.base_count,
                                 args.query_count,
                                 args.k)
    rprint(Markdown(
        f"(**Duration**: `{time.time() - section_time:.2f} seconds out of {time.time() - start_time:.2f} seconds`)"))
    rprint(Markdown("---"), '')

    if args.gen_hdf5:
        rprint(Markdown("**Generating hdf5 ......** "), '')
        section_time = time.time()
        generate_hdf5_file(args.data_dir,
                           args.model_name,
                           reduced_dimension,
                           base_filename,
                           query_filename,
                           f"{model_prefix}_{reduced_dimension}_final_indices_k{args.k}.parquet",
                           f"{model_prefix}_{reduced_dimension}_final_distances_k{args.k}.parquet",
                           args.base_count,
                           args.query_count,
                           args.k)
        rprint(Markdown(
            f"(**Duration**: `{time.time() - section_time:.2f} seconds out of {time.time() - start_time:.2f} seconds`)"))
        rprint(Markdown("---"), '')

    if args.post_validation:
        yes_no_str = input(
            "Dataset validation is enabled and it may take a very long time to finish. Do you want to continue? (y/n/yes/no): ")
        if yes_no_str == 'y' or yes_no_str == 'yes':
            rprint(Markdown("**Validating ivec's and fvec's ......** "), '')
            section_time = time.time()
            validate_files(args.data_dir,
                           query_vector_fvec,
                           base_vector_fvec,
                           indices_ivec,
                           distances_fvec)
            rprint(Markdown(
                f"(**Duration**: `{time.time() - section_time:.2f} seconds out of {time.time() - start_time:.2f} seconds`)"))
            rprint(Markdown("---"), '')


if __name__ == "__main__":
    if len(sys.argv) != 6:
        rprint(Markdown('Usage: `neighborhoodwatch.main(query_count base_count dimensions k)`'))
        sys.exit(1)

    query_count = int(sys.argv[2])
    base_count = int(sys.argv[4])
    dimensions = int(sys.argv[5])
    k = int(sys.argv[6])

    main(query_count, base_count, dimensions, k)
