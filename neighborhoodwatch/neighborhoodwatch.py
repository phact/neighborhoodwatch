import argparse

import neighborhoodwatch.cu_knn
from neighborhoodwatch.generate_dataset import generate_query_dataset, generate_base_dataset
from neighborhoodwatch.parquet_to_format import generate_ivec_fvec_files, generate_hdf5_file
from neighborhoodwatch.merge import merge_indices_and_distances
import sys
from rich import print as rprint
from rich.markdown import Markdown
import time
import os


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
    nw 10000 10000 1024 -k 100 -m 'intfloat/e5-large-v2' --disable-memory-tuning
    nw 10000 10000 768 -k 100 -m 'textembedding-gecko' --disable-memory-tuning
    nw 10000 10000 384 -k 100 -m 'intfloat/e5-small-v2' --disable-memory-tuning
    nw 10000 10000 768 -k 100 -m 'intfloat/e5-base-v2' --disable-memory-tuning
        """, formatter_class=KeepLineBreaksFormatter)
    parser.add_argument('query_count', type=int, help="number of query vectors to generate")
    parser.add_argument('base_count', type=int, help="number of base vectors to generate")
    parser.add_argument('dimensions', type=int, help="number of dimensions for the embedding model")
    parser.add_argument('-k', '--k', type=int, default=100, help='number of neighbors to compute per query vector')
    parser.add_argument('-m', '--model_name', type=str, default='ada-002', help='model name to use for generating embeddings, i.e. ada-002, textembedding-gecko, or intfloat/e5-large-v2')
    parser.add_argument('-d', '--data_dir', type=str, default='knn_dataset', help='Directory to store the generated data (default: knn_dataset)')
    parser.add_argument('--enable-memory-tuning', action='store_true', help='Enable memory tuning')
    parser.add_argument('--disable-memory-tuning', action='store_false', help='Disable memory tuning (useful for very small datasets)')
    parser.add_argument('--gen-hdf5', action=argparse.BooleanOptionalAction, default=True, help='Generate hdf5 files (default: True)')
    parser.add_argument('--validation', action=argparse.BooleanOptionalAction, default=False, help='Validate the generated files (default: False)')
    
    args = parser.parse_args()

    rprint('', Markdown(f"""**Neighborhood Watch** is generating brute force neighbors based on the wikipedia dataset with the following specification:\n
* query count: `{args.query_count}`\n
* base vector count: `{args.base_count}`\n
* model name: `{args.model_name}`\n
* dimension size: `{args.dimensions}`"""), '')

    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

    rprint(Markdown("**Generating query dataset** "),'')
    section_time = time.time()
    query_filename = generate_query_dataset(args.data_dir, args.query_count, args.model_name)
    rprint(Markdown(f"(**Duration**: `{time.time() - section_time:.2f} seconds out of {time.time() - start_time:.2f} seconds`)"))
    rprint(Markdown("---"),'')

    rprint(Markdown("**Generating base dataset** "),'')
    section_time = time.time()
    base_filename = generate_base_dataset(args.data_dir, query_filename, args.base_count, args.model_name)
    rprint(Markdown(f"(**Duration**: `{time.time() - section_time:.2f} seconds out of {time.time() - start_time:.2f} seconds`)"))
    rprint(Markdown("---"),'')

    cleanup_partial_parquet()

    rprint(Markdown("**Computing knn** "),'')
    section_time = time.time()
    neighborhoodwatch.cu_knn.compute_knn(args.data_dir, query_filename, args.query_count, base_filename, args.base_count,
                                         args.dimensions, args.enable_memory_tuning, args.k)
    rprint(Markdown(f"(**Duration**: `{time.time() - section_time:.2f} seconds out of {time.time() - start_time:.2f} seconds`)"))
    rprint(Markdown("---"),'')

    rprint(Markdown("**Merging indices and distances** "),'')
    section_time = time.time()
    neighborhoodwatch.merge.merge_indices_and_distances(args.data_dir)
    rprint(Markdown(f"(**Duration**: `{time.time() - section_time:.2f} seconds out of {time.time() - start_time:.2f} seconds`)"))
    rprint(Markdown("---"),'')

    rprint(Markdown("**Generating ivec's and fvec's** "), '')
    section_time = time.time()
    query_vector_fvec, indices_ivec, distances_fvec, base_vector_fvec = \
        neighborhoodwatch.parquet_to_format.generate_ivec_fvec_files(args.data_dir,
                                                                     'final_indices.parquet',
                                                                     base_filename,
                                                                     query_filename,
                                                                     'final_distances.parquet',
                                                                     args.base_count,
                                                                     args.query_count,
                                                                     args.k,
                                                                     args.dimensions,
                                                                     args.model_name)
    rprint(Markdown(f"(**Duration**: `{time.time() - section_time:.2f} seconds out of {time.time() - start_time:.2f} seconds`)"))
    rprint(Markdown("---"),'')

    if args.gen_hdf5:
        rprint(Markdown("**Generating hdf5** "), '')
        section_time = time.time()
        neighborhoodwatch.parquet_to_format.generate_hdf5_file(args.data_dir,
                                                               'final_indices.parquet',
                                                                base_filename,
                                                                query_filename,
                                                                'final_distances.parquet',
                                                                args.base_count,
                                                                args.query_count,
                                                                args.k,
                                                                args.dimensions,
                                                                args.model_name)
        rprint(Markdown(f"(**Duration**: `{time.time() - section_time:.2f} seconds out of {time.time() - start_time:.2f} seconds`)"))
        rprint(Markdown("---"),'')

    if args.validation:
        yes_no_str = input("Dataset validation is enabled and it may take a very long time to finish. Do you want to continue? (y/n/yes/no): ")
        if yes_no_str == 'y' or yes_no_str == 'yes':
            rprint(Markdown("**Validating ivec's and fvec's** "), '')
            section_time = time.time()
            neighborhoodwatch.parquet_to_format.validate_files(query_vector_fvec, indices_ivec, distances_fvec, base_vector_fvec)
            rprint(Markdown(f"(**Duration**: `{time.time() - section_time:.2f} seconds out of {time.time() - start_time:.2f} seconds`)"))
            rprint(Markdown("---"),'')


if __name__ == "__main__":
    if len(sys.argv) != 6:
        rprint(Markdown('Usage: `neighborhoodwatch.main(query_count base_count dimensions k)`'))
        sys.exit(1)

    query_count = int(sys.argv[2])
    base_count = int(sys.argv[4])
    dimensions = int(sys.argv[5])
    k = int(sys.argv[6])

    main(query_count, base_count, dimensions, k)