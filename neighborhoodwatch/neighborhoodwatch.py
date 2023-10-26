import argparse
import neighborhoodwatch.cu_knn
from neighborhoodwatch.generate_dataset import generate_query_dataset, generate_base_dataset
from neighborhoodwatch.parquet_to_ivec_fvec import generate_files
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


def main():

    start_time = time.time()

    parser = argparse.ArgumentParser(
        description='nw (neighborhood watch) uses GPU acceleration to generate ground truth KNN datasets')
    parser.add_argument('query_count', type=int)
    parser.add_argument('base_count', type=int)
    parser.add_argument('dimensions', type=int)
    parser.add_argument('-k', '--k', type=int, default=100)
    parser.add_argument('--enable-memory-tuning', action='store_true', help='Enable memory tuning')
    parser.add_argument('--disable-memory-tuning', action='store_false', help='Disable memory tuning (useful for very small datasets)')


    args = parser.parse_args()

    rprint('', Markdown(f"**Neighborhood Watch** is generating brute force neighbors based on the wikipedia dataset for `{args.query_count}` queries and `{args.base_count}` base vectors"), '')

    rprint(Markdown("**Generating query dataset** "),'')
    section_time = time.time()
    query_filename = generate_query_dataset(args.query_count)
    rprint(Markdown(f"(**Duration**: `{time.time() - section_time:.2f} seconds out of {time.time() - start_time:.2f} seconds`)"))
    rprint(Markdown("---"),'')

    rprint(Markdown("**Generating base dataset** "),'')
    section_time = time.time()
    base_filename = generate_base_dataset(query_filename, args.base_count)
    rprint(Markdown(f"(**Duration**: `{time.time() - section_time:.2f} seconds out of {time.time() - start_time:.2f} seconds`)"))
    rprint(Markdown("---"),'')

    cleanup_partial_parquet()

    rprint(Markdown("**Computing knn** "),'')
    section_time = time.time()
    neighborhoodwatch.cu_knn.compute_knn(query_filename, args.query_count, base_filename, args.base_count,
                                         args.dimensions, args.enable_memory_tuning, args.k)
    rprint(Markdown(f"(**Duration**: `{time.time() - section_time:.2f} seconds out of {time.time() - start_time:.2f} seconds`)"))
    rprint(Markdown("---"),'')

    rprint(Markdown("**Merging indices and distances** "),'')
    section_time = time.time()
    neighborhoodwatch.merge.merge_indices_and_distances()
    rprint(Markdown(f"(**Duration**: `{time.time() - section_time:.2f} seconds out of {time.time() - start_time:.2f} seconds`)"))
    rprint(Markdown("---"),'')

    rprint(Markdown("**Generating ivec's and fvec's** "), '')
    section_time = time.time()
    query_vector_fvec, indices_ivec, distances_fvec, base_vector_fvec = neighborhoodwatch.parquet_to_ivec_fvec.generate_files('final_indices.parquet', base_filename, query_filename, 'final_distances.parquet', args.base_count,
                                                          args.query_count, args.k, args.dimensions)
    rprint(Markdown(f"(**Duration**: `{time.time() - section_time:.2f} seconds out of {time.time() - start_time:.2f} seconds`)"))
    rprint(Markdown("---"),'')

    #rprint(Markdown("**Validating ivec's and fvec's** "), '')
    #section_time = time.time()
    #neighborhoodwatch.parquet_to_ivec_fvec.validate_files(query_vector_fvec, indices_ivec, distances_fvec, base_vector_fvec)
    #rprint(Markdown(f"(**Duration**: `{time.time() - section_time:.2f} seconds out of {time.time() - start_time:.2f} seconds`)"))
    #rprint(Markdown("---"),'')


if __name__ == "__main__":
    if len(sys.argv) != 6:
        rprint(Markdown('Usage: `neighborhoodwatch.main(query_count base_count dimensions k)`'))
        sys.exit(1)

    query_count = int(sys.argv[2])
    base_count = int(sys.argv[4])
    dimensions = int(sys.argv[5])
    k = int(sys.argv[6])

    main(query_count, base_count, dimensions, k)