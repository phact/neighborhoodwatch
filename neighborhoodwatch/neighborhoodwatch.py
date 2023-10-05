import argparse
import neighborhoodwatch.cu_knn
from neighborhoodwatch.parquet_to_ivec_fvec import generate_files
from neighborhoodwatch.merge import merge_indices_and_distances
import sys
from rich import print as rprint
from rich.markdown import Markdown
import time



def main():

    start_time = time.time()

    parser = argparse.ArgumentParser(
        description='nw (neighborhood watch) uses GPU acceleration to generate ground truth KNN datasets')
    parser.add_argument('query_filename', type=str)
    parser.add_argument('query_count', type=int)
    parser.add_argument('sorted_data_filename', type=str)
    parser.add_argument('base_count', type=int)
    parser.add_argument('dimensions', type=int)
    parser.add_argument('-k', '--k', type=int, default=100)
    parser.add_argument('--enable-memory-tuning', action='store_true', help='Enable a particular feature')
    parser.add_argument('--disable-memory-tuning', action='store_false', help='Disable a particular feature')


    args = parser.parse_args()

    rprint('', Markdown(f"**Neighborhood Watch** is generating brute force neighbors with queries `{args.query_filename}` and base vectors `{args.sorted_data_filename}` \n for `{args.query_count}` queries and `{args.base_count}` base vectors"), '')

    rprint(Markdown("**Computing knn** "),'')
    section_time = time.time()
    neighborhoodwatch.cu_knn.compute_knn(args.query_filename, args.query_count, args.sorted_data_filename, args.base_count,
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
    neighborhoodwatch.parquet_to_ivec_fvec.generate_files('final_indices.parquet', args.sorted_data_filename, args.base_count,
                                                          args.query_count)
    rprint(Markdown(f"(**Duration**: `{time.time() - section_time:.2f} seconds out of {time.time() - start_time:.2f} seconds`)"))
    rprint(Markdown("---"),'')


if __name__ == "__main__":
    if len(sys.argv) != 6:
        rprint(Markdown('Usage: `neighborhoodwatch.main(query_filename query_count sorted_data_filename base_count dimensions k input_parquet_filename)`'))
        sys.exit(1)

    query_filename = sys.argv[1]
    query_count = int(sys.argv[2])
    sorted_data_filename = sys.argv[3]
    base_count = int(sys.argv[4])
    dimensions = int(sys.argv[5])
    k = int(sys.argv[6])

    main(query_filename, query_count, sorted_data_filename, base_count, dimensions, k)