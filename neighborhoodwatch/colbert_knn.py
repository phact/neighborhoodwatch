import argparse
import requests
import datasets
import tarfile
import csv
import cudf
import cupy as cp
from tqdm import tqdm
import pyarrow as pa
import pandas as pd
import pyarrow.parquet as pq
import os
import sys
import time
import gc
import math
import numpy as np
from pylibraft.neighbors.brute_force import knn
import rmm

from pathlib import Path
from torch import Tensor
from tqdm import tqdm

from neighborhoodwatch.cu_knn import stream_cudf_to_parquet, cleanup, tune_memory, prep_table
from neighborhoodwatch.generate_dataset import EmbeddingGenerator, split_into_sentences, get_embeddings_from_map, \
    is_zero_embedding, ParquetStreamer
from neighborhoodwatch.merge import merge_indices_and_distances
from neighborhoodwatch.neighborhoodwatch import KeepLineBreaksFormatter
from neighborhoodwatch.nw_utils import *
from rich import print as rprint
from rich.markdown import Markdown
from colbert.infra.config import ColBERTConfig
from colbert.modeling.checkpoint import Checkpoint
from colbert.indexing.collection_encoder import CollectionEncoder

from neighborhoodwatch.parquet_to_format import generate_ivec_fvec_files

pa.jemalloc_set_decay_ms(0)


class ColbertPreTrainedEmbeddingGenerator(EmbeddingGenerator):
    def __init__(self, model_name="colbertv2.0", download_pretrained_model=False):
        # - In Colbert model, each input text token (NOTE not input text itself) corresponds to a "token" vector
        #   with a smaller dimension size (e.g. 128). The total "dimension" size of the entire input text is
        #   not fixed (depending on the tokens of the input text).
        # - The number here refers to the "token" dimension size.
        super().__init__(model_name, 128, 256)

        self.local_model_base_dir = f".pretrained"
        Path(self.local_model_base_dir).mkdir(parents=True, exist_ok=True)

        # Download the pretrained model if necessary
        self.download_colbert_model(download_pretrained_model)

        # Get Colbert collection encoder
        self.colbert_cfg = ColBERTConfig(checkpoint=f"{self.local_model_base_dir}/{self.model_name}")
        self.colbert_cp = Checkpoint(self.colbert_cfg.checkpoint, colbert_config=self.colbert_cfg)
        self.encoder = CollectionEncoder(self.colbert_cfg, self.colbert_cp)

    def download_colbert_model(self, force_download):
        local_model_file = f".pretrained/{self.model_name}.tar.gz"
        try:
            if not os.path.exists(local_model_file) or force_download is True:
                print(f"   download Colbert pre-trained model ...")
                session = requests.Session()

                model_url = "https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/colbertv2.0.tar.gz"
                response = session.get(url=model_url, stream=True)
                with open(local_model_file, 'wb') as fh:
                    # Walk through the request response in chunks of 1024 * 1024 * 1024 bytes, so 1 GB
                    for chunk in response.iter_content(1024 * 1024 * 1024):
                        # Write the chunk to the file
                        fh.write(chunk)

            tar = tarfile.open(local_model_file)
            tar.extractall(path=self.local_model_base_dir)
        except Exception as e:
            print("   ** failed to download the model", e)
            raise RuntimeError(e)

    def generate_embedding(self, text):
        # Ensure the text is a list of sentences
        if isinstance(text, str):
            text = [text]

        # Generate embeddings
        embeddings = []
        for t in text:
            # For each input text, the returned embedding is an 2-dimensional array with shape like <changing_num>x128,
            # aka, it contains a series of token-vectors of the size 128.
            tensor_embeddings, token_cnt = self.encoder.encode_passages(text)
            np_embed = Tensor.numpy(tensor_embeddings)
            embeddings.append(np_embed.flatten())

        return embeddings


def process_source_dataset(streamer,
                           dataset,
                           model_name,
                           input_dimensions,
                           row_count,
                           token_count,
                           column_to_embed,
                           skip_zero_vec=True,
                           remove_duplicates=False):
    row_batch_size = min(row_count, 2000)
    token_embedding_output_batch_size = 10000

    embedding_counter = 0
    token_embedding_counter = 0
    detected_zero_embedding_cnt = 0
    skipped_zero_embedding_cnt = 0

    cur_row = 0
    total_sentence_cnt_in_batch = 0
    total_sentence_cnt = 0
    total_duplicate_removed_cnt = 0

    text_map = []
    embedding_array = []

    if model_name == "colbertv2.0":
        generator = ColbertPreTrainedEmbeddingGenerator()
    else:
        assert False, f"Unsupported model name: {model_name}"

    def break_out_of_loop():
        return cur_row >= row_count or token_embedding_counter >= token_count

    for row in dataset:
        if break_out_of_loop():
            break

        cur_row += 1

        last_row = (cur_row == len(dataset) - 1)
        sentence_list = split_into_sentences(row[column_to_embed])
        text_map.append([cur_row, sentence_list])
        total_sentence_cnt_in_batch += len(sentence_list)
        total_sentence_cnt += len(sentence_list)

        embedding_tuple_list, detected_zero_embedding_cnt = get_embeddings_from_map(text_map, generator)
        text_map.clear()
        total_sentence_cnt_in_batch = 0

        # for embedding_tuple in tqdm(embedding_tuple_list):
        for embedding_tuple in embedding_tuple_list:
            embedding_list = embedding_tuple[1]

            for embedding in embedding_list:
                assert len(embedding) % input_dimensions == 0, \
                    f"embedding dimension length {len(embedding)} is not a multiple of token embedding size {input_dimensions}"

                if skip_zero_vec and is_zero_embedding(embedding):
                    skipped_zero_embedding_cnt += 1
                else:
                    token_embedding_list = [embedding[i * input_dimensions:(i + 1) * input_dimensions] for i in
                                           range((len(embedding) + input_dimensions - 1) // input_dimensions)]

                    embedding_counter += 1
                    for token_embedding in token_embedding_list:
                        embedding_array.append(token_embedding)
                        token_embedding_counter += 1

                        if (token_embedding_counter % token_embedding_output_batch_size == 0) and len(embedding_array) > 0:
                            cnt1 = len(embedding_array)
                            if remove_duplicates:
                                embedding_array = list(set(map(tuple, embedding_array)))
                            cnt2 = len(embedding_array)
                            total_duplicate_removed_cnt += (cnt1 - cnt2)
                            streamer.stream_to_parquet_without_src_metadata(embedding_array)
                            embedding_array.clear()

                        if break_out_of_loop():
                            break

                if break_out_of_loop():
                    break

            if break_out_of_loop():
                break

    if len(embedding_array) > 0:
        streamer.stream_to_parquet_without_src_metadata(embedding_array)
        embedding_array.clear()

    return (cur_row,
            embedding_counter,
            token_embedding_counter,
            detected_zero_embedding_cnt,
            skipped_zero_embedding_cnt,
            total_sentence_cnt,
            total_duplicate_removed_cnt)


def process_knn_computation(data_dir,
                            model_name,
                            input_dimensions,
                            query_filename,
                            base_filename,
                            mem_tune=False,
                            initial_batch_size=1000000,
                            max_memory_threshold=0.1,
                            k=100,
                            split=True):
    rmm.mr.set_current_device_resource(rmm.mr.PoolMemoryResource(rmm.mr.ManagedMemoryResource()))
    model_prefix = get_model_prefix(model_name)

    print(f"-- prepare query source table for brute-force KNN computation.")
    query_table = pq.read_table(get_full_filename(data_dir, query_filename))
    print(f"   query_table.shape: {query_table.shape}")
    print(f"-- prepare base source table for brute-force KNN computation.")
    base_table = pq.read_table(get_full_filename(data_dir, base_filename))
    print(f"   base_table.shape: {base_table.shape}")

    batch_size = initial_batch_size
    if mem_tune:
        batch_size = tune_memory(base_table, batch_size, max_memory_threshold, rmm)

    batch_count = math.ceil(len(base_table) / batch_size)
    assert (len(base_table) % batch_size == 0) or k <= (
            len(base_table) % batch_size), f"Cannot generate k of {k} with only {len(base_table)} rows and batch_size of {batch_size}."

    # for start in tqdm(range(0, batch_count)):
    for start in range(0, batch_count):
        batch_offset = start * batch_size
        batch_length = batch_size if start != batch_count - 1 else len(base_table) - batch_offset

        dataset_batch = base_table.slice(batch_offset, batch_length)
        df = cudf.DataFrame.from_arrow(dataset_batch)

        df_numeric = df.select_dtypes(['float32', 'float64'])
        cleanup(df)

        dataset = cp.from_dlpack(df_numeric.to_dlpack()).copy(order='C')

        # Split the DataFrame into parts (floor division)
        # TODO: pull out this variable
        # split_factor = 50
        split_factor = 1
        splits = split_factor * batch_count
        rows_per_split = len(query_table) // splits

        distances = cudf.DataFrame()
        indices = cudf.DataFrame()
        if split:
            # for i in tqdm(range(splits)):
            for i in range(splits):
                offset = i * rows_per_split
                length = rows_per_split if i != splits - 1 else len(query_table) - offset  # To handle the last chunk

                query_batch = query_table.slice(offset, length)

                df1 = cudf.DataFrame.from_arrow(query_batch)
                df_numeric1 = df1.select_dtypes(['float32', 'float64'])

                cleanup(df1)
                query = cp.from_dlpack(df_numeric1.to_dlpack()).copy(order='C')

                assert (k <= len(dataset))

                cupydistances1, cupyindices1 = knn(dataset.astype(np.float32),
                                                   query.astype(np.float32),
                                                   k)

                distances1 = cudf.from_pandas(pd.DataFrame(cp.asarray(cupydistances1).get()))
                # add batch_offset to indices
                indices1 = cudf.from_pandas(pd.DataFrame((cp.asarray(cupyindices1) + batch_offset).get()))

                distances = cudf.concat([distances, distances1], ignore_index=True)
                indices = cudf.concat([indices, indices1], ignore_index=True)

            distances.columns = distances.columns.astype(str)
            indices.columns = indices.columns.astype(str)

        assert (len(distances) == len(query_table))
        assert (len(indices) == len(query_table))

        stream_cudf_to_parquet(distances, 1000000,
                               f'{data_dir}/{model_prefix}_{input_dimensions}_distances{start}.parquet')
        stream_cudf_to_parquet(indices, 1000000, f'{data_dir}/{model_prefix}_{input_dimensions}_indices{start}.parquet')

        cleanup(df_numeric, distances, indices, dataset)


def print_dataset_info(source_dataset_name,
                       row_count,
                       token_count,
                       actual_row_cnt,
                       actual_sentence_cnt,
                       actual_embedding_counter,
                       actual_token_embedding_counter,
                       detected_zero_embedding_cnt,
                       skip_zero_vec,
                       skipped_zero_embedding_cnt,
                       remove_duplicates,
                       duplicate_removed_cnt):
    print(f"=================================================")
    print(f"== '{source_dataset_name}' source dataset stats")
    print(f"== ----------------------------------------------")
    print(f"== Expected total count of source data rows: {row_count}")
    print(f"== Expected total count of source data tokens: {token_count}")
    print(f"== Total count of source data rows: {actual_row_cnt}")
    print(f"== Total count of sentences: {actual_sentence_cnt}")
    print(f"== Total count of sentence-embeddings: {actual_embedding_counter}")
    print(f"== Total count of token-embeddings: {actual_token_embedding_counter}")
    print(f"== Total count of detected zero sentence-embeddings: {detected_zero_embedding_cnt}")
    if skip_zero_vec:
        print(f"== Total count of skipped zero sentence-embeddings: {skipped_zero_embedding_cnt}")
    if remove_duplicates:
        print(f"== Total count of duplicate token_embeddings removed: {duplicate_removed_cnt}")
    print(f"=================================================")


def main():
    start_time = time.time()

    parser = argparse.ArgumentParser(
        description="""ck (Colbert KNN) uses GPU acceleration to generate ground truth KNN datasets with Colbert embeddings.""",
        epilog="""
Some example commands:\n
    ckef 1000 10000 100000 1000000 -k 100 --disable-memory-tuning
        """, formatter_class=KeepLineBreaksFormatter)
    parser.add_argument('query_count', type=int, help="number of query vectors to generate")
    parser.add_argument('base_count', type=int, help="number of base vectors to generate")
    parser.add_argument('query_token_count', type=int, help="number of query token vectors to generate")
    parser.add_argument('base_token_count', type=int, help="number of base token vectors to generate")
    parser.add_argument('-m', '--model_name', type=str, default="Colbertv2.0",
                        help='Colbert model name (default: Colbertv2.0)')
    parser.add_argument('-k', '--k', type=int, default=100, help='number of neighbors to compute per query vector')
    parser.add_argument('--data_dir', type=str, default='knn_dataset',
                        help='Directory to store the generated data (default: knn_dataset)')
    parser.add_argument('--skip-zero-vec', action=argparse.BooleanOptionalAction, default=True,
                        help='Skip generating zero vectors when failing to retrieve the embedding (default: True)')
    parser.add_argument('--use-dataset-api', action=argparse.BooleanOptionalAction, default=False,
                        help='Use \'pyarrow.dataset\' API to read the dataset (default: True). Recommended for large datasets.')
    parser.add_argument('--gen-hdf5', action=argparse.BooleanOptionalAction, default=True,
                        help='Generate hdf5 files (default: True)')
    parser.add_argument('--enable-memory-tuning', action='store_true', help='Enable memory tuning')
    parser.add_argument('--disable-memory-tuning', action='store_false',
                        help='Disable memory tuning (useful for very small datasets)')

    args = parser.parse_args()

    if not check_dataset_exists_remote():
        rprint(Markdown(f"The specified wikipedia dataset configuration/subset does not exist: {BASE_CONFIG}"))
        sys.exit(1)

    rprint('', Markdown(f"""**Colbert KNN** is generating brute force neighbors based on the wikipedia dataset with the following specification:\n
--- dataset/model specification ---\n
* source dataset version: `{BASE_DATASET}-{BASE_CONFIG}`\n
* query count: `{args.query_count}`\n
* base vector count: `{args.base_count}`\n
* query token vector count: `{args.query_token_count}`\n
* base token vector count: `{args.base_token_count}`\n
* model name: `{args.model_name}`\n
--- behavior specification ---\n
* skip zero vector: `{args.skip_zero_vec}`\n
* use dataset API: `{args.use_dataset_api}`\n
* generated hdf5 file: `{args.gen_hdf5}`\n
* enable memory tuning: `{args.enable_memory_tuning}`
"""))

    model_prefix = get_model_prefix(args.model_name)
    data_dir = f"{args.data_dir}/{model_prefix}/q{args.query_count}_qk{args.query_token_count}_b{args.base_count}_bk{args.base_token_count}_k{args.k}"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    input_dimensions = get_embedding_size(args.model_name)

    rprint('', Markdown("---"))
    rprint(Markdown("**Generating query dataset with embeddings ......** "), '')
    section_time = time.time()
    src_query_dataset = datasets.load_dataset(QUERY_DATASET, cache_dir=".cache", trust_remote_code=True)["train"]

    token_embed_columns = [f'token_embedding_{i}' for i in range(input_dimensions)]

    query_embed_token_filename = f'{data_dir}/{model_prefix}_{input_dimensions}_query{args.query_count}_src_embed_token.parquet'
    query_embed_streamer = ParquetStreamer(query_embed_token_filename, token_embed_columns)
    query_remove_duplicates = False
    query_print_info = True
    if not os.path.exists(query_embed_token_filename):
        (query_row_cnt, query_embed_cnt, query_embed_cnt_token, query_detected0cnt,
         query_skipped0cnt, query_sentence_cnt, query_duplicate_cnt) = (
            process_source_dataset(query_embed_streamer,
                                   src_query_dataset,
                                   args.model_name,
                                   input_dimensions,
                                   args.query_count,
                                   args.query_token_count,
                                   "question",
                                   skip_zero_vec=args.skip_zero_vec,
                                   remove_duplicates=query_remove_duplicates))
    else:
        query_print_info = False
        print(f"The source query embed file already exists, skip processing the query source dataset.\n")

    query_embed_streamer.close()
    rprint(Markdown(
        f"(**Duration**: `{time.time() - section_time:.2f} seconds out of {time.time() - start_time:.2f} seconds`)"))

    if query_print_info:
        print("\n")
        print_dataset_info("query",
                           args.query_count,
                           args.query_token_count,
                           query_row_cnt,
                           query_sentence_cnt,
                           query_embed_cnt,
                           query_embed_cnt_token,
                           query_detected0cnt,
                           args.skip_zero_vec,
                           query_skipped0cnt,
                           False,
                           query_duplicate_cnt)

    rprint(Markdown("---"), '')
    rprint(Markdown("**Generating base dataset with embeddings ......** "), '')
    section_time = time.time()
    src_base_dataset = datasets.load_dataset(BASE_DATASET,
                                             BASE_CONFIG,
                                             cache_dir=".cache",
                                             beam_runner='DirectRunner',
                                             trust_remote_code=True,
                                             split='train')

    base_embed_token_filename = f'{data_dir}/{model_prefix}_{input_dimensions}_base{args.base_count}_src_embed_token.parquet'
    base_embed_streamer = ParquetStreamer(base_embed_token_filename, token_embed_columns)
    base_remove_duplicates = False
    base_print_info = True
    if not os.path.exists(base_embed_token_filename):
        (base_row_cnt, base_embed_cnt, base_embed_cnt_token, base_detected0cnt,
         base_skipped0cnt, base_sentence_cnt, base_duplicate_cnt) = (
            process_source_dataset(base_embed_streamer,
                                   src_base_dataset,
                                   args.model_name,
                                   input_dimensions,
                                   args.base_count,
                                   args.base_token_count,
                                   "text",
                                   skip_zero_vec=args.skip_zero_vec,
                                   remove_duplicates=base_remove_duplicates))
    else:
        base_print_info = False
        print(f"The source base embed file already exists, skip processing the base source dataset.\n")

    base_embed_streamer.close()
    rprint(Markdown(
        f"(**Duration**: `{time.time() - section_time:.2f} seconds out of {time.time() - start_time:.2f} seconds`)"))

    if base_print_info:
        print("\n")
        print_dataset_info("base",
                           args.base_count,
                           args.base_token_count,
                           base_row_cnt,
                           base_sentence_cnt,
                           base_embed_cnt,
                           base_embed_cnt_token,
                           base_detected0cnt,
                           args.skip_zero_vec,
                           base_skipped0cnt,
                           False,
                           base_duplicate_cnt)
    print("\n")

    rprint(Markdown("---"), '')
    rprint(Markdown("**Computing knn ......** "), '')
    section_time = time.time()
    process_knn_computation(data_dir,
                            args.model_name,
                            input_dimensions,
                            query_embed_token_filename,
                            base_embed_token_filename,
                            mem_tune=args.enable_memory_tuning,
                            k=args.k)
    rprint(Markdown(
        f"(**Duration**: `{time.time() - section_time:.2f} seconds out of {time.time() - start_time:.2f} seconds`)"))

    rprint(Markdown("---"), '')
    rprint(Markdown("**Merging indices and distances ......** "), '')
    section_time = time.time()
    merge_indices_and_distances(data_dir, args.model_name, get_embedding_size(args.model_name), args.k)
    rprint(Markdown(
        f"(**Duration**: `{time.time() - section_time:.2f} seconds out of {time.time() - start_time:.2f} seconds`)"))

    rprint(Markdown("---"), '')
    rprint(Markdown("**Generating ivec's and fvec's ......** "), '')
    section_time = time.time()
    query_vector_fvec, base_vector_fvec, indices_ivec, distances_fvec = \
        generate_ivec_fvec_files(data_dir, args.model_name, input_dimensions,
                                 None, None,
                                 f"{model_prefix}_{input_dimensions}_final_indices_k{args.k}.parquet",
                                 f"{model_prefix}_{input_dimensions}_final_distances_k{args.k}.parquet",
                                 args.base_count,
                                 args.query_count,
                                 args.k)
    rprint(Markdown(
        f"(**Duration**: `{time.time() - section_time:.2f} seconds out of {time.time() - start_time:.2f} seconds`)"))
    rprint(Markdown("---"), '')
