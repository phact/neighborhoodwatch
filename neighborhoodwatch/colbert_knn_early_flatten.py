import argparse
import requests
import datasets
import tarfile
import csv
import cudf
import pyarrow.parquet as pq
import cupy as cp
from tqdm import tqdm
import pandas as pd
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

from neighborhoodwatch.cu_knn import stream_cudf_to_parquet, cleanup, tune_memory
from neighborhoodwatch.generate_dataset import EmbeddingGenerator, split_into_sentences, get_embeddings_from_map, \
    is_zero_embedding
from neighborhoodwatch.merge import merge_indices_and_distances
from neighborhoodwatch.neighborhoodwatch import KeepLineBreaksFormatter
from neighborhoodwatch.nw_utils import *
from rich import print as rprint
from rich.markdown import Markdown
from colbert.infra.config import ColBERTConfig
from colbert.modeling.checkpoint import Checkpoint
from colbert.indexing.collection_encoder import CollectionEncoder

from neighborhoodwatch.parquet_to_format import generate_ivec_fvec_files


class ColbertPreTrainedEmbeddingGenerator(EmbeddingGenerator):
    def __init__(self, model_name="colbertv2.0", download_pretrained_model=False):
        # - In Colbert model, each input text token (NOTE not input text itself) corresponds to a "mini" vector
        #   with a smaller dimension size (e.g. 128). The total "dimension" size of the entire input text is
        #   not fixed (depending on the tokens of the input text).
        # - The number here refers to the "mini" dimension size.
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
            # aka, it contains a series of mini-vectors of the size 128.
            tensor_embeddings, token_cnt = self.encoder.encode_passages(text)
            np_embed = Tensor.numpy(tensor_embeddings)
            embeddings.append(np_embed.flatten())

        return embeddings


def check_src_embed_files_exist(metadata_file, embed_batch_files):
    batch_count = len(embed_batch_files)
    all_file_exists = [os.path.exists(embed_batch_files[i]) for i in range(batch_count)]
    all_file_exists.append(os.path.exists(metadata_file))
    proceed_further = not all(all_file_exists)
    return proceed_further


def close_files(metadata_file, embed_files):
    if metadata_file is not None:
        metadata_file.close()

    for embed_file in embed_files:
        if embed_file is not None:
            embed_file.close()


def process_source_dataset(dataset,
                           model_name,
                           row_count,
                           column_to_embed,
                           metadata_filename,
                           embed_batch_filenames,
                           batch_size,
                           skip_zero_vec=True):
    batch_count = len(embed_batch_filenames)
    assert batch_count == math.ceil(row_count / batch_size), f"batch_count mismatch: {batch_count} != {math.ceil(row_count / batch_size)}"

    metadata_file = open(metadata_filename, 'w', 1)
    metadata_file_writer =csv.writer(metadata_file)

    embed_files = []
    embed_file_writers = []
    for i in range(batch_count):
        embed_files.append(open(embed_batch_filenames[i], 'w', 1))
        embed_file_writers.append(csv.writer(embed_files[i]))

    sentence_batch_size = 10000
    sentence_batch_size = min(sentence_batch_size, row_count)

    embedding_counter = 0
    detected_zero_embedding_cnt = 0
    skipped_zero_embedding_cnt = 0

    i = 0
    row_counter = 0
    sentence_batch_counter = 0
    text_map = []
    active_rows = []
    for row in tqdm(dataset):
        last_row = row_counter == len(dataset) - 1
        active_rows.append(row)
        sentence_list = split_into_sentences(row[column_to_embed])
        sentence_batch_counter += len(sentence_list)
        text_map.append([i, sentence_list])

        i += 1
        if sentence_batch_counter >= sentence_batch_size or last_row:
            sentence_batch_counter = 0

            if model_name == "colbertv2.0":
                generator = ColbertPreTrainedEmbeddingGenerator()
            else:
                assert False, f"Unsupported model name: {model_name}"

            embedding_tuple_list, detected_zero_embedding_cnt = get_embeddings_from_map(text_map, generator)

            # for embedding_tuple in tqdm(embedding_tuple_list):
            for embedding_tuple in embedding_tuple_list:
                index = embedding_tuple[0]
                embedding_list = embedding_tuple[1]

                # for idx, embedding in tqdm(enumerate(embedding_list)):
                for idx, embedding in enumerate(embedding_list):
                    if skip_zero_vec and is_zero_embedding(embedding):
                        skipped_zero_embedding_cnt += 1
                    else:
                        meta_row_array = []
                        for column in dataset.column_names:
                            if column == "title":
                                # replace spaces with _
                                # title_column_value = dataset[column][index].as_py()
                                title_column_value = active_rows[index][column]
                                # assert dataset[column][index] == active_rows[index][column], f"index mismatch {dataset[column][index]} != {row[column]}"
                                meta_row_array.append(title_column_value.replace("_", " "))
                            elif column == column_to_embed:
                                assert text_map[index][0] == index, f"index mismatch {text_map[0][0]} != {index}"
                                value = text_map[index][1][idx]
                                meta_row_array.append(value)
                            else:
                                meta_row_array.append(active_rows[index][column])

                        metadata_file_writer.writerow(meta_row_array)

                        current_batch = embedding_counter // batch_size
                        embed_file_writers[current_batch].writerow(embedding)
                        embedding_counter += 1

                        if (embedding_counter + skipped_zero_embedding_cnt) >= row_count:
                            close_files(metadata_file, embed_files)
                            return embedding_counter, detected_zero_embedding_cnt, skipped_zero_embedding_cnt

            i = 0
            active_rows = []
            text_map = []

        row_counter += 1

    close_files(metadata_file, embed_files)
    return embedding_counter, detected_zero_embedding_cnt, skipped_zero_embedding_cnt


def flatten_wide_table(input_data_arr,
                       input_dimensions):
    flattened_data_arr = []
    mini_embed_columns = [f'mini_embedding_{i}' for i in range(input_dimensions)]

    for i in tqdm(range(len(input_data_arr))):
        embed_row = input_data_arr[i]
        mini_vec_num = len(embed_row) // input_dimensions
        for j in tqdm(range(mini_vec_num)):
            mini_embed = embed_row[j * input_dimensions: (j + 1) * input_dimensions]
            flattened_data_arr.append(mini_embed)

    flattened_data_arr = np.array(flattened_data_arr, dtype=np.float32)
    df = cudf.DataFrame(flattened_data_arr, columns=mini_embed_columns)

    return df


def read_embed_arr_from_file(embed_file):
    embed_arr = []
    csv_file_reader = csv.reader(embed_file)
    for row in tqdm(csv_file_reader):
        row_arr = np.array(row, dtype=np.float32)
        embed_arr.append(row_arr)

    return embed_arr


def process_knn_computation(data_dir,
                            model_name,
                            input_dimensions,
                            query_embed_files,
                            query_embed_cnt,
                            base_embed_files,
                            base_embed_cnt,
                            batch_size,
                            k=100,
                            split=True):
    assert (base_embed_cnt % batch_size == 0) or k <= (
            base_embed_cnt % batch_size), f"Cannot generate k of {k} with only {base_embed_cnt} rows and batch_size of {batch_size}."

    rmm.mr.set_current_device_resource(rmm.mr.PoolMemoryResource(rmm.mr.ManagedMemoryResource()))
    model_prefix = get_model_prefix(model_name)

    query_src_batch_count = len(query_embed_files)
    base_src_batch_count = len(base_embed_files)

    for base_start in tqdm(range(base_src_batch_count)):
        batch_offset = base_start * batch_size

        base_embed_file = open(base_embed_files[base_start], 'r')
        base_src_embed_arr = read_embed_arr_from_file(base_embed_file)
        base_df_flat = flatten_wide_table(base_src_embed_arr, input_dimensions)
        base_dataset = cp.from_dlpack(base_df_flat.to_dlpack()).copy(order='C')
        base_embed_file.close()

        cleanup(base_df_flat, base_src_embed_arr)

        distances = cudf.DataFrame()
        indices = cudf.DataFrame()
        for query_start in tqdm(range(query_src_batch_count)):
            query_embed_file = open(query_embed_files[query_start], 'r')
            query_src_embed_arr = read_embed_arr_from_file(query_embed_file)
            query_df_flat = flatten_wide_table(query_src_embed_arr, input_dimensions)
            query_dataset = cp.from_dlpack(query_df_flat.to_dlpack()).copy(order='C')
            query_embed_file.close()

            cleanup(query_df_flat, query_src_embed_arr)

            assert (k <= len(base_dataset))
            cupydistances1, cupyindices1 = knn(base_dataset.astype(np.float32),
                                               query_dataset.astype(np.float32),
                                               k)

            distances1 = cudf.from_pandas(pd.DataFrame(cp.asarray(cupydistances1).get()))
            # add batch_offset to indices
            indices1 = cudf.from_pandas(pd.DataFrame((cp.asarray(cupyindices1) + batch_offset).get()))

            distances = cudf.concat([distances, distances1], ignore_index=True)
            indices = cudf.concat([indices, indices1], ignore_index=True)

        distances.columns = distances.columns.astype(str)
        indices.columns = indices.columns.astype(str)

        chunk_size = 10000
        stream_cudf_to_parquet(distances, chunk_size,
                           f'{data_dir}/{model_prefix}_{input_dimensions}_distances{base_start}.parquet')
        stream_cudf_to_parquet(indices, chunk_size, f'{data_dir}/{model_prefix}_{input_dimensions}_indices{base_start}.parquet')

        cleanup(distances, indices, base_dataset)


def main():
    start_time = time.time()

    parser = argparse.ArgumentParser(
        description="""ck (Colbert KNN) uses GPU acceleration to generate ground truth KNN datasets with Colbert embeddings.""",
        epilog="""
Some example commands:\n
    ck 1000 10000 -k 100 --disable-memory-tuning
        """, formatter_class=KeepLineBreaksFormatter)
    parser.add_argument('query_count', type=int, help="number of query vectors to generate")
    parser.add_argument('base_count', type=int, help="number of base vectors to generate")
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
* model name: `{args.model_name}`\n
--- behavior specification ---\n
* skip zero vector: `{args.skip_zero_vec}`\n
* use dataset API: `{args.use_dataset_api}`\n
* generated hdf5 file: `{args.gen_hdf5}`\n
* enable memory tuning: `{args.enable_memory_tuning}`
"""))

    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

    input_dimensions = get_embedding_size(args.model_name)
    src_data_proc_batch_size = int(args.query_count)

    rprint('', Markdown("---"))
    rprint(Markdown("**Generating query dataset with embeddings ......** "), '')
    section_time = time.time()
    src_query_dataset = datasets.load_dataset(QUERY_DATASET, cache_dir=".cache", trust_remote_code=True)["train"]

    query_metadata_file = f'{args.data_dir}/{args.model_name.replace("/", "_")}_{input_dimensions}_query{args.query_count}_src_metadata.csv'
    query_embed_filename_base = f'{args.data_dir}/{args.model_name.replace("/", "_")}_{input_dimensions}_query{args.query_count}_src_embed'
    query_embed_cnt = int(args.query_count)
    query_embed_files = [f"{query_embed_filename_base}_batch{i}.csv" for i in range(math.ceil(query_embed_cnt / src_data_proc_batch_size))]
    if check_src_embed_files_exist(query_metadata_file, query_embed_files):
        query_embed_cnt, query_detected0cnt, query_skipped0cnt = (
            process_source_dataset(src_query_dataset,
                                   args.model_name,
                                   args.query_count,
                                   "question",
                                   query_metadata_file,
                                   query_embed_files,
                                   src_data_proc_batch_size,
                                   args.skip_zero_vec))
    else:
        print(f"All embed batch files already exist, skip processing the query source dataset.\n")
    rprint(Markdown(
        f"(**Duration**: `{time.time() - section_time:.2f} seconds out of {time.time() - start_time:.2f} seconds`)"))

    rprint(Markdown("---"), '')
    rprint(Markdown("**Generating base dataset with embeddings ......** "), '')
    section_time = time.time()
    src_base_dataset = datasets.load_dataset(BASE_DATASET,
                                             BASE_CONFIG,
                                             cache_dir=".cache",
                                             beam_runner='DirectRunner',
                                             trust_remote_code=True,
                                             split='train')

    base_metadata_file = f'{args.data_dir}/{args.model_name.replace("/", "_")}_{input_dimensions}_base{args.base_count}_src_metadata.csv'
    base_embed_filename_base = f'{args.data_dir}/{args.model_name.replace("/", "_")}_{input_dimensions}_base{args.base_count}_src_embed'
    base_embed_cnt = int(args.base_count)
    base_embed_files = [f"{base_embed_filename_base}_batch{i}.csv" for i in range(math.ceil(base_embed_cnt / src_data_proc_batch_size))]
    if check_src_embed_files_exist(base_metadata_file, base_embed_files):
        base_embed_cnt, base_detected0cnt, base_skipped0cnt = (
            process_source_dataset(src_base_dataset,
                                   args.model_name,
                                   args.base_count,
                                   "text",
                                   base_metadata_file,
                                   base_embed_files,
                                   src_data_proc_batch_size,
                                   args.skip_zero_vec))
    else:
        print(f"All embed batch files already exist, skip processing the base source dataset.\n")
    rprint(Markdown(
        f"(**Duration**: `{time.time() - section_time:.2f} seconds out of {time.time() - start_time:.2f} seconds`)"))

    rprint(Markdown("---"), '')
    rprint(Markdown("**Computing knn ......** "), '')
    section_time = time.time()
    process_knn_computation(args.data_dir,
                            args.model_name,
                            input_dimensions,
                            query_embed_files,
                            query_embed_cnt,
                            base_embed_files,
                            base_embed_cnt,
                            batch_size=src_data_proc_batch_size,
                            k=args.k,
                            split=True)
    rprint(Markdown(
        f"(**Duration**: `{time.time() - section_time:.2f} seconds out of {time.time() - start_time:.2f} seconds`)"))

    rprint(Markdown("---"), '')
    rprint(Markdown("**Merging indices and distances ......** "), '')
    section_time = time.time()
    merge_indices_and_distances(args.data_dir, args.model_name, get_embedding_size(args.model_name), args.k)
    rprint(Markdown(
        f"(**Duration**: `{time.time() - section_time:.2f} seconds out of {time.time() - start_time:.2f} seconds`)"))

    rprint(Markdown("---"), '')
    rprint(Markdown("**Generating ivec's and fvec's ......** "), '')
    model_prefix = get_model_prefix(args.model_name)
    section_time = time.time()
    query_vector_fvec, base_vector_fvec, indices_ivec, distances_fvec = \
        generate_ivec_fvec_files(args.data_dir, args.model_name, input_dimensions,
                                 None, None,
                                 f"{model_prefix}_{input_dimensions}_final_indices_k{args.k}.parquet",
                                 f"{model_prefix}_{input_dimensions}_final_distances_k{args.k}.parquet",
                                 args.base_count,
                                 args.query_count,
                                 args.k)
    rprint(Markdown(
        f"(**Duration**: `{time.time() - section_time:.2f} seconds out of {time.time() - start_time:.2f} seconds`)"))
    rprint(Markdown("---"), '')
