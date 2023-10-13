import math
import os
import gc
import multiprocessing
import time

import spacy
from datasets import load_dataset, Dataset
from tqdm import tqdm
import openai
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

BASE_DATASET = "wikipedia"
BASE_CONFIG = "20220301.en"

QUERY_DATASET = "squad"

nlp = spacy.blank("en")
nlp.add_pipe("sentencizer")


def get_batch_embeddings_ada_002(text_list):
    embeddings = []
    chunk_size = 100
    total_items = len(text_list)
    chunks = math.ceil(total_items / chunk_size)

    for i in tqdm(range(chunks)):
        start = i * chunk_size
        end = min(start + chunk_size, total_items)
        process = text_list[start:end]
        zero_vector = [0.0] * 1536

        try:
            response = openai.Embedding.create(model="text-embedding-ada-002", input=process)
        except Exception as e:
            print(f"failed to get embeddings for {process}")
            print(e)
            # append zero vector when openai fails
            for _ in process:
                embeddings.append(zero_vector)
            continue

        for item in response["data"]:
            if len(item["embedding"]) != 1536:
                print("got a bad embedding from openai, skipping it:")
                print(item["embedding"])
                print(f"for input {process}")
            else:
                embeddings.append(item["embedding"])

    return embeddings


def split_into_sentences(text):
    # if type(text) == pa.lib.StringScalar:
    #     text = text.as_py()
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]


def get_embeddings_from_map(text_map):
    flattened_sentences = [item for _, value_list in text_map for item in value_list]
    embedding_array = get_batch_embeddings_ada_002(flattened_sentences)
    iterator = iter(embedding_array)
    return [(key, [next(iterator) for _ in value_list]) for key, value_list in text_map]


def process_dataset(dataset, row_count, embedding_column, meta_array=[], embedding_array=[]):
    sentence_batch_size = 1000
    sentence_batch_size = min(sentence_batch_size, row_count)
    embedding_counter = 0

    documents = dataset[embedding_column]

    i = 0
    row_counter = 0
    sentence_batch_counter = 0
    text_map = []
    active_rows = []
    for row in tqdm(dataset):
        last_row = row_counter == len(dataset) - 1
        active_rows.append(row)
        sentence_list = split_into_sentences(row[embedding_column])
        sentence_batch_counter += len(sentence_list)
        text_map.append([i, sentence_list])

        i += 1
        if sentence_batch_counter >= sentence_batch_size or last_row:
            sentence_batch_counter = 0

            embedding_tuple_list = get_embeddings_from_map(text_map)

            #for embedding_tuple in tqdm(embedding_tuple_list):
            for embedding_tuple in embedding_tuple_list:
                index = embedding_tuple[0]
                embedding_list = embedding_tuple[1]
                #for idx, embedding in tqdm(enumerate(embedding_list)):
                for idx, embedding in enumerate(embedding_list):
                    meta_row_array = []
                    for column in dataset.column_names:
                        if column == "title":
                            # replace spaces with _
                            # title_column_value = dataset[column][index].as_py()
                            title_column_value = active_rows[index][column]
                            #assert dataset[column][index] == active_rows[index][column], f"index mismatch {dataset[column][index]} != {row[column]}"
                            meta_row_array.append(title_column_value.replace("_", " "))
                        elif column == embedding_column:
                            if (text_map[index][0] != index):
                                print("hi")
                            assert text_map[index][0] == index, f"index mismatch {text_map[0][0]} != {index}"

                            value = text_map[index][1][idx]
                            meta_row_array.append(value)
                        else:
                            meta_row_array.append(active_rows[index][column])

                    meta_array.append(meta_row_array)

                    embedding_array.append(
                        embedding
                    )
                    embedding_counter += 1
                    if embedding_counter >= row_count:
                        print(f"Total embeddings so far {embedding_counter} out of {row_count}")
                        return (meta_array, embedding_array)
            i = 0
            active_rows = []
            text_map = []
        row_counter += 1
    return (meta_array, embedding_array)


def write_to_parquet(source, columns, meta_array, embedding_array):
    print("embedding array length ")
    print(len(embedding_array))
    print("meta array length ")
    print(len(meta_array))

    lengths = [len(item) for item in meta_array]
    print("meta array item lengths ")
    print(set(lengths))

    lengths = [len(item) for item in embedding_array]
    print("embedding array item lengths ")
    print(set(lengths))

    embeddings = np.array(embedding_array)
    meta = np.array(meta_array)

    filename = f'./{source}_data_{len(embeddings)}.parquet'
    chunk_size = 1000

    num_chunks = len(embeddings) // chunk_size + (len(embeddings) % chunk_size != 0)

    meta_columns = columns.copy()
    #TODO fix here maybe?
    for i in range(embeddings.shape[1]):
        columns.append(f"embedding_{i}")

    writer = None
    for idx in range(num_chunks):
        start_idx = idx * chunk_size
        end_idx = start_idx + chunk_size

        meta_chunk = meta[start_idx:end_idx]
        embeddings_chunk = embeddings[start_idx:end_idx]

        # nparray_chunk = np.hstack((meta_chunk, embeddings_chunk))
        # df_chunk = pd.DataFrame(nparray_chunk, columns=columns)

        columns_list = [pd.DataFrame(meta_chunk, columns=meta_columns)]
        for i, column in enumerate(embeddings_chunk.T):
            columns_list.append(pd.DataFrame(column.astype('float32'), columns=[f'embedding_{i}']))

        df_chunk = pd.concat(columns_list, axis=1)
        table_chunk = pa.Table.from_pandas(df_chunk)

        if writer is None:
            writer = pq.ParquetWriter(filename, table_chunk.schema)
        writer.write_table(table_chunk)

    if writer:
        writer.close()
    print(f"wrote {filename}")
    return filename


def generate_query_dataset(row_count):
    source = "query_vector"
    filename = f'./{source}_data_{row_count}.parquet'
    if os.path.exists(filename):
        print(f"file {filename} already exists")
        return filename

    full_dataset = load_dataset(QUERY_DATASET, cache_dir="./data")["train"]
    both_arrays = process_dataset(full_dataset, row_count, "question")
    meta_array = both_arrays[0]
    embedding_array = both_arrays[1]

    meta_array = meta_array[:row_count]
    embedding_array = embedding_array[:row_count]

    print(f"Loaded dataset: {QUERY_DATASET}")
    return write_to_parquet(source, full_dataset.column_names, meta_array, embedding_array)


def generate_base_dataset(query_vector_filename, row_count):
    meta_array = []
    embedding_array = []

    source = "base_vector"
    filename = f'./{source}_data_{row_count}.parquet'
    if os.path.exists(filename):
        print(f"file {filename} already exists")
        return filename

    query_dataset = pq.read_table(query_vector_filename)
    query_titles = pc.unique(query_dataset.column("title")).to_pylist()

    full_dataset = load_dataset(BASE_DATASET, BASE_CONFIG, cache_dir="./data")["train"]
    # TODO consider itterable datset
    num_cores = multiprocessing.cpu_count()
    shuffled_dataset = full_dataset  # .shuffle(seed=42).flatten_indices(num_proc=num_cores)

    def title_is_in(example):
        return example['title'] in query_titles

    # TODO: benchmark if pyarrow compute is_in is significantly faster
    print("filtering dataset")
    filtered_dataset = shuffled_dataset.filter(title_is_in, num_proc=num_cores)

    if len(filtered_dataset) == 0:
        print(f"no matching base title for query titles {query_titles}")
    else:
        print("processing dataset")
        both_arrays = process_dataset(filtered_dataset, row_count, "text")
        meta_array = both_arrays[0]
        embedding_array = both_arrays[1]

        meta_array = meta_array[:row_count]
        embedding_array = embedding_array[:row_count]

    if row_count > len(meta_array):
        def title_is_not_in(example):
            return example['title'] not in query_titles

        start_time = time.time()
        filtered_dataset = shuffled_dataset.filter(title_is_not_in, num_proc=num_cores)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f'filter dataset')
        print(f'Time taken: {elapsed_time} seconds')

        # title_in_mask = pa.compute.is_in(arrow_table.column("title"), query_titles)
        # title_not_in_mask = pc.invert(title_in_mask)
        # filtered_table = arrow_table.filter(title_not_in_mask)
        # filtered_table_arrow = pa.Table.from_batches(filtered_table.to_batches())

        # n_rows = filtered_table.num_rows
        # shuffled_indices = np.arange(n_rows)
        # np.random.shuffle(shuffled_indices)

        # shuffled_table = filtered_table_arrow.take(pa.array(shuffled_indices))

        # base_dataset = Dataset.from_pandas(filtered_table_arrow.to_pandas())

        # consider filterd_dataset.map(
        # process_dataset,
        # kwargs={"row_count": row_count - len(meta_array), "embedding_column": "text"},
        # batched=True,
        # batch_size=1000,
        # num_proc=num_cores
        # )
        print("processing dataset")
        both_arrays = process_dataset(filtered_dataset, row_count - len(meta_array), "text", meta_array,
                                      embedding_array)
        meta_array = both_arrays[0]
        embedding_array = both_arrays[1]

        meta_array = meta_array[:row_count]
        embedding_array = embedding_array[:row_count]

    print("writing dataset")
    return write_to_parquet(source, full_dataset.column_names, meta_array, embedding_array)
