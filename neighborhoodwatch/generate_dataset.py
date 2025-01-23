import math
import os
import multiprocessing
import time

import spacy
import datasets
from tqdm import tqdm
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

from neighborhoodwatch.model_generator import EmbeddingGenerator, get_embedding_generator_for_model, \
    CohereEmbeddingV3Generator
from neighborhoodwatch.nw_utils import *

nlp = spacy.blank(f"{BASE_DATASET_LANG}")
nlp.add_pipe("sentencizer")

# Set huggingface datasets logging level to debug
# datasets.logging.set_verbosity_debug()
datasets.logging.set_verbosity_warning()


# datasets.logging.set_verbosity_info()


def is_zero_embedding(embedding):
    if all(value == 0 for value in embedding):
        return True
    else:
        return False


def split_into_sentences(text):
    # if type(text) == pa.lib.StringScalar:
    #     text = text.as_py()
    if isinstance(text, dict) and 'text' in text:
        text = text['text']
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 0]


def get_batch_embeddings_from_generator(text_list, generator, dataset_type=None):
    assert dataset_type in ["query", "document", None]

    embeddings = []
    chunk_size = generator.chunk_size
    total_items = len(text_list)
    chunks = math.ceil(total_items / chunk_size)

    zero_vec_cnt = 0

    for i in tqdm(range(chunks)):
        start = i * chunk_size
        end = min(start + chunk_size, total_items)
        process = text_list[start:end]
        zero_vector = [0.0] * generator.dimensions

        try:
            if "e5" in generator.model_name:
                process = ["query:" + s for s in process]

            if isinstance(generator, CohereEmbeddingV3Generator):
                if dataset_type == "query":
                    response = generator.generate_embedding(process, input_type="search_query")
                elif dataset_type == "document":
                    response = generator.generate_embedding(process, input_type="search_document")
                else:
                    response = generator.generate_embedding(process)
            else:
                response = generator.generate_embedding(process)

            for item in response:
                # if len(item) != generator.model.get_sentence_embedding_dimension():
                #    print("got a bad embedding from SentenceTransformer, skipping it:")
                #    print(item)
                #    print(f"for input {process}")
                # else:
                #    embeddings.append(item)
                embeddings.append(item)
        except Exception as e:
            print(f"   failed to get embeddings for text chunk (length: {len(process)}) with error: {e}")
            zero_vec_cnt = zero_vec_cnt + 1
            # zero embeddings will be skipped in the output parquet file
            for _ in process:
                embeddings.append(zero_vector)
            continue

    return embeddings, zero_vec_cnt


def get_embeddings_from_map(text_map, generator, dataset_type=None):
    flattened_sentences = [item for _, value_list in text_map for item in value_list]
    embedding_array = generator.generate_embedding(flattened_sentences, generator)
    iterator = iter(embedding_array)
    return [(key, [next(iterator) for _ in value_list]) for key, value_list in text_map]


def process_dataset(streamer,
                    dataset_type,
                    dataset,
                    row_count,
                    embedding_column,
                    model_name,
                    output_dimension,
                    output_dtype):
    meta_array = []
    embedding_array = []

    sentence_batch_size = 10000
    sentence_batch_size = min(sentence_batch_size, row_count)

    i = 0
    row_counter = 0
    sentence_batch_counter = 0
    text_map = []
    active_rows = []

    embedding_counter = 0
    skipped_embedding_cnt = 0

    generator: EmbeddingGenerator = get_embedding_generator_for_model(model_name=model_name,
                                                                      output_dimension=output_dimension,
                                                                      dataset_type=dataset_type,
                                                                      output_dtype=output_dtype)
    assert generator is not None

    for row in tqdm(dataset):
        last_row = row_counter == len(dataset) - 1
        active_rows.append(row)
        sentence_list = split_into_sentences(row[embedding_column])
        sentence_batch_counter += len(sentence_list)
        text_map.append([i, sentence_list])

        i += 1
        if sentence_batch_counter >= sentence_batch_size or last_row:
            sentence_batch_counter = 0

            embedding_tuple_list = get_embeddings_from_map(text_map, generator)

            # for embedding_tuple in tqdm(embedding_tuple_list):
            for embedding_tuple in embedding_tuple_list:
                index = embedding_tuple[0]
                embedding_list = embedding_tuple[1]

                # for idx, embedding in tqdm(enumerate(embedding_list)):
                for idx, embedding in enumerate(embedding_list):
                    if is_zero_embedding(embedding):
                        skipped_embedding_cnt += 1
                        continue

                    meta_row_array = []
                    for column in dataset.column_names:
                        if column == "title":
                            # replace spaces with _
                            # title_column_value = dataset[column][index].as_py()
                            title_column_value = active_rows[index][column]
                            # assert dataset[column][index] == active_rows[index][column], f"index mismatch {dataset[column][index]} != {row[column]}"
                            meta_row_array.append(title_column_value.replace("_", " "))
                        elif column == embedding_column:
                            assert text_map[index][0] == index, f"index mismatch {text_map[0][0]} != {index}"
                            value = text_map[index][1][idx]
                            meta_row_array.append(value)
                        else:
                            meta_row_array.append(active_rows[index][column])

                    meta_array.append(meta_row_array)
                    embedding_array.append(embedding)
                    embedding_counter += 1
                    if ((embedding_counter >= row_count or last_row) and
                            (len(meta_array) > 0) and (len(embedding_array) > 0)):
                        print(f"Total {embedding_counter} embeddings processed so far with {skipped_embedding_cnt} "
                              f"skipped out of total count {row_count}")
                        streamer.stream_to_parquet(meta_array, embedding_array)
                        return embedding_counter, skipped_embedding_cnt

            if (len(meta_array) > 0) and (len(embedding_array) > 0):
                streamer.stream_to_parquet(meta_array, embedding_array)
            i = 0
            meta_array = []
            embedding_array = []
            active_rows = []
            text_map = []

        row_counter += 1

    return embedding_counter, skipped_embedding_cnt


def write_to_parquet(source, columns, meta_array, embedding_array):
    filename = f'./{source}_data_{len(embedding_array)}.parquet'

    meta_columns = columns.copy()
    for i in range(len(embedding_array[0])):
        columns.append(f"embedding_{i}")

    writer = None
    print(f"writing file {filename}")

    columns_list = [pd.DataFrame(meta_array, columns=meta_columns)]
    for i, column in enumerate(embedding_array.T):
        columns_list.append(pd.DataFrame(column.astype('float32'), columns=[f'embedding_{i}']))

    df = pd.concat(columns_list, axis=1)
    table = pa.Table.from_pandas(df)

    if writer is None:
        writer = pq.ParquetWriter(filename, table.schema)
    writer.write_table(table)

    if writer:
        writer.close()
    print(f"wrote {filename}")
    return filename


class ParquetStreamer:
    def __init__(self, filename, columns):
        self.filename = filename
        self.columns = columns
        self.writer = None
        print(f"Initiated streaming to file {self.filename}")

    def stream_to_parquet(self, meta_array, embedding_array):
        meta_array = np.array(meta_array)
        embedding_array = np.array(embedding_array)
        meta_columns = self.columns.copy()
        for i in range(embedding_array.shape[1]):
            meta_columns.append(f"embedding_{i}")

        columns_list = [pd.DataFrame(meta_array, columns=self.columns)]
        for i, column in enumerate(embedding_array.T):
            columns_list.append(pd.DataFrame(column.astype('float32'), columns=[f'embedding_{i}']))

        df = pd.concat(columns_list, axis=1)
        table = pa.Table.from_pandas(df)

        if self.writer is None:
            self.writer = pq.ParquetWriter(self.filename, table.schema)

        self.writer.write_table(table)

    def stream_to_parquet_without_src_metadata(self, embedding_array):
        assert len(self.columns) == len(
            embedding_array[0]), f"column count mismatch: {len(self.columns)} != {len(embedding_array[0])}"

        embedding_array = np.array(embedding_array)
        df = pd.DataFrame(embedding_array.astype('float32'), columns=self.columns)
        table = pa.Table.from_pandas(df)

        if self.writer is None:
            self.writer = pq.ParquetWriter(self.filename, table.schema)

        self.writer.write_table(table)

    def close(self):
        if self.writer:
            self.writer.close()
            print(f"Finished streaming to {self.filename}")


def generate_query_dataset(data_dir, model_name, row_count, output_dimension=None, output_dtype=None):
    if output_dtype is not None:
        filename_base = f"{model_name.replace('/', '_')}_{output_dimension}_{output_dtype}_query_vector_data_{row_count}"
    else:
        filename_base = f"{model_name.replace('/', '_')}_{output_dimension}_query_vector_data_{row_count}"

    filename = f'{data_dir}/{filename_base}.parquet'
    if os.path.exists(filename):
        print(f"file {filename} already exists")
        return filename

    full_dataset = datasets.load_dataset(QUERY_DATASET, cache_dir=".cache", trust_remote_code=True)["train"]
    streamer = ParquetStreamer(filename, full_dataset.column_names)
    processed_count, skipped_count = process_dataset(streamer,
                                                     'query',
                                                     full_dataset,
                                                     row_count,
                                                     "question",
                                                     model_name,
                                                     output_dimension,
                                                     output_dtype)
    assert processed_count == row_count, f"Expected {row_count} rows, got {processed_count} rows."

    streamer.close()
    print(f"   totally processed {processed_count} non-zero embeddings and skipped {skipped_count} zero embeddings")

    return filename


def generate_base_dataset(data_dir,
                          model_name,
                          query_vector_filename,
                          row_count,
                          output_dimension=None,
                          output_dtype=None):
    processed_count = 0
    skipped_count = 0

    if output_dtype is not None:
        filename_base = f"{model_name.replace('/', '_')}_{output_dimension}_{output_dtype}_base_vector_data_{row_count}"
    else:
        filename_base = f"{model_name.replace('/', '_')}_{output_dimension}_base_vector_data_{row_count}"

    filename = f'{data_dir}/{filename_base}.parquet'
    if os.path.exists(filename):
        print(f"file {filename} already exists")
        return filename

    query_dataset = pq.read_table(get_full_filename(data_dir, query_vector_filename))
    query_titles = pc.unique(query_dataset.column("title")).to_pylist()

    # TODO: for a large dataset, it is recommended to use a remote runner like Dataflow or Spark
    full_dataset = datasets.load_dataset(BASE_DATASET,
                                         BASE_CONFIG,
                                         cache_dir=".cache",
                                         trust_remote_code=True,
                                         split='train')
    streamer = ParquetStreamer(filename, full_dataset.column_names)

    # TODO consider iterable dataset
    num_cores = multiprocessing.cpu_count()
    shuffled_dataset = full_dataset  # .shuffle(seed=42).flatten_indices(num_proc=num_cores)

    def title_is_in(example):
        return example['title'] in query_titles

    # TODO: benchmark if pyarrow compute is_in is significantly faster
    print("-- filtering base dataset 1 (title in)")
    filtered_dataset = shuffled_dataset.filter(title_is_in, num_proc=num_cores)

    if len(filtered_dataset) == 0:
        print(f"   no matching base title for query titles {query_titles}")
    else:
        print("-- processing filtered base dataset 1 (title in)")
        processed_count, skipped_count = process_dataset(streamer,
                                                         'document',
                                                         filtered_dataset,
                                                         row_count,
                                                         "text",
                                                         model_name,
                                                         output_dimension,
                                                         output_dtype)
        print(f"   so far processed {processed_count} non-zero embeddings and skipped {skipped_count} zero embeddings")
        assert processed_count <= row_count, f"Expected less than or equal to {row_count} rows, got {processed_count} rows."

    if row_count > processed_count:
        def title_is_not_in(example):
            return example['title'] not in query_titles

        start_time = time.time()
        filtered_dataset = shuffled_dataset.filter(title_is_not_in, num_proc=num_cores)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f'-- filter base dataset 2 (title not in). time taken: {elapsed_time} seconds')

        print("-- processing filtered base dataset 2 (title not in)")
        # filtered_dataset.map(process_dataset, batched=True, batch_size=10, num_proc=16, fn_kwargs={"streamer": streamer, "row_count": row_count - processed_count, "embedding_column": "text"})
        p2, d2 = process_dataset(streamer,
                                 'document',
                                 filtered_dataset,
                                 row_count - processed_count,
                                 "text",
                                 model_name,
                                 output_dimension,
                                 output_dtype)

        processed_count += p2
        skipped_count += d2
        assert processed_count == row_count, f"Expected {row_count} rows, got {processed_count} rows."

    print(f"   totally processed {processed_count} non-zero embeddings and skipped {skipped_count} zero embeddings")
    streamer.close()

    return filename
