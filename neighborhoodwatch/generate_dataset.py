import math
import os
import multiprocessing
import time

import spacy
import datasets
from tqdm import tqdm
import openai
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

from sentence_transformers import SentenceTransformer
from vertexai.preview.language_models import TextEmbeddingModel

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


def get_batch_embeddings_ada_002(text_list):
    embeddings = []
    chunk_size = 50
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


def get_batch_embeddings_from_generator(text_list, generator):
    embeddings = []
    chunk_size = generator.chunk_size
    total_items = len(text_list)
    chunks = math.ceil(total_items / chunk_size)

    for i in tqdm(range(chunks)):
        start = i * chunk_size
        end = min(start + chunk_size, total_items)
        process = text_list[start:end]
        zero_vector = [0.0] * generator.dimensions

        try:
            if "e5" in generator.model_name:
                process = ["query:"+ s for s in process]
            response = generator.generate_embedding(process)
        except Exception as e:
            print(f"failed to get embeddings for {process}")
            print(e)
            # append zero vector when openai fails
            for _ in process:
                embeddings.append(zero_vector)
            continue

        for item in response:
            #if len(item) != generator.model.get_sentence_embedding_dimension():
            #    print("got a bad embedding from SentenceTransformer, skipping it:")
            #    print(item)
            #    print(f"for input {process}")
            #else:
            #    embeddings.append(item)
            embeddings.append(item)

    return embeddings


class VertexAIEmbeddingGenerator:
    def __init__(self, model_name='textembedding-gecko'):
        self.model_name = model_name
        self.client = TextEmbeddingModel.from_pretrained(model_name)
        self.chunk_size = 250
        self.dimensions = 768

    def generate_embedding(self, text):
        # Ensure the text is a list of sentences
        if isinstance(text, str):
            text = [text]

        # Generate embeddings
        embeddings = self.client.get_embeddings(text)
        embeddings = [embedding.values for embedding in embeddings]
        return embeddings



class EmbeddingGenerator:
    def __init__(self, model_name='e5-v2-small'):
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name)
        self.chunk_size = 1000
        self.dimensions = self.model.get_sentence_embedding_dimension()

    # def simulate_zero_embedding(self, text):
    #     zero_vectors = [np.zeros(self.dimensions) for _ in range(len(text))]
    #     zero_vectors_array = np.stack(zero_vectors)
    #     return zero_vectors_array
    
    def generate_embedding(self, text):
        # Ensure the text is a list of sentences
        if isinstance(text, str):
            text = [text]

        # Generate embeddings
        embeddings = self.model.encode(text)
        return embeddings


def split_into_sentences(text):
    # if type(text) == pa.lib.StringScalar:
    #     text = text.as_py()
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]


def get_embeddings_from_map(text_map, generator):
    flattened_sentences = [item for _, value_list in text_map for item in value_list]
    embedding_array = None

    if generator is not None:
        embedding_array = get_batch_embeddings_from_generator(flattened_sentences, generator)
    else:
        embedding_array = get_batch_embeddings_ada_002(flattened_sentences)
    
    iterator = iter(embedding_array)
    return [(key, [next(iterator) for _ in value_list]) for key, value_list in text_map]


def process_dataset(streamer, dataset, row_count, embedding_column, model_name, skip_zero_vec=True):
    meta_array = []
    embedding_array = []

    sentence_batch_size = 10000
    sentence_batch_size = min(sentence_batch_size, row_count)
    embedding_counter = 0

    i = 0
    row_counter = 0
    skipped_embedding_cnt = 0
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

            embedding_tuple_list = None
            if ((model_name is not None) and (model_name != "ada-002")):
                generator = None
                if model_name == 'textembedding-gecko':
                    generator = VertexAIEmbeddingGenerator(model_name=model_name)
                else:
                    generator = EmbeddingGenerator(model_name=model_name)

                embedding_tuple_list = get_embeddings_from_map(text_map, generator)
            else:
                embedding_tuple_list = get_embeddings_from_map(text_map, None)

            #for embedding_tuple in tqdm(embedding_tuple_list):
            for embedding_tuple in embedding_tuple_list:
                index = embedding_tuple[0]
                embedding_list = embedding_tuple[1]
            
                #for idx, embedding in tqdm(enumerate(embedding_list)):
                for idx, embedding in enumerate(embedding_list):
                    if skip_zero_vec and is_zero_embedding(embedding):
                        skipped_embedding_cnt += 1
                        continue

                    meta_row_array = []
                    for column in dataset.column_names:
                        if column == "title":
                            # replace spaces with _
                            # title_column_value = dataset[column][index].as_py()
                            title_column_value = active_rows[index][column]
                            #assert dataset[column][index] == active_rows[index][column], f"index mismatch {dataset[column][index]} != {row[column]}"
                            meta_row_array.append(title_column_value.replace("_", " "))
                        elif column == embedding_column:
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
                        if (len(meta_array) > 0) and (len(embedding_array) > 0):
                            streamer.stream_to_parquet(meta_array, embedding_array)
                        return embedding_counter
        
            if (len(meta_array) > 0) and (len(embedding_array) > 0):
                streamer.stream_to_parquet(meta_array, embedding_array)
            i = 0
            meta_array = []
            embedding_array = []
            active_rows = []
            text_map = []
        
        row_counter += 1

    if skip_zero_vec:
        print(f"Skipped {skipped_embedding_cnt} zero embeddings")

    return embedding_counter


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

    def close(self):
        if self.writer:
            self.writer.close()
            print(f"Finished streaming to {self.filename}")


def generate_query_dataset(data_dir, row_count, model_name=None, skip_zero_vec=True):
    source = "query_vector"
    filename = f'{data_dir}/{model_name.replace("/","_")}_{source}_data_{row_count}.parquet'

    if model_name is None:
        filename = f'{data_dir}/ada_002_{source}_data_{row_count}.parquet'

    if os.path.exists(filename):
        print(f"file {filename} already exists")
        return filename

    full_dataset = datasets.load_dataset(QUERY_DATASET, cache_dir=".cache")["train"]
    streamer = ParquetStreamer(filename, full_dataset.column_names)
    processed_count = process_dataset(streamer, full_dataset, row_count, "question", model_name, skip_zero_vec)
    streamer.close()
    assert processed_count == row_count, f"Expected {row_count} rows, got {processed_count} rows."

    return filename


def generate_base_dataset(data_dir, query_vector_filename, row_count, model_name, skip_zero_vec=True):
    processed_count = 0
    skipped_count = 0

    source = "base_vector"
    filename = f'{data_dir}/{model_name.replace("/","_")}_{source}_data_{row_count}.parquet'
    if model_name is None:
        filename = f'{data_dir}/ada_002_{source}_data_{row_count}.parquet'

    if os.path.exists(filename):
        print(f"file {filename} already exists")
        return filename

    query_dataset = pq.read_table(get_full_filename(data_dir, query_vector_filename))
    query_titles = pc.unique(query_dataset.column("title")).to_pylist()

    # TODO: for a large dataset, it is recommended to use a remote runner like Dataflow or Spark
    full_dataset = datasets.load_dataset(BASE_DATASET, 
                                         BASE_CONFIG, 
                                         cache_dir=".cache", 
                                         beam_runner='DirectRunner', 
                                         split='train')
    streamer = ParquetStreamer(filename, full_dataset.column_names)

    # TODO consider iterable dataset
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
        processed_count = process_dataset(streamer, filtered_dataset, row_count, "text", model_name, skip_zero_vec)
        assert processed_count <= row_count, f"Expected less than or equal to {row_count} rows, got {processed_count} rows."

    if row_count > processed_count:
        def title_is_not_in(example):
            return example['title'] not in query_titles

        start_time = time.time()
        filtered_dataset = shuffled_dataset.filter(title_is_not_in, num_proc=num_cores)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f'filter dataset')
        print(f'Time taken: {elapsed_time} seconds')

        print("processing dataset")
        #filtered_dataset.map(process_dataset, batched=True, batch_size=10, num_proc=16, fn_kwargs={"streamer": streamer, "row_count": row_count - processed_count, "embedding_column": "text"})
        processed_count += process_dataset(streamer, filtered_dataset, row_count - processed_count, "text", model_name, skip_zero_vec)
        if not skip_zero_vec:
            assert processed_count == row_count, f"Expected {row_count} rows, got {processed_count} rows."

    streamer.close()

    return filename
