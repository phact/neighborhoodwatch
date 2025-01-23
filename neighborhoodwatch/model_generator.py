import math
import os
import numpy as np
import requests

from abc import ABC, abstractmethod
from enum import Enum

from tqdm import tqdm

from openai import OpenAI
from sentence_transformers import SentenceTransformer
from vertexai.preview.language_models import TextEmbeddingModel

import cohere
import voyageai


class EmbeddingModelName(Enum):
    OPENAI_ADA_002 = 'text-embedding-ada-002'
    OPENAI_V3_SMALL = 'text-embedding-3-small'
    OPENAI_V3_LARGE = 'text-embedding-3-large'
    GOOGLE_TEXT_GECKO_003 = 'textembedding-gecko@003'
    GOOGLE_TEXT_EMBEDDING_004 = 'text-embedding-004'
    GOOGLE_TEXT_EMBEDDING_005 = 'text-embedding-005'
    INTFLOAT_E5_LARGE_V2 = 'intfloat/e5-large-v2'
    INTFLOAT_E5_BASE_V2 = 'intfloat/e5-base-v2'
    INTFLOAT_E5_SMALL_V2 = 'intfloat/e5-small-v2'
    NVIDIA_NEMO = 'nvidia-nemo'
    COHERE_ENGLISH_V3 = 'cohere/embed-english-v3.0'
    COHERE_ENGLISH_LIGHT_V3 = 'cohere/embed-english-light-3.0'
    VOYAGE_3_LARGE = 'voyage-3-large'
    VOYAGE_3_LITE = 'voyage-3-lite'
    ## Colbert model is a per-token embedding model
    COLBERT_V2 = 'colbert-v2.0'


def get_valid_model_name_list():
    return [model.value for model in EmbeddingModelName]


def get_valid_model_names_string() -> str:
    names = get_valid_model_name_list()
    return ', '.join(names)


def is_valid_model_name(model_name: str) -> bool:
    if model_name is not None and model_name in get_valid_model_name_list():
        return True
    else:
        return False


def get_default_model_dimension_size(model_name: str):
    assert is_valid_model_name(model_name)

    # OpenAI embedding models
    if model_name == EmbeddingModelName.OPENAI_ADA_002.value or model_name == EmbeddingModelName.OPENAI_V3_SMALL.value:
        return 1536
    elif model_name == EmbeddingModelName.OPENAI_V3_LARGE.value:
        return 3072
    # VertexAI embedding models
    elif (model_name == EmbeddingModelName.GOOGLE_TEXT_GECKO_003.value or
          model_name == EmbeddingModelName.GOOGLE_TEXT_EMBEDDING_004.value or
          model_name == EmbeddingModelName.GOOGLE_TEXT_EMBEDDING_005.value):
        return 768
    # HuggingFace embedding models
    elif model_name == EmbeddingModelName.INTFLOAT_E5_LARGE_V2.value:
        return 1024
    elif model_name == EmbeddingModelName.INTFLOAT_E5_BASE_V2.value:
        return 768
    elif model_name == EmbeddingModelName.INTFLOAT_E5_SMALL_V2.value:
        return 384
    # Nvidia models
    elif model_name == EmbeddingModelName.NVIDIA_NEMO.value:
        return 1024
    # Cohere models
    elif model_name == EmbeddingModelName.COHERE_ENGLISH_V3.value:
        return 1024
    elif model_name == EmbeddingModelName.COHERE_ENGLISH_LIGHT_V3.value:
        return 384
    # Voyage models
    elif model_name == EmbeddingModelName.VOYAGE_3_LARGE.value:
        return 1024
    elif model_name == EmbeddingModelName.VOYAGE_3_LITE.value:
        return 512
    # Colbert models
    elif model_name == EmbeddingModelName.COLBERT_V2.value:
        return 128


def get_effective_embedding_size(model_name: str, output_dimension_size: int = None):
    default_dimension_size = get_default_model_dimension_size(model_name)

    if output_dimension_size is not None:
        if model_name == EmbeddingModelName.OPENAI_V3_SMALL.value or model_name == EmbeddingModelName.OPENAI_V3_LARGE.value:
            assert (output_dimension_size <= default_dimension_size)
            return output_dimension_size
        elif model_name == EmbeddingModelName.VOYAGE_3_LARGE.value:
            assert (output_dimension_size in [256, 512, 1024, 2048])
            return output_dimension_size
        else:
            # Ignore output dimension size
            return default_dimension_size
    else:
        return default_dimension_size


class EmbeddingGenerator(ABC):
    def __init__(self,
                 model_name: str,
                 chunk_size: int,
                 # output dimension; None if the model doesn't support this feature
                 output_dimension: int = None):
        self.model_name = model_name
        assert is_valid_model_name(self.model_name), \
            f"The given model name is invalid; must be one of: {get_valid_model_names_string()}"

        # Most models have some restrictions on the size of the batch of the texts to be processed
        #   in one call. E.g. Cohere's limit is 96. Otherwise, it will fail the API call.
        assert chunk_size is not None and 0 < chunk_size <= 64

        self.model_dimension = get_default_model_dimension_size(self.model_name)
        self.output_dimension = get_effective_embedding_size(self.model_name, output_dimension)
        self.chunk_size = chunk_size

        assert self.output_dimension is None or self.output_dimension > 0

    def generate_embedding(self, text_list, *args, **kwargs):
        if isinstance(text_list, str):
            text_list = [text_list]

        embeddings = []
        total_items = len(text_list)
        chunks = math.ceil(total_items / self.chunk_size)
        zero_vector = [0.0] * self.output_dimension

        total_embed_cnt = 0
        total_zero_embed_cnt = 0

        for i in tqdm(range(chunks)):
            start = i * self.chunk_size
            end = min(start + self.chunk_size, total_items)

            process = text_list[start:end]
            if "e5" in self.model_name:
                process = ["query:" + s for s in process]

            try:
                model_output = self._call_model_api(process, *args, **kwargs)
                embeddings.extend(model_output)
            except Exception as e:
                print(f"   >>> [WARN] failed to retrieve the embeddings: {e}")
                for _ in process:
                    embeddings.append(zero_vector)
                total_zero_embed_cnt += len(process)
                continue
            finally:
                total_embed_cnt += len(process)

        return embeddings

    @abstractmethod
    def _call_model_api(self, text: list, *args, **kwargs):
        pass


class OpenAIEmbeddingGenerator(EmbeddingGenerator):
    """Description
    OpenAI text embedding generator. Supports all text embedding models from OpenAI
    * text-embedding-ada-002 (default) - default dimension size: 1536 (doesn't support reduced output dimension size)
    * text-embedding-3-small - default dimension size: 1536 (support reduced output dimension size)
    * text-embedding-3-large - default dimension size: 3072 (support reduced output dimension size)
    """

    def __init__(self,
                 model_name=EmbeddingModelName.OPENAI_V3_SMALL,
                 output_dimension_size=None):
        assert (model_name == EmbeddingModelName.OPENAI_ADA_002.value or
                model_name == EmbeddingModelName.OPENAI_V3_SMALL.value or
                model_name == EmbeddingModelName.OPENAI_V3_LARGE.value)

        super().__init__(model_name=model_name,
                         chunk_size=64,
                         output_dimension=output_dimension_size)

        assert (0 < output_dimension_size <= self.model_dimension)

        if os.getenv("OPENAI_API_KEY") is None:
            raise Exception("'OPENAI_API_KEY' environment variable is not set!")

        self.client = OpenAI()

    def _call_model_api(self, text_list: list, *args, **kwargs):
        if self.model_name == EmbeddingModelName.OPENAI_ADA_002.value:
            results = self.client.embeddings.create(input=text_list,
                                                    model=self.model_name)
        else:
            results = self.client.embeddings.create(input=text_list,
                                                    model=self.model_name,
                                                    dimensions=get_effective_embedding_size(self.model_name,
                                                                                            self.output_dimension))
        embeddings = [item.embedding for item in results.data]
        return embeddings


class VertexAIEmbeddingGenerator(EmbeddingGenerator):
    def __init__(self,
                 model_name=EmbeddingModelName.GOOGLE_TEXT_EMBEDDING_005,
                 normalize_embed=False):
        assert (model_name == EmbeddingModelName.GOOGLE_TEXT_GECKO_003.value or
                model_name == EmbeddingModelName.GOOGLE_TEXT_EMBEDDING_004.value or
                model_name == EmbeddingModelName.GOOGLE_TEXT_EMBEDDING_005.value)

        super().__init__(model_name=model_name, chunk_size=64)

        self.client = TextEmbeddingModel.from_pretrained(self.model_name)

    def _call_model_api(self, text_list: list, *args, **kwargs):
        results = self.client.get_embeddings(text_list)
        output = [embedding.values for embedding in results]
        return output


class IntfloatE5EmbeddingGenerator(EmbeddingGenerator):
    def __init__(self,
                 model_name=EmbeddingModelName.INTFLOAT_E5_BASE_V2,
                 normalize_embed=False):
        assert (model_name == EmbeddingModelName.INTFLOAT_E5_SMALL_V2.value or
                model_name == EmbeddingModelName.INTFLOAT_E5_BASE_V2.value or
                model_name == EmbeddingModelName.INTFLOAT_E5_LARGE_V2.value)

        super().__init__(model_name=model_name, chunk_size=64)

        self.model = SentenceTransformer(model_name, trust_remote_code=True)

    def _call_model_api(self, text, *args, **kwargs):
        results = self.model.encode(text, normalize_embeddings=True)
        return results


class NvdiaNemoEmbeddingGenerator(EmbeddingGenerator):
    def __init__(self,
                 model_name=EmbeddingModelName.NVIDIA_NEMO,
                 embedding_srv_url='http://localhost:8080/v1/embeddings',
                 normalize_embed=False):
        assert (model_name == EmbeddingModelName.NVIDIA_NEMO.value)

        super().__init__(model_name=model_name, chunk_size=64)
        self.embedding_srv_url = embedding_srv_url
        self.stand_headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}

    def _call_model_api(self, text_list: list, *args, **kwargs):
        data = {"input": text_list,
                "model": "NV-Embed-QA",
                "input_type": "passage"}

        response = requests.post(self.embedding_srv_url, json=data, headers=self.stand_headers)
        response_json_data = response.json() if response and response.status_code == 200 else None

        if response_json_data:
            output = [item['embedding'] for item in response_json_data['data']]
            return output
        else:
            raise Exception(f"Failed to get embeddings from {self.model_name} with response: {response}")


class CohereEmbeddingV3Generator(EmbeddingGenerator):
    def __init__(self,
                 model_name=EmbeddingModelName.COHERE_ENGLISH_V3.value):
        assert (model_name == EmbeddingModelName.COHERE_ENGLISH_V3.value or
                model_name == EmbeddingModelName.COHERE_ENGLISH_LIGHT_V3.value)

        ## Cohere Doc (https://docs.cohere.com/v2/reference/embed):
        #  - Maximum number of texts per call is 96.
        #  - We recommend reducing the length of each text to be under 512 tokens for optimal quality.
        super().__init__(model_name=model_name, chunk_size=64)

        if os.getenv("COHERE_API_KEY") is None:
            raise Exception("'COHERE_API_KEY' environment variable is not set")

        self.co_client = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))

        # remove the leading "cohere/" from the model name
        self.model_name = model_name.split("/")[1]

    def _call_model_api(self, text, *args, **kwargs):
        valid_input_types = ["search_query", "search_document", "classification", "clustering"]

        input_type = kwargs.get("input_type")
        assert (input_type is not None and
                input_type in valid_input_types), \
            "input_type is required for Cohere embeddings and must be one of: " + ", ".join(valid_input_types)

        results = self.co_client.embed(texts=text, model=self.model_name, input_type=input_type)
        return np.array(results.embeddings)


class VoyageAIEmbeddingGenerator(EmbeddingGenerator):
    def __init__(self,
                 model_name='voyage-3-large',
                 input_type='document',
                 output_dtype='float',
                 output_dimension_size=None):

        assert (model_name == EmbeddingModelName.VOYAGE_3_LARGE.value or
                model_name == EmbeddingModelName.VOYAGE_3_LITE.value)

        assert input_type in ['query', 'document']

        if model_name == EmbeddingModelName.VOYAGE_3_LARGE.value:
            assert output_dimension_size is None or output_dimension_size in [256, 512, 1024, 2048]
            assert output_dtype in ['float', 'int8', 'uint8', 'binary', 'ubinary']
        elif model_name == EmbeddingModelName.VOYAGE_3_LITE.value:
            assert output_dtype in ['float']

        ## Cohere Doc (https://docs.voyageai.com/docs/embeddings#python-api):
        #  - Maximum number of texts per call is 128.
        super().__init__(model_name=model_name,
                         chunk_size=64,
                         output_dimension=output_dimension_size)

        self.input_type = input_type
        self.output_dtype = output_dtype

        if os.getenv("VOYAGE_API_KEY") is None:
            raise Exception("'VOYAGE_API_KEY' environment variable is not set")

        self.vo_client = voyageai.Client()

    def _call_model_api(self, text, *args, **kwargs):
        results = self.vo_client.embed(text,
                                       model=self.model_name,
                                       input_type=self.input_type,
                                       output_dimension=get_effective_embedding_size(self.model_name,
                                                                                     self.output_dimension),
                                       output_dtype=self.output_dtype)

        output = [item for item in results.embeddings]

        return output


def get_embedding_generator_for_model(model_name,
                                      output_dimension=None,
                                      dataset_type=None,
                                      output_dtype=None):
    assert is_valid_model_name(model_name)

    if model_name == EmbeddingModelName.OPENAI_ADA_002.value:
        return OpenAIEmbeddingGenerator(model_name=model_name)
    elif (model_name == EmbeddingModelName.OPENAI_V3_SMALL.value or
          model_name == EmbeddingModelName.OPENAI_V3_LARGE.value):
        return OpenAIEmbeddingGenerator(model_name=model_name,
                                        output_dimension_size=output_dimension)
    elif (model_name == EmbeddingModelName.GOOGLE_TEXT_GECKO_003.value or
          model_name == EmbeddingModelName.GOOGLE_TEXT_EMBEDDING_004.value or
          model_name == EmbeddingModelName.GOOGLE_TEXT_EMBEDDING_005.value):
        return VertexAIEmbeddingGenerator(model_name=model_name)
    elif (model_name == EmbeddingModelName.INTFLOAT_E5_SMALL_V2.value or
          model_name == EmbeddingModelName.INTFLOAT_E5_BASE_V2.value or
          model_name == EmbeddingModelName.INTFLOAT_E5_LARGE_V2.value):
        return IntfloatE5EmbeddingGenerator(model_name=model_name)
    elif model_name == EmbeddingModelName.NVIDIA_NEMO.value:
        return NvdiaNemoEmbeddingGenerator(model_name=model_name)
    elif (model_name == EmbeddingModelName.COHERE_ENGLISH_V3.value or
          model_name == EmbeddingModelName.COHERE_ENGLISH_LIGHT_V3.value):
        return CohereEmbeddingV3Generator(model_name=model_name)
    elif model_name == EmbeddingModelName.VOYAGE_3_LARGE.value:
        return VoyageAIEmbeddingGenerator(model_name=model_name,
                                          input_type=dataset_type,
                                          output_dtype=output_dtype,
                                          output_dimension_size=output_dimension)
    elif model_name == EmbeddingModelName.VOYAGE_3_LITE.value:
        return VoyageAIEmbeddingGenerator(model_name=model_name,
                                          input_type=dataset_type,
                                          output_dtype=output_dtype)
    else:
        return None
