### neighborhoodwatch

gpu powered brute force knn ground truth dataset generator


### to run

install cuda toolkit 12 per https://developer.nvidia.com/cuda-downloads


```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda
```

to generate local embeddings install cudnn https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html and nccl per https://docs.nvidia.com/deeplearning/nccl/install-guide/index.html#debian

install poetry per https://python-poetry.org/docs/#installation

    curl -sSL https://install.python-poetry.org | python3 -

install python dependencies using poetry:

    poetry install

run nw with poetry:

    poetry run nw 10000 100000 1536 -k 100 --enable-memory-tuning

or for local embeddings:

    poetry run nw 10000 100000 384 -k 100 -m 'intfloat/multilingual-e5-small' --enable-memory-tuning


usage:


```
$ poetry run nw -h
usage: nw [-h] [-k K] [--enable-memory-tuning] [--disable-memory-tuning] query_filename query_count sorted_data_filename base_count dimensions

nw (neighborhood watch) uses GPU acceleration to generate ground truth KNN datasets

positional arguments:
  query_filename
  query_count
  sorted_data_filename
  base_count
  dimensions

options:
  -h, --help            show this help message and exit
  -k K, --k K
  --enable-memory-tuning
                        Enable memory tuning
  --disable-memory-tuning
                        Disable memory tuning (useful for very small datasets)

```

### to run tests

    poetry run pytest

#### cli:

![cli](docs/cli.png)

#### nvtop:

![nvtop](docs/nvtop.png)
