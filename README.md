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


install python dependencies using poetry

    poetry install


run nw with poetry

   poetry run nw 'data/pages_ada_002_query_data_100k_test.parquet' 10000 'data/pages_ada_002_sorted.parquet' 100000 1536 -k 100


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