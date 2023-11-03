## NeighborhoodWatch

NeighborhoodWatch (`nw`) is a GPU powered brute force knn ground truth dataset generator

### Set Up the Environment

**NOTE**: Please refer to [install_env.sh](bash/install_env.sh) for more information about the environment setup details.
* This script has been verified on AWS `p3.8xlarge` instance type with `Ubuntun 22.04` OS

---

At high level, in order to run this program, the following prerqusites need to be satsified:
* One computing instance with Nividia GPU (e.g. AWS `p3.8xlarge` instance type)
* Nivdia CUDA toolkit and driver 12 installed ([link](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html))
* Nividia cuDNN library installed ([link](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html))
* Nividia NCCL library installed ([link](https://docs.nvidia.com/deeplearning/nccl/install-guide/index.html))
* Python version 3.10
   * A python virtual environemtn (e.g. MiniConda) is highly recommended
* Poetry Python dependency management

### Run the Program

Run `nw` with poetry. Please see the `usage` description (as below) for the available CLI parameters.
```
poetry run nw <list of parameters>
```

`usage`:

```
$ poetry run nw -h
usage: nw [-h] [-k K] [-m MODEL_NAME] [-d DATA_DIR] [--enable-memory-tuning] [--disable-memory-tuning] [--gen-hdf5 | --no-gen-hdf5] [--validation | --no-validation] query_count base_count dimensions

nw (neighborhood watch) uses GPU acceleration to generate ground truth KNN datasets

positional arguments:
  query_count           number of query vectors to generate
  base_count            number of base vectors to generate
  dimensions            number of dimensions for the embedding model

options:
  -h, --help            show this help message and exit
  -k K, --k K
  -m MODEL_NAME, --model_name MODEL_NAME
  -d DATA_DIR, --data_dir DATA_DIR
                        Directory to store the generated data (default: knn_dataset)
  --enable-memory-tuning
                        Enable memory tuning
  --disable-memory-tuning
                        Disable memory tuning (useful for very small datasets)
  --gen-hdf5, --no-gen-hdf5
                        Generate hdf5 files (default: True) (default: True)
  --validation, --no-validation
                        Validate the generated files (default: False) (default: False)

Some example commands:

    nw 10000 10000 1024 -k 100 -m 'intfloat/e5-large-v2' --disable-memory-tuning
    nw 10000 10000 768 -k 100 -m 'textembedding-gecko' --disable-memory-tuning
    nw 10000 10000 384 -k 100 -m 'intfloat/e5-small-v2' --disable-memory-tuning
    nw 10000 10000 768 -k 100 -m 'intfloat/e5-base-v2' --disable-memory-tuning
```

### Generated Datasets

After the program is successfully run, it will generate a set of data sets under a spcified folder which is default to `knn_dataset` subfolder. 
You can override the output directory using the `--data_dir <dir_name>` option.

In particular, the following datasets include the KNN ground truth results:
| file format | dataset name | dataset file | 
| ----------- | ------------ | ------------ | 
| `fvec` | `train` dataset (base) | `<model_name>_<base_count>_base_vectors` |
| `fvec` | `test` dataset (query)| `<model_name>_<base_count>_query_vectors_<query_count>` |
| `fvec` | `distances` dataset (distances) | `<model_name>_<base_count>_distances_<query_count>` |
| `ivec` | `neighors` dataset (indices) | `<model_name>_<base_count>_indices_query_<query_count>` |
| `hdf5` | consolidated hdf5 dataset of the above 4 datasets | `<model_name>_base_<base_count>_query_<query_count>` |

### Run the Tests

```
poetry run pytest
```

#### cli:

![cli](docs/cli.png)

#### nvtop:

![nvtop](docs/nvtop.png)
