import struct
import numpy as np
import pandas as pd


def read_fvec(filename):
    vectors = []

    with open(filename, "rb") as f:
        while True:
            dim_data = f.read(4)
            if not dim_data:
                break

            dim = struct.unpack('i', dim_data)[0]
            vector_data = f.read(4 * dim)
            vector = struct.unpack('f' * dim, vector_data)
            vectors.append(vector)

    return pd.DataFrame(np.array(vectors))


def read_ivec(filename):
    vectors = []

    with open(filename, "rb") as f:
        while True:
            dim_data = f.read(4)
            if not dim_data:
                break

            dim = struct.unpack('i', dim_data)[0]
            vector_data = f.read(4 * dim)
            vector = struct.unpack('i' * dim, vector_data)
            vectors.append(vector)

    return pd.DataFrame(np.array(vectors))


def main():
    filename = '/path/to/your/fvec/file.fvec'
    query_df = read_fvec(filename)
    print(f"query fvec file: {filename}")
    print(f"shape: {query_df.shape}")
    print(f"{query_df.head()}")


if __name__ == "__main__":
    main()
