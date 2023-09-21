#!/usr/bin/env python3

import sys
import struct

def count_vectors(filename):
    with open(filename, 'rb') as f:
        count = 0
        while True:
            dim = f.read(4)  # read dimensionality
            if not dim:
                break
            dim = struct.unpack('i', dim)[0]
            f.read(4 * dim)  # skip the vector data
            count += 1
        return count

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <filename>")
        sys.exit(1)

    filename = sys.argv[1]
    print(count_vectors(filename))

