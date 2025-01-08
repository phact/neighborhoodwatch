import h5py
import numpy as np
from collections import Counter
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description="Check for duplicates in HDF5 datasets.")
parser.add_argument("file_path", type=str, help="Path to the HDF5 file")
args = parser.parse_args()

# Open the HDF5 file
file_path = args.file_path

# Open the HDF5 file in read mode
with h5py.File(file_path, "r") as hdf:

    # Check the structure of the file
    print("Available datasets in the file:")
    hdf.visit(print)

    # Choose a dataset you want to check for duplicates
    dataset_names = ["train", "test"]

    for dataset_name in dataset_names:
        if dataset_name in hdf:
            dataset = hdf[dataset_name][:]
            
            # If the dataset is structured (not just flat arrays), specify a type or column
            # For example, for structured data like records or named fields:
            # dataset = dataset['your_field_name']
            
            data_array = np.array(dataset)
            
            # Check for duplicate rows
            unique_rows, counts = np.unique(data_array, axis=0, return_counts=True)
            
            # Find duplicates
            duplicates = unique_rows[counts > 1]
            
            if len(duplicates) > 0:
                print(f"Found {len(unique_rows)} unique rows:")
                print(f"Found {len(duplicates)} duplicate rows:")
                print(duplicates)
                print(f"out of {len(data_array)} rows")
            else:
                print("No duplicate rows found.")

        else:
            print(f"Dataset '{dataset_name}' not found in the file.")

