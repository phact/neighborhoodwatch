import math

import cudf
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import glob
import re
import numpy as np

from tqdm import tqdm


pattern = re.compile(r'./indices(\d+)\.parquet')

indices_table = pq.read_table(f"./indices0.parquet")
distances_table = pq.read_table(f"./distances0.parquet")

#rownum_index = indices_table.schema.get_field_index('RowNum')
#indices_table = indices_table.remove_column(rownum_index)

#rownum_index = distances_table.schema.get_field_index('RowNum')
#distances_table = distances_table.remove_column(rownum_index)

batch_size = min(10000000, len(indices_table))

batch_count = math.ceil(len(indices_table) / batch_size)

final_indices_filename = 'final_indices.parquet'
final_distances_filename = 'final_distances.parquet'

final_indices_writer = pq.ParquetWriter(final_indices_filename, indices_table.schema)
final_distances_writer = pq.ParquetWriter(final_distances_filename, distances_table.schema)


#def get_top_n_indices(row):
#    indices_to_access = row.sort_values().index[:n]
#    return combined_indices.loc[row.name].reindex(indices_to_access)


#def get_top_n_distances(row):
#    return row.sort_values()[:n]

for start in tqdm(range(0, batch_count)):

    print(f'batch: {start}')
    final_indices = pd.DataFrame()
    final_distances = pd.DataFrame()

    #final_indices = None
    #final_distances = None
    for filename in sorted(glob.glob('./indices*.parquet')):
        match = pattern.match(filename)
        print(f'filename: {filename}')
        if match:
            i = int(match.group(1))

            indices_table = pq.read_table(f"./indices{i}.parquet")
            distances_table = pq.read_table(f"./distances{i}.parquet")

            rownum_index = indices_table.schema.get_field_index('RowNum')
            indices_table = indices_table.remove_column(rownum_index)

            rownum_index = distances_table.schema.get_field_index('RowNum')
            distances_table = distances_table.remove_column(rownum_index)

            if start != batch_count:
                indices_batch = indices_table.slice(start, batch_size).to_pandas()
                distances_batch = distances_table.slice(start, batch_size).to_pandas()
            else:
                indices_batch = indices_table.slice(start, len(indices_table) - start * batch_size).to_pandas()
                distances_batch = distances_table.slice(start, len(distances_table) - start * batch_size).to_pandas()

            #indices_df = cudf.DataFrame.from_arrow(indices_batch).drop('RowNum', axis=1).to_pandas()
            #distances_df = cudf.DataFrame.from_arrow(distances_batch).drop('RowNum', axis=1).to_pandas()

            if ((final_indices.empty) & (final_distances.empty)):
                final_indices = indices_batch 
                final_distances = distances_batch
            else:

                # Concatenate the distances and indices along the column axis (axis=1)
                concatenated_distances = pd.concat([final_distances, distances_batch], axis=1)
                concatenated_indices = pd.concat([final_indices, indices_batch], axis=1)

                # Convert the DataFrame to numpy arrays for argsort
                concatenated_distances_np = concatenated_distances.values
                concatenated_indices_np = concatenated_indices.values

                # Get the sorted indices for each row in the concatenated distances DataFrame
                sorted_indices_np = np.argsort(concatenated_distances_np, axis=1)

                # Use the sorted indices to sort the distances and indices DataFrames
                rows = np.arange(sorted_indices_np.shape[0])[:, None]
                sorted_distances_np = concatenated_distances_np[rows, sorted_indices_np]
                sorted_indices_np = concatenated_indices_np[rows, sorted_indices_np]

                # Convert the numpy arrays back to DataFrames
                sorted_distances = pd.DataFrame(sorted_distances_np, index=concatenated_distances.index, columns=concatenated_distances.columns)
                sorted_indices = pd.DataFrame(sorted_indices_np, index=concatenated_indices.index, columns=concatenated_indices.columns)

                # Select the top 100 distances and corresponding indices for each row
                final_distances = sorted_distances.iloc[:, :100]
                final_indices = sorted_indices.iloc[:, :100]

                # Ensure the final distances are sorted in ascending order for each row
                assert (final_distances.apply(lambda row: row.is_monotonic_increasing, axis=1).all())

                #for row in tqdm(range(batch_size)):
                    #print(distances_df.loc[row])
                    #print(final_distances.loc[row])
                    #take the top distances from both and their corresponding indices
                    #each row is already sorted

                    #with cudf
                    #final_distances.loc[row] = cudf.concat([final_distances.loc[row], distances_df.loc[row]]).sort_values()[:100]

                    #with pandas
                #    concatenated_distances = np.concatenate([final_distances.loc[row].values, distances_batch.loc[row].values])
                #    concatenated_distances.sort()
                #    distance_arg_sort = concatenated_distances.argsort()

                #    concatenated_indices = np.concatenate([final_indices.loc[row].values, indices_batch.loc[row].values])
                #    sorted_indices= concatenated_indices[distance_arg_sort]

                #    sorted_distances = concatenated_distances[:100]
                #    sorted_indices = concatenated_indices[:100]


                    #print("sizes")
                    #print(len(final_distances.loc[row]))
                    #print(len(sorted_distances))
                #    final_distances.loc[row] = sorted_distances
                #    final_indices.loc[row] = sorted_indices


                    #print(final_distances.loc[row])
                #    assert(final_distances.loc[row].is_monotonic_increasing)
                #    assert(final_distances.loc[row][99] <= distances_batch.loc[row][99])


                #exit()



                # and then append them to the final indices and distances
                #final_indices.append(indices_df.set_index('RowNum'))
                #final_distances.append(distances_df.set_index('RowNum'))
                


    #combined_indices = cudf.concat(all_indices).to_pandas()
    #combined_distances = cudf.concat(all_distances).to_pandas()

    #n = 100

    #sorted_indices = combined_distances.apply(get_top_n_indices, axis=1)
    #sorted_distances = combined_distances.apply(get_top_n_distances, axis=1)

    # Remove duplicates
    #final_indices = sorted_indices.groupby('RowNum').first().reset_index()
    #final_distances = sorted_distances.groupby('RowNum').first().reset_index()

    final_distances['RowNum'] = range(start, start + len(final_distances))
    final_indices['RowNum'] = range(start, start + len(final_indices))

    final_indices_writer.write_table(pa.Table.from_pandas(final_indices))
    final_distances_writer.write_table(pa.Table.from_pandas(final_distances))

final_indices_writer.close()
final_distances_writer.close()
