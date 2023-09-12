import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

split_path = 'pages_ada_002_split.parquet'

# to start an in-memory database
con = duckdb.connect(database=':memory:')
con.sql(f"SELECT count(*) FROM './{split_path}'").show()

con.sql(f"describe table './{split_path}'").show()

con.execute(f"describe table './{split_path}'")
print(con.fetchall())


con.execute(f"SELECT count(*) FROM './{split_path}'")
table_count = con.fetchone()[0]

query = f"SELECT * FROM '{split_path}' order by partition_key, clustering_key_0"


# Define the path for the new Parquet file


batch_size = 1000000 
batch_count = table_count // batch_size
print(f"table_count {table_count} batch_size {batch_size} batch_count {batch_count}")

new_parquet_path='pages_ada_002_sorted.parquet'

writer = None
total_rows: int = 0
batch_count = 1
for i in range(batch_count):
    offset = i * batch_size
    if i == batch_count -1:
        limit = table_count - offset
    else:
        limit = batch_size

    #result = con.execute(f'{query} OFFSET {offset} LIMIT {limit}')
    result = con.execute(f'{query}')

    record_batch_reader = result.fetch_record_batch()

    if writer == None:
        writer = pq.ParquetWriter(where=new_parquet_path, schema=record_batch_reader.schema)

    chunk = record_batch_reader.read_next_batch()
    while len(chunk) > 0:
        try:
            total_rows += chunk.num_rows
            writer.write_batch(batch=chunk)
            print(f"Wrote batch of {chunk.num_rows:,d} row(s) - total row(s) written thus far: {total_rows:,d}")
            chunk = record_batch_reader.read_next_batch()
        except StopIteration:
            print('Fetched all chunks for this batch')
            break
