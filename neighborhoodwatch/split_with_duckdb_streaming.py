import duckdb
import pyarrow as pa
import pyarrow.parquet as pq


# to start an in-memory database
con = duckdb.connect(database=':memory:')
con.sql("SELECT count(*) FROM './ada_002.parquet'").show()

con.sql("describe table './ada_002.parquet'").show()

con.execute("describe table './ada_002.parquet'")
print(con.fetchall())


con.execute("SELECT count(*) FROM './ada_002.parquet'")
table_count = con.fetchone()[0]

con.execute("SELECT ARRAY_LENGTH(embedding) FROM './ada_002.parquet' limit 1")
max_length = con.fetchone()[0]

print(f"max_length {max_length}")

select_columns = ", ".join([f"embedding[{i + 1}] AS embedding_{i}" for i in range(max_length)])
query = f"SELECT partition_key, clustering_key_0, content, title, url,  {select_columns} FROM 'ada_002.parquet'"


# Define the path for the new Parquet file
new_parquet_path = 'ada_002_split.parquet'


batch_size = 1000000 
batch_count = table_count // batch_size
print(f"table_count {table_count} batch_size {batch_size} batch_count {batch_count}")

writer = None
total_rows: int = 0
for i in range(batch_count):
    offset = i * batch_size
    if i == batch_count -1:
        limit = table_count - offset
    else:
        limit = batch_size

    result = con.execute(f'{query} OFFSET {offset} LIMIT {limit}')
#result = con.execute(f'{query}')

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
