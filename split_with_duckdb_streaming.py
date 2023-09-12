import duckdb
import pyarrow as pa
import pyarrow.parquet as pq


# to start an in-memory database
con = duckdb.connect(database=':memory:')
con.sql("SELECT count(*) FROM './pages_ada_002.parquet'").show()

con.execute("describe table './pages_ada_002.parquet'")
print(con.fetchall())


con.execute("SELECT ARRAY_LENGTH(embedding) FROM './pages_ada_002.parquet' limit 1")
max_length = con.fetchone()[0]

print(f"max_length {max_length}")

select_columns = ", ".join([f"embedding[{i + 1}] AS embedding_{i}" for i in range(max_length)])
query = f"SELECT *, {select_columns} FROM 'pages_ada_002.parquet'"


# Define the path for the new Parquet file
new_parquet_path = 'pages_ada_002_split.parquet'

copy_command = f"{query}"
result = con.execute(copy_command)

record_batch_reader = result.fetch_record_batch()

writer = pq.ParquetWriter(where=new_parquet_path, schema=record_batch_reader.schema)

chunk = record_batch_reader.read_next_batch()
total_rows: int = 0
while len(chunk) > 0:
    chunk = record_batch_reader.read_next_batch()
    total_rows += chunk.num_rows
    writer.write_batch(batch=chunk)
    print(f"Wrote batch of {chunk.num_rows:,d} row(s) - total row(s) written thus far: {total_rows:,d}")
