import duckdb
# to start an in-memory database
con = duckdb.connect(database=':memory:')
con.sql("SELECT count(*) FROM './sorted_out.parquet'").show()

con.execute("describe table './sorted_out.parquet'")
print(con.fetchall())


con.execute("SELECT ARRAY_LENGTH(embedding) FROM 'sorted_out.parquet' limit 1")
max_length = con.fetchone()[0]

print(f"max_length {max_length}")

select_columns = ", ".join([f"embedding[{i + 1}] AS embedding_{i}" for i in range(max_length)])
query = f"SELECT *, {select_columns} FROM 'sorted_out.parquet'"


# Define the path for the new Parquet file
new_parquet_path = 'split.parquet'

# Use the COPY command to write the result to the new Parquet file
copy_command = f"COPY ({query}) TO '{new_parquet_path}' WITH (FORMAT 'PARQUET')"
con.execute(copy_command)
