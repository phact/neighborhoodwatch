import duckdb
# to start an in-memory database
con = duckdb.connect(database=':memory:')

print("--------------------------------------DISTANCES")
con.execute("select * from './final_distances.parquet' limit 10")
print(con.fetchall())

print("--------------------------------------INDICES")
con.execute("select * from './final_indices.parquet' limit 10")
print(con.fetchall())


con.execute("describe table './final_indices.parquet'")
print(con.fetchall())

print("--------------------------------------FINAL")
con.sql("select * from './final_indices.parquet' as indices limit 10").show()
con.sql("select * from './final_distances.parquet' as distances limit 10").show()

con.sql("select count(*) from './final_indices.parquet' limit 10").show()
con.sql("select count(*) from './final_distances.parquet' limit 10").show()
print("--------------------------------------CHUNKS")
con.sql("select * from './indices0.parquet' limit 10").show()
con.sql("select * from './distances0.parquet' limit 10").show()
con.sql("select count(*) from './indices0.parquet' limit 10").show()
con.sql("select count(*) from './distances0.parquet' limit 10").show()
con.sql("select * from './indices1.parquet' limit 10").show()
con.sql("select * from './distances1.parquet' limit 10").show()
con.sql("select count(*) from './indices1.parquet' limit 10").show()
con.sql("select count(*) from './distances1.parquet' limit 10").show()
con.sql("select * from './indices2.parquet' limit 10").show()
con.sql("select * from './distances2.parquet' limit 10").show()
con.sql("select count(*) from './indices2.parquet' limit 10").show()
con.sql("select count(*) from './distances2.parquet' limit 10").show()
con.sql("select * from './indices3.parquet' limit 10").show()
con.sql("select * from './distances3.parquet' limit 10").show()
con.sql("select count(*) from './indices3.parquet' limit 10").show()
con.sql("select count(*) from './distances3.parquet' limit 10").show()
#con.sql("select * from './indices4.parquet' limit 10").show()
#con.sql("select * from './distances4.parquet' limit 10").show()
#con.sql("select count(*) from './indices4.parquet' limit 10").show()
#con.sql("select count(*) from './distances4.parquet' limit 10").show()








con.sql("select * from './final_indices.parquet' where RowNum = 4314").show()
con.sql("select * from './final_distances.parquet' where RowNum = 4314").show()
