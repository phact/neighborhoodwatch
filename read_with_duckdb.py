import duckdb
# to start an in-memory database
con = duckdb.connect(database=':memory:')

print("--------------------------------------DISTANCES0")
con.execute("select * from './distances0.parquet' limit 10")
print(con.fetchall())

print("--------------------------------------INDICES0")
con.execute("describe table './indices0.parquet'")
print(con.fetchall())
con.execute("select * from './indices0.parquet' limit 10")
print(con.fetchall())

#print("--------------------------------------SORTED")
#con.execute("select seq_no, content from './sorted_out.parquet' limit 10")
#print(con.fetchall())


print("--------------------------------------COUNTS")
con.sql("select count(*) from './distances*.parquet' ").show()
con.sql("select count(*) from './indices*.parquet' ").show()


#con.sql("select * from './distances1.parquet' where RowNum=4314515 ").show()
#con.sql("select * from './indices1.parquet' where RowNum=4314515 ").show()
