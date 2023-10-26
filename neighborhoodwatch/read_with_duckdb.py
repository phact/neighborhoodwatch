import json
import duckdb

# to start an in-memory database
con = duckdb.connect(database=':memory:')

def log(data):
    print(json.dumps(data, indent=4))




print("--------------------------------------DISTANCES0")
con.execute("describe table '../distances0.parquet'")
log(con.fetchall())
con.execute("select * from '../distances0.parquet' limit 10")
log(con.fetchall())

print("--------------------------------------INDICES0")
con.execute("describe table '../indices0.parquet'")
log(con.fetchall())
con.execute("select * from '../indices0.parquet' limit 10")
log(con.fetchall())

#print("--------------------------------------SORTED")
#con.execute("select seq_no, content from './sorted_out.parquet' limit 10")
#print(con.fetchall())


print("--------------------------------------COUNTS")
con.sql("select count(*) from '../distances*.parquet' ").show()
con.sql("select count(*) from '../indices*.parquet' ").show()

print("--------------------------------------FINAL DISTANCES")
con.execute("describe table '../final_distances.parquet'")
log(con.fetchall())
con.execute("select * from '../final_distances.parquet' limit 10")
log(con.fetchall())

print("--------------------------------------FINAL INDICES")
con.execute("describe table '../final_indices.parquet'")
log(con.fetchall())
con.execute("select * from '../final_indices.parquet' limit 10")
log(con.fetchall())


print("--------------------------------------FINAL COUNTS")
con.sql("select count(*) from '../final_indices.parquet' ").show()
con.sql("select count(*) from '../final_distances.parquet' ").show()


#con.sql("select * from './distances1.parquet' where RowNum=4314515 ").show()
#con.sql("select * from './indices1.parquet' where RowNum=4314515 ").show()
con.sql("select count(*) from '/home/tato/Desktop/neighborhoodwatch/ada_002_sorted.parquet'").show()
con.sql("select count(*) from '/home/tato/Desktop/neighborhoodwatch/ada_002_query_data_100k_test.parquet'").show()

