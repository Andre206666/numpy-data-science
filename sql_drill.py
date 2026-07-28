import sqlite3
import pandas as pd

conn = sqlite3.connect("drill.db")
cursor = conn.cursor()
cursor.execute("DROP TABLE IF EXISTS sales")
cursor.execute("CREATE TABLE sales (id INTEGER, region TEXT, amount REAL)")
cursor.executemany("INSERT INTO sales VALUES (?,?,?)", [
    (1, "North", 500), (2, "South", 300), (3, "North", 800),
    (4, "East", 200), (5, "South", 600), (6, "East", 900)
])
conn.commit()

query1 = "SELECT * FROM sales WHERE amount > 400"
df1 = pd.read_sql_query(query1, conn)
print(df1)

query2 = "SELECT region, AVG(amount) FROM sales GROUP BY region"
df2 = pd.read_sql_query(query2, conn)
print(df2)

query3 = "SELECT region, AVG(amount) FROM sales GROUP BY region HAVING AVG(amount) > 400"
df3 = pd.read_sql_query(query3, conn)
print(df3)

query4 = "SELECT * FROM sales WHERE amount > (SELECT AVG(amount) FROM sales)"
df4 = pd.read_sql_query(query4, conn)
print(df4)