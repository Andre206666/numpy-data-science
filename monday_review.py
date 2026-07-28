import pandas as pd
import seaborn as sns

df = sns.load_dataset("tips")

big_bills = df[df["total_bill"] > 20]
print("Big bills:")
print(big_bills.head())

avg_tip_by_day = df.groupby("day")["tip"].mean()
print("\nAverage tip by day:")
print(avg_tip_by_day)

df["tip_size"] = df["tip"].apply(lambda x: "Big" if x > 4 else "Small")
print("\nWith tip_size column:")
print(df.head())

sorted_df = df.sort_values("total_bill", ascending=False)
print("\nSorted by total_bill:")
print(sorted_df.head())

weekend_big = df[(df["day"] == "Sat") & (df["total_bill"] > 30)]
print("\nWeekend big bills:")
print(weekend_big)

import sqlite3

conn = sqlite3.connect("monday_practice.db")
cursor = conn.cursor()

cursor.execute("DROP TABLE IF EXISTS products")
cursor.execute("""
    CREATE TABLE products (
        id INTEGER PRIMARY KEY,
        category TEXT,
        price REAL
    )
""")

cursor.executemany("INSERT INTO products VALUES (?,?,?)", [
    (1, "Electronics", 500),
    (2, "Electronics", 800),
    (3, "Clothing", 50),
    (4, "Clothing", 80),
    (5, "Food", 20),
    (6, "Food", 15),
    (7, "Electronics", 1200),
])
conn.commit()

print("Products over 100:")

df1 = pd.read_sql_query("SELECT * FROM products WHERE price > 100", conn)
print(df1)

print(f"\nAverage price per category: ")
df2 = pd.read_sql("SELECT category, AVG(price) FROM products GROUP BY category", conn)
print(df2)

print(f"\nCategories with average price > 100")
df3 = pd.read_sql("SELECT category, AVG(price) FROM products GROUP BY category HAVING AVG(price) > 100", conn)
print(df3)

print("\nProducts above average price:")
df4 = pd.read_sql("SELECT * FROM products WHERE price > (SELECT AVG(price) FROM products)", conn)
print(df4)

conn.close()

