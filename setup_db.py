import sqlite3
import pandas as pd

# 1. Read your downloaded CSV file
df = pd.read_csv("data.csv", encoding="latin1")

# 2. Connect to our local SQLite database
conn = sqlite3.connect("database.db")

# 3. Write the entire dataset to a SQL table named 'sales'
df.to_sql("sales", conn, if_exists="replace", index=False)
conn.close()

print("Real data successfully loaded into SQLite!")