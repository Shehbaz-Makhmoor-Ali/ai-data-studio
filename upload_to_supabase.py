import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

print("Loading CSV file with latin-1 encoding...")
df = pd.read_csv("data.csv", encoding="latin-1") 

DATABASE_URL = os.getenv("SUPABASE_DATABASE_URL")
engine = create_engine(DATABASE_URL)

print("Uploading to Supabase in chunks (this will take a moment)...")
df.to_sql("sales", engine, if_exists="replace", index=False, chunksize=10000)
print("Upload completed successfully!")