import sqlite3
from langchain_core.tools import tool

@tool
def run_sqlite_query(query: str) -> str:
    """
    Executes a SQL query against the SQLite database and returns the result.
    If the query has a syntax error, it returns the error message.
    """
    print(f"\n--- TOOL TRIGGERED: Running SQL ---\nQuery: {query}\n-----------------------------------")
    
    try:
        conn = sqlite3.connect("database.db")
        cursor = conn.cursor()
        cursor.execute(query)
        result = cursor.fetchall()
        conn.close()
        return str(result)
    except Exception as e:
        return f"Error executing query: {e}"