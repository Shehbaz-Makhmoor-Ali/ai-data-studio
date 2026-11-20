import os
import re
import streamlit as st
import pandas as pd
import psycopg2
from sqlalchemy import create_engine
import plotly.express as px
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load environment variables
load_dotenv()

# Configure Hugging Face Inference Client
HF_TOKEN = os.getenv("HF_TOKEN")
hf_client = None
if HF_TOKEN:
    hf_client = InferenceClient(model="defog/sqlcoder-7b-2", token=HF_TOKEN)

# Page Configuration
st.set_page_config(
    page_title="Autonomous SQL Data Analyst Copilot",
    page_icon="🤖",
    layout="wide"
)

# Database Connection setup using Supabase / PostgreSQL
@st.cache_resource
def get_db_engine():
    db_url = os.getenv("SUPABASE_DATABASE_URL")
    if not db_url:
        st.error("SUPABASE_DATABASE_URL is missing in environment variables.")
        st.stop()
    if db_url.startswith("postgresql://"):
        db_url = db_url.replace("postgresql://", "postgresql+psycopg2://")
    return create_engine(db_url)

engine = get_db_engine()

# Helper function to run SQL queries safely
def run_query(query: str):
    try:
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
        return df, query, None
    except Exception as e:
        return None, query, str(e)

# Function to generate SQL using SQLCoder Model via Hugging Face
def generate_sql_with_hf(user_prompt: str, schema_info: str) -> str:
    if not hf_client:
        st.error("❌ Hugging Face Client is not initialized. Check your HF_TOKEN in .env")
        return None
    
    try:
        prompt = f"""### Task
Generate a PostgreSQL query to answer the following question: `{user_prompt}`

### Database Schema
The database contains a table named `SalesData` with the following columns:
{schema_info}

### Rules
1. The table name must strictly be `"SalesData"`.
2. Use double quotes for column names (e.g. "Country", "Quantity", "UnitPrice", "InvoiceNo", "Description").
3. Output ONLY the raw SQL query without any markdown formatting, backticks, or explanation.

### SQL
SELECT"""

        response = hf_client.text_generation(
            prompt, 
            max_new_tokens=200, 
            temperature=0.0,
            return_full_text=False
        )
        
        sql_query = "SELECT " + response.strip()
        sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
        
        if "SELECT" in sql_query.upper():
            select_idx = sql_query.upper().find("SELECT")
            sql_query = sql_query[select_idx:]
        else:
            return None
            
        return sql_query
    except Exception as e:
        st.error(f"❌ Hugging Face API Error: {e}")
        return None

# Sidebar - Schema Explorer & Navigation
st.sidebar.title("🛠️ Control Panel")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", ["💬 SQL Chat Copilot", "📊 Executive Analytics Dashboard"])

st.sidebar.markdown("---")
st.sidebar.subheader("📂 Database Schema")
schema_summary = ""
try:
    schema_query = "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'SalesData';"
    schema_df, _, err = run_query(schema_query)
    
    if err or schema_df is None or schema_df.empty:
        st.sidebar.error("Could not fetch schema.")
    else:
        schema_lines = []
        for _, row in schema_df.iterrows():
            schema_lines.append(f'- "{row["column_name"]}" ({row["data_type"]})')
            st.sidebar.text(f"• {row['column_name']} ({row['data_type']})")
        schema_summary = "\n".join(schema_lines)
except Exception as e:
    st.sidebar.warning("Connect to Supabase to view schema.")

# Main Page Routing
if page == "💬 SQL Chat Copilot":
    st.title("🤖 Autonomous SQL Data Analyst Copilot")
    st.caption("Chat with your retail sales database using natural language powered by SQLCoder.")
    st.markdown("---")

    # Chat interface container
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "df" in message and message["df"] is not None:
                st.dataframe(message["df"], use_container_width=True)

    user_prompt = st.chat_input("Ask any question about your sales data (e.g., total sell, top countries)...")

    if user_prompt:
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            with st.spinner("SQLCoder AI is generating SQL..."):
                
                ai_sql = generate_sql_with_hf(user_prompt, schema_summary)
                
                if ai_sql:
                    sql_query = ai_sql
                else:
                    sql_query = """
                        SELECT "InvoiceNo", "InvoiceDate", "CustomerID", "Country", ("Quantity" * "UnitPrice") as revenue 
                        FROM "SalesData" 
                        LIMIT 10;
                    """
                
                df_result, final_executed_query, query_error = run_query(sql_query)
                
                st.code(final_executed_query, language="sql")

                if query_error:
                    response_text = f"❌ Error executing query: {query_error}"
                    st.error(response_text)
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                else:
                    response_text = "Here are the results based on your query:"
                    st.markdown(response_text)
                    st.dataframe(df_result, use_container_width=True)
                    
                    # CSV Export
                    csv_data = df_result.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv_data,
                        file_name="query_results.csv",
                        mime="text/csv"
                    )

                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response_text, 
                        "df": df_result
                    })

elif page == "📊 Executive Analytics Dashboard":
    st.title("📊 Retail Executive Analytics")
    st.caption("High-level performance metrics derived live from Supabase.")
    st.markdown("---")

    kpi_query = """
        SELECT 
            COUNT(DISTINCT "InvoiceNo") as total_orders,
            COUNT(DISTINCT "CustomerID") as total_customers,
            SUM("Quantity" * "UnitPrice") as gross_revenue
        FROM "SalesData" 
        WHERE "Quantity" > 0;
    """
    kpi_df, _, err = run_query(kpi_query)

    if err:
        st.error(f"Error loading dashboard metrics: {err}")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("📦 Total Orders", f"{int(kpi_df['total_orders'].iloc[0]):,}")
        col2.metric("👥 Active Customers", f"{int(kpi_df['total_customers'].iloc[0]):,}")
        col3.metric("💰 Gross Revenue", f"${kpi_df['gross_revenue'].iloc[0]:,.2f}")

    st.markdown("---")
    st.subheader("🌍 Top Countries by Revenue")
    
    chart_query = """
        SELECT "Country", SUM("Quantity" * "UnitPrice") AS revenue
        FROM "SalesData"
        WHERE "Quantity" > 0 AND "Country" != 'United Kingdom'
        GROUP BY "Country"
        ORDER BY revenue DESC
        LIMIT 10;
    """
    chart_df, _, err2 = run_query(chart_query)
    if err2:
        st.error(err2)
    else:
        fig = px.bar(chart_df, x='Country', y='revenue', title="Top 10 Countries (Excluding UK)", color='revenue', color_continuousScale='Viridis')
        st.plotly_chart(fig, use_container_width=True)