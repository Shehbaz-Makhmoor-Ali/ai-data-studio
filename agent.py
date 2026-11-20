import os
import pandas as pd
import plotly.express as px
import google.generativeai as genai

class SQLAgent:
    """Agent 1: Responsible for generating and self-correcting SQL queries with conversational memory."""
    def __init__(self, api_key, schema_info, table_names):
        self.api_key = api_key
        self.schema = schema_info
        self.tables = table_names
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash-002')
        else:
            self.model = None

    def generate_and_fix_sql(self, user_prompt: str, run_query_func, history: list = None, max_retries=2):
        if not self.model:
            return None, "SELECT 1;", "Gemini API key not initialized. Please check your .env file for GEMINI_API_KEY."
        
        error_message = None
        sql_query = ""
        
        history_context = ""
        if history:
            recent_history = history[-3:]
            history_context = "\nConversation History (for context):\n"
            for entry in recent_history:
                history_context += f"User: {entry['question']}\nPrevious SQL: {entry['sql']}\n"
        
        for attempt in range(max_retries):
            try:
                error_context = f"\nPrevious Error encountered: {error_message}. Fix the query." if error_message else ""
                
                # Combined prompt for Gemini
                prompt_text = f"""You are an expert PostgreSQL autonomous agent. Available tables in this database: {self.tables}. 
Output ONLY a valid SQL query starting with SELECT and nothing else. No markdown formatting, no explanations. Use proper double quotes for table and column names if they have mixed case.

Database Schema:
{self.schema}
{history_context}
Rules:
1. Use the correct table and column names.
2. Handle case-sensitivity with double quotes.
3. If you use aggregate functions (MAX, SUM, etc.), all other selected columns MUST be in a GROUP BY clause.
4. To find the 'highest' or 'lowest' item alongside its details, use ORDER BY [column] DESC LIMIT 1 instead of MAX().
5. Use the Conversation History to understand follow-up questions.
{error_context}

Question: {user_prompt}
SQL Query:"""
                
                response = self.model.generate_content(prompt_text, generation_config={"temperature": 0.0})
                sql_query = response.text.strip()
                sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
                
                if "SELECT" in sql_query.upper():
                    select_idx = sql_query.upper().find("SELECT")
                    sql_query = sql_query[select_idx:]
                
                df, _, err = run_query_func(sql_query)
                if not err:
                    return df, sql_query, None
                else:
                    error_message = err
            except Exception as e:
                error_message = str(e)
                
        return None, sql_query, error_message

class VisualizationAgent:
    """Agent 2: Inspects dynamic data frames and builds smart Plotly charts matched to the Reddish-Pink theme."""
    @staticmethod
    def render_smart_chart(df: pd.DataFrame, user_prompt: str):
        if len(df) == 0 or len(df.columns) < 2:
            return None
            
        try:
            cols = df.columns
            x_col = cols[0]
            numeric_cols = [c for c in cols[1:] if pd.api.types.is_numeric_dtype(df[c])]
            
            if not numeric_cols: 
                return None
                
            y_col = numeric_cols[0]
            prompt_lower = user_prompt.lower()

            if any(kw in prompt_lower for kw in ["top", "highest", "product", "item", "customer", "rank"]) and len(df) <= 20:
                fig = px.bar(df, x=y_col, y=x_col, orientation='h', color=y_col, color_continuous_scale="Reds", text_auto=True)
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
            elif any(kw in prompt_lower for kw in ["share", "distribution", "proportion", "versus", "vs", "count of"]) and len(df) <= 10:
                fig = px.pie(df, names=x_col, values=y_col, hole=0.4, color_discrete_sequence=['#E11D48', '#F43F5E', '#FB7185', '#FDA4AF'])
            else:
                fig = px.bar(df, x=x_col, y=y_col, color=y_col, color_continuous_scale="Reds", text_auto=True)
                fig.update_layout(xaxis_tickangle=-45)

            fig.update_layout(
                title=dict(text=f"Visual Analysis: '{user_prompt}'", font=dict(size=14, color="#0F172A", family="Inter")),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#475569", family="Inter", size=12),
                margin=dict(l=20, r=20, t=50, b=20),
                xaxis=dict(gridcolor="#FFF1F2", zerolinecolor="#FFF1F2", tickfont=dict(color="#475569")),
                yaxis=dict(gridcolor="#FFF1F2", zerolinecolor="#FFF1F2", tickfont=dict(color="#475569"))
            )
            return fig
        except Exception:
            return None

class InsightAgent:
    """Agent 3: Analyzes the resulting dataframe and STREAMs a short natural language answer live."""
    def __init__(self, api_key):
        self.api_key = api_key
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash-002')
        else:
            self.model = None
        
    def generate_short_answer_stream(self, user_prompt: str, df: pd.DataFrame):
        if not self.model or df is None or df.empty: 
            yield ""
            return
            
        try:
            data_context = df.head(5).to_string(index=False)
            
            prompt_text = f"""You are a professional data analyst. Based on the user's question and the data provided, write a direct, short, and natural 1-2 sentence answer. Do NOT explain how you got the answer, do NOT show SQL. Just answer the question clearly using the data.

Question: {user_prompt}

Data Result:
{data_context}

Short Answer:"""
            
            response = self.model.generate_content(prompt_text, stream=True, generation_config={"temperature": 0.1})
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            yield f"Data processed successfully. (Error: {str(e)})"