import os
import re
import json
import time
import pandas as pd
import numpy as np
import scipy.stats as stats
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text
import google.generativeai as genai
import sqlglot
from sqlglot import exp

# Load environment variables, overriding any cached values
load_dotenv(override=True)

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Data Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Cloud mesh gradient styling */
    .stApp {
        background: radial-gradient(circle at top left, #fff1f2 0%, #ffffff 50%, #ffe4e6 100%);
        color: #1e293b;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Sidebar background & border */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #f1f5f9;
    }

    /* Accent headers */
    h1, h2, h3, h4 {
        color: #0f172a;
        font-weight: 700;
    }

    /* Prominent squarish chat input */
    .stChatInputContainer {
        border-radius: 8px !important;
        border: 1.5px solid #e2e8f0 !important;
        background-color: #ffffff !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05) !important;
        transition: border-color 0.2s ease;
    }
    .stChatInputContainer:focus-within {
        border-color: #e11d48 !important;
    }

    /* Chat messages */
    .stChatMessage {
        border-radius: 10px;
        padding: 12px 18px;
        margin-bottom: 12px;
        border: 1px solid #f1f5f9;
        background-color: #ffffff;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    }

    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        border: 1px solid #e2e8f0;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        border-color: #e11d48;
        color: #e11d48;
    }

    /* Empty state hero text */
    .hero-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        margin-top: 15vh;
        text-align: center;
    }
    .hero-title {
        font-size: 3.2rem;
        font-weight: 800;
        color: #0f172a;
        letter-spacing: -0.03em;
    }
    .hero-accent {
        color: #e11d48;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        color: #64748b;
        margin-top: 10px;
        max-width: 520px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Security AST Guardrail & Pre-Flight Intent Filter
# ---------------------------------------------------------
class SQLGuardrail:
    FORBIDDEN_OPERATIONS = (
        exp.Drop,
        exp.Delete,
        exp.Update,
        exp.Insert,
        exp.Alter,
        exp.Create,
    )

    @classmethod
    def check_prompt_intent(cls, user_prompt: str):
        """
        Blocks destructive intent immediately before calling the LLM or database.
        """
        destructive_pattern = r"\b(delete|drop|truncate|alter|update|insert\s+into)\b"
        if re.search(destructive_pattern, user_prompt, re.IGNORECASE):
            return False, (
                "🔒 **Action Blocked (Read-Only Mode):** I cannot execute `DELETE`, `DROP`, `UPDATE`, "
                "`INSERT`, or `ALTER` operations. AI Data Studio is strictly read-only to protect your database."
            )
        return True, None

    @classmethod
    def validate_and_sanitize(cls, raw_sql: str, target_dialect: str = "postgres"):
        """
        Parses SQL into an AST, ensures strictly read-only SELECT/UNION operations,
        and transpiles to the target dialect.
        """
        try:
            parsed = sqlglot.parse_one(raw_sql, read=target_dialect)
            if not parsed:
                return None, "Query could not be parsed into a valid AST."

            # Verify no destructive DML/DDL nodes exist anywhere in the syntax tree
            for node in parsed.walk():
                if isinstance(node, cls.FORBIDDEN_OPERATIONS):
                    return None, f"Security Violation: Destructive statement '{node.key.upper()}' blocked by AST Guardrail."

            # Verify root statement is a read-only query
            if not isinstance(parsed, (exp.Select, exp.Union)):
                return None, "Security Violation: Only SELECT/UNION statements are permitted."

            sanitized_sql = parsed.sql(dialect=target_dialect)
            return sanitized_sql, None
        except sqlglot.errors.ParseError as e:
            return None, f"AST Syntax Parse Error: {str(e)}"
        except Exception as e:
            return None, f"Guardrail Error: {str(e)}"

# ---------------------------------------------------------
# Semantic Business Rules Layer
# ---------------------------------------------------------
class SemanticLayer:
    def __init__(self, filepath: str = "semantic_layer.json"):
        self.filepath = filepath
        self.rules = self._load_rules()

    def _load_rules(self) -> dict:
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def get_prompt_context(self) -> str:
        if not self.rules:
            return "No custom semantic business rules defined."
        return json.dumps(self.rules, indent=2)

# ---------------------------------------------------------
# Statistical Diagnostics Engine
# ---------------------------------------------------------
class StatisticalDiagnosticsEngine:
    @staticmethod
    def run_diagnostics(df: pd.DataFrame) -> dict:
        """
        Performs automated hypothesis tests, anomaly scans, and parametric diagnostics.
        """
        diagnostics = {}
        if df.empty or len(df) < 3:
            return diagnostics

        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return diagnostics

        # 1. Outlier & Anomaly Detection via IQR
        outliers_detected = {}
        for col in numeric_df.columns:
            clean_series = numeric_df[col].dropna()
            if len(clean_series) >= 4:
                q25, q75 = np.percentile(clean_series, [25, 75])
                iqr = q75 - q25
                lower_bound = q25 - (1.5 * iqr)
                upper_bound = q75 + (1.5 * iqr)
                num_outliers = int(((clean_series < lower_bound) | (clean_series > upper_bound)).sum())
                if num_outliers > 0:
                    outliers_detected[col] = num_outliers
        if outliers_detected:
            diagnostics["anomalies_iqr"] = outliers_detected

        # 2. Automated Correlation Matrix & Significance
        if len(numeric_df.columns) >= 2:
            cols = numeric_df.columns[:2]
            valid_subset = numeric_df[cols].dropna()
            if len(valid_subset) >= 4:
                r, p_val = stats.pearsonr(valid_subset[cols[0]], valid_subset[cols[1]])
                diagnostics["correlation"] = {
                    "features": f"{cols[0]} vs {cols[1]}",
                    "pearson_r": round(float(r), 3),
                    "p_value": round(float(p_val), 5),
                    "statistically_significant": bool(p_val < 0.05),
                }

        # 3. Distribution Metrics (Skewness)
        skewness = {}
        for col in numeric_df.columns:
            clean_col = numeric_df[col].dropna()
            if len(clean_col) >= 4:
                skewness[col] = round(float(stats.skew(clean_col)), 2)
        if skewness:
            diagnostics["skewness"] = skewness

        return diagnostics

# ---------------------------------------------------------
# Gemini Model Auto-Discovery
# ---------------------------------------------------------
def initialize_gemini():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None, "GEMINI_API_KEY is not set in environment or .env."

    genai.configure(api_key=api_key)
    candidate_models = ["models/gemini-2.5-flash", "models/gemini-1.5-flash", "models/gemini-1.5-pro"]

    try:
        available_models = [m.name for m in genai.list_models() if "generateContent" in m.supported_generation_methods]
        selected_model = None
        for candidate in candidate_models:
            if candidate in available_models:
                selected_model = candidate
                break

        if not selected_model:
            selected_model = available_models[0] if available_models else "models/gemini-1.5-flash"

        return genai.GenerativeModel(selected_model), selected_model.replace("models/", "")
    except Exception:
        return genai.GenerativeModel("gemini-1.5-flash"), "gemini-1.5-flash"

# ---------------------------------------------------------
# Database Engine & Schema Inspector
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_db_engine(uri: str):
    return create_engine(
        uri,
        pool_pre_ping=True,
        connect_args={
            "connect_timeout": 10,
            "options": "-c statement_timeout=15000",  # 15 seconds max per query
        },
    )

def fetch_schema_context(engine) -> str:
    inspector = inspect(engine)
    schema_details = []
    for table_name in inspector.get_table_names():
        columns = inspector.get_columns(table_name)
        col_strs = [f"{col['name']} ({str(col['type'])})" for col in columns]
        schema_details.append(f"Table '{table_name}': {', '.join(col_strs)}")
    return "\n".join(schema_details)

def execute_query(engine, sql: str):
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text(sql), conn)
        return df, None
    except Exception as e:
        return None, str(e)

# ---------------------------------------------------------
# Specialized Multi-Agent System
# ---------------------------------------------------------
class SQLAgent:
    def __init__(self, model, schema_context: str, semantic_context: str = ""):
        self.model = model
        self.schema = schema_context
        self.semantic_context = semantic_context

    def generate_and_verify(self, user_prompt: str, engine, history: list, max_retries: int = 3):
        if not self.model:
            return None, "", "Gemini API client is not initialized.", {}

        error_context = ""
        history_context = ""
        if history:
            for turn in history[-3:]:
                history_context += f"User: {turn.get('question', '')}\nGenerated SQL: {turn.get('sql', '')}\n"

        for attempt in range(max_retries):
            prompt = f"""You are an enterprise PostgreSQL senior architect.
Database Schema:
{self.schema}

Semantic Business Rules (MUST BE OBEYED):
{self.semantic_context}

Conversational History:
{history_context}

User Task: "{user_prompt}"
{error_context}

Rules:
1. Write a read-only SELECT or UNION query that strictly solves the user task and obeys the Semantic Business Rules.
2. Target PostgreSQL syntax. Wrap case-sensitive identifiers in double quotes.
3. Return ONLY a valid JSON object with the exact keys:
{{
  "sql": "SELECT ...",
  "confidence": "High" | "Medium" | "Low",
  "assumptions": ["List of business rules, filters, or type casts applied"]
}}
Do not wrap the JSON in markdown fences or add commentary.
"""
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config={"temperature": 0.0},
                    request_options={"timeout": 20},
                )
                raw_text = response.text.strip().replace("```json", "").replace("```", "").strip()

                # Parse structured response
                try:
                    payload = json.loads(raw_text)
                    raw_sql = payload.get("sql", "").strip()
                    metadata = {
                        "confidence": payload.get("confidence", "Medium"),
                        "assumptions": payload.get("assumptions", []),
                    }
                except json.JSONDecodeError:
                    raw_sql = raw_text.replace("```sql", "").strip()
                    metadata = {"confidence": "Medium", "assumptions": ["Fallback parser used."]}

                if "SELECT" in raw_sql.upper():
                    raw_sql = raw_sql[raw_sql.upper().find("SELECT"):]

                # 1. AST Guardrail Verification
                sanitized_sql, guardrail_err = SQLGuardrail.validate_and_sanitize(raw_sql, target_dialect="postgres")
                if guardrail_err:
                    error_context = f"\nPrevious attempt blocked by AST Guardrail: {guardrail_err}. Fix the SQL."
                    continue

                # 2. Database Execution Test
                df, db_err = execute_query(engine, sanitized_sql)
                if not db_err:
                    return df, sanitized_sql, None, metadata
                else:
                    error_context = f"\nPrevious attempt failed with Database Error: {db_err}. Fix the SQL."

            except Exception as e:
                error_context = f"\nAPI or Pipeline Exception: {str(e)}"

        return None, "", f"Failed after {max_retries} attempts. Last issue: {error_context}", {}


class VisualizationAgent:
    def __init__(self, model):
        self.model = model

    def generate_figure(self, user_prompt: str, df: pd.DataFrame):
        if not self.model or df.empty or len(df.columns) < 2:
            return None

        df_head_str = df.head(3).to_dict(orient="records")
        prompt = f"""You are a dynamic Plotly visualization specialist.
User Question: "{user_prompt}"
Dataframe Columns: {list(df.columns)}
Data Preview (df): {df_head_str}

Task: Write a Python code block using Plotly Express (px) or Plotly Graph Objects (go) that creates a 'fig' object.
Rules:
- The DataFrame is already provided in the variable `df`.
- Assign the result to a variable named `fig`.
- Use a clean theme with rose/red highlights (e.g. template='plotly_white', color_discrete_sequence=['#e11d48', '#fb7185', '#fda4af', '#f43f5e']).
- Output strictly Python code. No markdown fences, no comments, no explanations.
"""
        try:
            res = self.model.generate_content(
                prompt,
                generation_config={"temperature": 0.1},
                request_options={"timeout": 20},
            )
            code = res.text.strip().replace("```python", "").replace("```", "").strip()

            local_vars = {"df": df, "px": px, "go": go, "pd": pd, "np": np}
            exec(code, {}, local_vars)
            fig = local_vars.get("fig")
            if isinstance(fig, go.Figure):
                fig.update_layout(
                    margin=dict(l=20, r=20, t=40, b=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                return fig
        except Exception:
            try:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
                if categorical_cols and numeric_cols:
                    fig = px.bar(
                        df.head(15),
                        x=categorical_cols[0],
                        y=numeric_cols[0],
                        template="plotly_white",
                        color_discrete_sequence=["#e11d48"],
                    )
                    fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))
                    return fig
            except Exception:
                return None
        return None


class InsightAgent:
    def __init__(self, model, schema_context: str):
        self.model = model
        self.schema = schema_context

    def generate_insight_stream(self, user_prompt: str, original_sql: str, df: pd.DataFrame, engine):
        if not self.model or df.empty:
            yield "No tabular data returned for analytical evaluation."
            return

        # 1. Run Statistical Diagnostics
        stats_data = StatisticalDiagnosticsEngine.run_diagnostics(df)
        stats_str = f"\nEmpirical Statistical Diagnostics:\n{stats_data}" if stats_data else ""

        # 2. Autonomous Secondary Deep-Dive Query
        secondary_summary = ""
        try:
            deep_dive_prompt = f"""Given this user question: "{user_prompt}"
Database Schema: {self.schema}
Primary SQL executed: {original_sql}

Suggest 1 secondary complementary SQL SELECT query that provides broader baseline metrics (e.g., overall average, totals, or standard deviation) to enrich the primary answer.
Output strictly the SQL query or 'NONE'."""
            res = self.model.generate_content(
                deep_dive_prompt,
                generation_config={"temperature": 0.0},
                request_options={"timeout": 15},
            )
            sec_sql = res.text.strip().replace("```sql", "").replace("```", "").strip()

            if "SELECT" in sec_sql.upper():
                sanitized_sec, err = SQLGuardrail.validate_and_sanitize(sec_sql, target_dialect="postgres")
                if not err:
                    sec_df, sec_err = execute_query(engine, sanitized_sec)
                    if not sec_err and not sec_df.empty:
                        secondary_summary = f"\nSecondary Context Data:\n{sec_df.head(3).to_string(index=False)}"
        except Exception:
            secondary_summary = ""

        # 3. Stream Synthesis
        synthesis_prompt = f"""You are a Lead Quantitative Data Scientist.
User Query: "{user_prompt}"
Executed SQL: {original_sql}
Primary Data Sample:
{df.head(5).to_string(index=False)}
{secondary_summary}
{stats_str}

Provide a concise, publication-grade analytical interpretation (2-3 paragraphs max). 
Directly cite identified statistical metrics, anomalies, or correlations where applicable. Do not display code or raw SQL."""

        try:
            response = self.model.generate_content(
                synthesis_prompt,
                stream=True,
                generation_config={"temperature": 0.2},
                request_options={"timeout": 30},
            )
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            yield f"Insight synthesis error: {str(e)}"


class RootCauseAgent:
    def __init__(self, model, schema_context: str, semantic_context: str = ""):
        self.model = model
        self.schema = schema_context
        self.semantic_context = semantic_context

    def investigate(self, user_prompt: str, original_sql: str, df: pd.DataFrame, engine):
        """
        Generates 2-3 diagnostic slice queries, validates them via SQLGuardrail,
        executes them, and synthesizes a root-cause attribution briefing.
        """
        if not self.model:
            return [], "Gemini API client is not initialized."

        plan_prompt = f"""You are a Senior Root-Cause Data Analyst.
Database Schema:
{self.schema}

Semantic Business Rules:
{self.semantic_context}

Original User Question: "{user_prompt}"
Original SQL: {original_sql}
Primary Result Preview:
{df.head(5).to_string(index=False)}

Task: Propose 2 to 3 diagnostic PostgreSQL SELECT queries to explain *why* the primary result looks the way it does (e.g., slicing by Country, Product Description, CustomerID, or comparing cancellations/outliers).
Return ONLY a valid JSON array of objects with exact keys:
[
  {{"dimension": "Short label of what is being sliced", "sql": "SELECT ..."}}
]
Limit each query to at most 10 rows. Do not wrap in markdown fences.
"""
        diagnostic_results = []
        try:
            plan_res = self.model.generate_content(
                plan_prompt,
                generation_config={"temperature": 0.0},
                request_options={"timeout": 20},
            )
            raw_plan = plan_res.text.strip().replace("```json", "").replace("```", "").strip()
            queries = json.loads(raw_plan)

            for item in queries[:3]:
                dim = item.get("dimension", "Diagnostic Slice")
                raw_sql = item.get("sql", "").strip()
                sanitized_sql, err = SQLGuardrail.validate_and_sanitize(raw_sql, target_dialect="postgres")
                if not err:
                    slice_df, db_err = execute_query(engine, sanitized_sql)
                    if not db_err and slice_df is not None and not slice_df.empty:
                        diagnostic_results.append({
                            "dimension": dim,
                            "sql": sanitized_sql,
                            "df": slice_df,
                        })
        except Exception as e:
            return [], f"Root-cause planning failed: {str(e)}"

        if not diagnostic_results:
            return [], "No additional diagnostic slices could be executed for this query."

        # Synthesize findings across all diagnostic slices
        evidence_text = ""
        for res in diagnostic_results:
            evidence_text += f"\n--- Slice: {res['dimension']} ---\n{res['df'].head(5).to_string(index=False)}\n"

        synthesis_prompt = f"""You are a Lead Quantitative Analyst delivering a Root-Cause Attribution Brief.
Original Question: "{user_prompt}"
Diagnostic Evidence Across Dimensions:
{evidence_text}

Summarize the top 2-3 concrete drivers explaining the original result. Quantify shares, concentrations, or anomalies directly from the evidence. Keep it sharp and executive-ready (bullet points).
"""
        try:
            brief_res = self.model.generate_content(
                synthesis_prompt,
                generation_config={"temperature": 0.2},
                request_options={"timeout": 25},
            )
            return diagnostic_results, brief_res.text.strip()
        except Exception as e:
            return diagnostic_results, f"Synthesis error: {str(e)}"

# ---------------------------------------------------------
# UI Helper for Confidence & Assumptions
# ---------------------------------------------------------
def render_query_metadata(meta: dict):
    if not meta:
        return
    conf = meta.get("confidence", "Medium")
    badge_color = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}.get(conf, "🟡")
    st.caption(f"{badge_color} **Query Confidence: {conf}**")

    assumptions = meta.get("assumptions", [])
    if assumptions:
        with st.expander("Semantic Rules & Assumptions Applied", expanded=False):
            for item in assumptions:
                st.markdown(f"- {item}")

# ---------------------------------------------------------
# Session Initialization
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "db_connected" not in st.session_state:
    st.session_state.db_connected = False
if "db_uri" not in st.session_state:
    st.session_state.db_uri = os.getenv("SUPABASE_DATABASE_URL", "")

gemini_model, model_display_name = initialize_gemini()

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚡ AI Data Studio")
    st.caption(f"Engine: `{model_display_name}`")
    st.markdown("---")

    st.markdown("#### Connect Database")
    uri_input = st.text_input(
        "PostgreSQL URI",
        value=st.session_state.db_uri,
        type="password",
        placeholder="postgresql://user:pass@host:5432/postgres",
        label_visibility="collapsed",
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Connect 🚀", use_container_width=True):
            if uri_input:
                st.session_state.db_uri = uri_input.strip()
                try:
                    engine = get_db_engine(st.session_state.db_uri)
                    with engine.connect() as conn:
                        conn.execute(text("SELECT 1"))
                    st.session_state.db_connected = True
                    st.success("Connected successfully!")
                    st.rerun()
                except Exception as e:
                    st.session_state.db_connected = False
                    st.error(f"Connection Error: {str(e)}")
            else:
                st.warning("Please supply a valid URI.")

    with col2:
        if st.button("Disconnect", use_container_width=True):
            st.session_state.db_connected = False
            st.session_state.messages = []
            st.rerun()

    if st.session_state.db_connected:
        st.markdown("---")
        if st.button("Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.markdown("---")
    st.caption("Crafted by Shehbaz Makhmoor Ali\n[GitHub](https://github.com/Shehbaz-Makhmoor-Ali) • [LinkedIn](https://linkedin.com/in/shehbaz-makhmoor-ali-5b8643235)")

# ---------------------------------------------------------
# Main Chat Canvas
# ---------------------------------------------------------
if not st.session_state.db_connected:
    st.markdown(
        """
        <div class="hero-container">
            <div class="hero-title">Ask your <span class="hero-accent">database.</span> ⚡</div>
            <div class="hero-subtitle">Connect your database securely in the sidebar to initialize the AI Data Studio copilot.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    engine = get_db_engine(st.session_state.db_uri)
    schema_context = fetch_schema_context(engine)
    semantic_layer = SemanticLayer("semantic_layer.json")

    # Initialize agents
    sql_agent = SQLAgent(gemini_model, schema_context, semantic_layer.get_prompt_context())
    viz_agent = VisualizationAgent(gemini_model)
    insight_agent = InsightAgent(gemini_model, schema_context)

    # Render previous conversation history
    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(msg["content"])
            else:
                if "meta" in msg:
                    render_query_metadata(msg["meta"])
                if "sql" in msg:
                    with st.expander("Executed SQL Query", expanded=False):
                        st.code(msg["sql"], language="sql")
                if "df" in msg and msg["df"] is not None:
                    st.dataframe(msg["df"], use_container_width=True)
                if "figure" in msg and msg["figure"] is not None:
                    st.plotly_chart(msg["figure"], use_container_width=True)
                st.markdown(msg["content"])

                # Root-Cause Investigation Trigger / Display
                if msg.get("df") is not None and not msg["df"].empty:
                    if "root_cause" in msg:
                        with st.expander("🔍 Root-Cause Attribution Brief", expanded=True):
                            st.markdown(msg["root_cause"]["brief"])
                            for diag in msg["root_cause"]["slices"]:
                                st.caption(f"**Slice:** {diag['dimension']}")
                                st.dataframe(diag["df"], use_container_width=True)
                    else:
                        if st.button("🔍 Investigate Root Causes", key=f"rca_{idx}"):
                            with st.spinner("Running multi-dimensional root-cause analysis..."):
                                rca_agent = RootCauseAgent(
                                    gemini_model, schema_context, semantic_layer.get_prompt_context()
                                )
                                slices, brief = rca_agent.investigate(
                                    msg.get("question", ""), msg.get("sql", ""), msg["df"], engine
                                )
                                msg["root_cause"] = {"slices": slices, "brief": brief}
                                st.rerun()

    # Chat prompt handler
    user_prompt = st.chat_input("Ask a question about your data...")
    if user_prompt:
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            # Pre-flight intent check before calling LLM or DB
            is_safe, block_msg = SQLGuardrail.check_prompt_intent(user_prompt)
            if not is_safe:
                st.warning(block_msg)
                st.session_state.messages.append({"role": "assistant", "content": block_msg})
                st.stop()

            with st.spinner("Synthesizing query with AST validation & semantic rules..."):
                df, final_sql, sql_err, sql_meta = sql_agent.generate_and_verify(
                    user_prompt=user_prompt,
                    engine=engine,
                    history=st.session_state.messages,
                )

            if sql_err:
                st.error(f"Execution Error: {sql_err}")
                st.session_state.messages.append({"role": "assistant", "content": f"Failed to execute query: {sql_err}"})
            else:
                render_query_metadata(sql_meta)

                with st.expander("Executed SQL Query", expanded=False):
                    st.code(final_sql, language="sql")

                if df is not None and not df.empty:
                    st.dataframe(df, use_container_width=True)

                    # Dynamic Visualizer
                    fig = viz_agent.generate_figure(user_prompt, df)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

                    # Streamed Quantitative Insights
                    st.markdown("#### Analytical Briefing")
                    insight_placeholder = st.empty()
                    accumulated_insight = ""

                    for chunk in insight_agent.generate_insight_stream(user_prompt, final_sql, df, engine):
                        accumulated_insight += chunk
                        insight_placeholder.markdown(accumulated_insight + "▌")
                    insight_placeholder.markdown(accumulated_insight)

                    # Commit assistant turn to session state and rerun to render RCA button cleanly
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": accumulated_insight,
                            "sql": final_sql,
                            "meta": sql_meta,
                            "df": df,
                            "figure": fig,
                            "question": user_prompt,
                        }
                    )
                    st.rerun()
                else:
                    st.info("Query returned an empty result set.")
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": "The query executed successfully but returned 0 rows.",
                            "sql": final_sql,
                            "meta": sql_meta,
                            "df": df,
                            "figure": None,
                            "question": user_prompt,
                        }
                    )