import os
import io
import re
import json
import time
import uuid
from datetime import datetime
import duckdb
import pandas as pd
import numpy as np
import scipy.stats as stats
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text
import google.generativeai as genai
import sqlglot
from sqlglot import exp

# Load environment variables, overriding any cached values
load_dotenv(override=True)

# ---------------------------------------------------------
# Page Configuration & Professional UI Styling
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
    /* Clean executive canvas */
    .stApp {
        background: linear-gradient(160deg, #fff5f6 0%, #ffffff 42%, #f8fafc 100%);
        color: #0f172a;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Professional Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.1rem;
        padding-bottom: 1.25rem;
    }
    .sidebar-brand {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.75rem 0.9rem;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        margin-bottom: 0.85rem;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }
    .sidebar-brand-title {
        font-size: 1.05rem;
        font-weight: 800;
        color: #0f172a;
        letter-spacing: -0.02em;
        margin: 0;
    }
    .sidebar-brand-sub {
        font-size: 0.72rem;
        color: #64748b;
        margin: 2px 0 0 0;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    }
    .status-pill-on {
        font-size: 0.7rem;
        font-weight: 700;
        color: #047857;
        background: #d1fae5;
        padding: 3px 8px;
        border-radius: 999px;
    }
    .status-pill-off {
        font-size: 0.7rem;
        font-weight: 700;
        color: #64748b;
        background: #e2e8f0;
        padding: 3px 8px;
        border-radius: 999px;
    }
    .sidebar-section-label {
        font-size: 0.68rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        color: #64748b;
        margin: 0.95rem 0 0.4rem 0;
    }

    /* Typography */
    h1, h2, h3, h4 {
        color: #0f172a;
        font-weight: 700;
        letter-spacing: -0.015em;
    }

    /* Chat Header Bar */
    .chat-header-bar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.65rem 1rem;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
    }
    .chat-header-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #0f172a;
    }
    .chat-header-badge {
        font-size: 0.75rem;
        font-weight: 600;
        color: #475569;
        background: #f1f5f9;
        padding: 3px 10px;
        border-radius: 999px;
    }

    /* Polished Chat Messages */
    div[data-testid="stChatMessage"] {
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.9rem;
        border: 1px solid #e2e8f0;
        background-color: #ffffff;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.03);
    }
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) {
        background-color: #fff1f2;
        border: 1px solid #fecdd3;
    }

    /* Metadata Pill Row */
    .meta-row {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-bottom: 0.6rem;
    }
    .meta-pill {
        font-size: 0.74rem;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 999px;
        border: 1px solid #e2e8f0;
        background: #f8fafc;
        color: #334155;
    }
    .meta-pill-pii {
        background: #fef2f2;
        border-color: #fecaca;
        color: #b91c1c;
    }

    /* Briefing Card */
    .briefing-header {
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #e11d48;
        margin-top: 0.75rem;
        margin-bottom: 0.35rem;
    }

    /* Chat Input */
    .stChatInputContainer {
        border-radius: 10px !important;
        border: 1.5px solid #cbd5e1 !important;
        background-color: #ffffff !important;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.06) !important;
        transition: border-color 0.2s ease;
    }
    .stChatInputContainer:focus-within {
        border-color: #e11d48 !important;
    }

    /* Buttons */
    .stButton > button, .stDownloadButton > button {
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.84rem;
        border: 1px solid #e2e8f0;
        transition: all 0.15s ease;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        border-color: #e11d48;
        color: #e11d48;
    }

    /* Empty state hero */
    .hero-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        margin-top: 10vh;
        margin-bottom: 2rem;
        text-align: center;
    }
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        color: #0f172a;
        letter-spacing: -0.03em;
    }
    .hero-accent {
        color: #e11d48;
    }
    .hero-subtitle {
        font-size: 1.02rem;
        color: #64748b;
        margin-top: 8px;
        max-width: 540px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Role-Based Access Control (RBAC) Policies
# ---------------------------------------------------------
RBAC_POLICIES = {
    "Admin (All Data)": None,
    "UK Manager (United Kingdom Only)": {
        "role": "UK Manager",
        "table": "SalesData",
        "column": "Country",
        "value": "United Kingdom",
    },
    "France Manager (France Only)": {
        "role": "France Manager",
        "table": "SalesData",
        "column": "Country",
        "value": "France",
    },
}

# ---------------------------------------------------------
# Persistent Multi-Session Chat History Store
# ---------------------------------------------------------
class ChatHistoryStore:
    """
    Persists multi-session chat conversations (including DataFrames, Plotly charts,
    SQL, metadata, and Root-Cause slices) locally to JSON so history survives reloads.
    """
    def __init__(self, filepath: str = "chat_history.json"):
        self.filepath = filepath
        self.sessions = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data if isinstance(data, dict) else {}
            except Exception:
                return {}
        return {}

    def _persist(self):
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(self.sessions, f, indent=2)

    @staticmethod
    def _serialize_message(msg: dict) -> dict:
        out = {
            "role": msg.get("role", "assistant"),
            "content": msg.get("content", ""),
            "sql": msg.get("sql"),
            "meta": msg.get("meta"),
            "masked_cols": msg.get("masked_cols", []),
            "question": msg.get("question"),
            "verified": msg.get("verified", False),
        }
        if msg.get("df") is not None and isinstance(msg["df"], pd.DataFrame) and not msg["df"].empty:
            out["df_json"] = msg["df"].head(100).to_json(orient="split", date_format="iso")
        if msg.get("figure") is not None:
            try:
                out["fig_json"] = msg["figure"].to_json()
            except Exception:
                out["fig_json"] = None
        if msg.get("root_cause"):
            rc = msg["root_cause"]
            out["root_cause"] = {
                "brief": rc.get("brief", ""),
                "slices": [
                    {
                        "dimension": s.get("dimension", ""),
                        "sql": s.get("sql", ""),
                        "df_json": s["df"].head(20).to_json(orient="split", date_format="iso")
                        if isinstance(s.get("df"), pd.DataFrame)
                        else None,
                    }
                    for s in rc.get("slices", [])
                ],
            }
        return out

    @staticmethod
    def _deserialize_message(raw: dict) -> dict:
        msg = {
            "role": raw.get("role", "assistant"),
            "content": raw.get("content", ""),
            "masked_cols": raw.get("masked_cols", []),
            "verified": raw.get("verified", False),
        }
        if raw.get("sql") is not None:
            msg["sql"] = raw["sql"]
        if raw.get("meta") is not None:
            msg["meta"] = raw["meta"]
        if raw.get("question") is not None:
            msg["question"] = raw["question"]
        if raw.get("df_json"):
            try:
                msg["df"] = pd.read_json(io.StringIO(raw["df_json"]), orient="split")
            except Exception:
                msg["df"] = None
        if raw.get("fig_json"):
            try:
                msg["figure"] = pio.from_json(raw["fig_json"])
            except Exception:
                msg["figure"] = None
        if raw.get("root_cause"):
            rc = raw["root_cause"]
            slices = []
            for s in rc.get("slices", []):
                sdf = None
                if s.get("df_json"):
                    try:
                        sdf = pd.read_json(io.StringIO(s["df_json"]), orient="split")
                    except Exception:
                        sdf = None
                slices.append({"dimension": s.get("dimension", ""), "sql": s.get("sql", ""), "df": sdf})
            msg["root_cause"] = {"brief": rc.get("brief", ""), "slices": slices}
        return msg

    def create_session(self, title: str = "New Chat") -> str:
        sid = str(uuid.uuid4())[:8]
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.sessions[sid] = {
            "title": title,
            "created_at": now,
            "updated_at": now,
            "messages": [],
        }
        self._persist()
        return sid

    def get_messages(self, sid: str) -> list:
        session = self.sessions.get(sid)
        if not session:
            return []
        return [self._deserialize_message(m) for m in session.get("messages", [])]

    def save_messages(self, sid: str, messages: list):
        if sid not in self.sessions:
            self.create_session()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Auto-title from the first user prompt
        title = self.sessions[sid].get("title", "New Chat")
        if title == "New Chat":
            for m in messages:
                if m.get("role") == "user" and m.get("content"):
                    raw_t = m["content"].strip()
                    title = (raw_t[:36] + "…") if len(raw_t) > 36 else raw_t
                    break
        self.sessions[sid]["title"] = title
        self.sessions[sid]["updated_at"] = now
        self.sessions[sid]["messages"] = [self._serialize_message(m) for m in messages]
        self._persist()

    def delete_session(self, sid: str):
        if sid in self.sessions:
            del self.sessions[sid]
            self._persist()

    def list_sessions(self) -> list:
        items = [{"id": sid, **data} for sid, data in self.sessions.items()]
        items.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        return items

# ---------------------------------------------------------
# Security AST Guardrail, RBAC Rewriter & PII Masking
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
        destructive_pattern = r"\b(delete|drop|truncate|alter|update|insert\s+into)\b"
        if re.search(destructive_pattern, user_prompt, re.IGNORECASE):
            return False, (
                "🔒 **Action Blocked (Read-Only Mode):** I cannot execute `DELETE`, `DROP`, `UPDATE`, "
                "`INSERT`, or `ALTER` operations. AI Data Studio is strictly read-only to protect your database."
            )
        return True, None

    @classmethod
    def _apply_rls_to_ast(cls, parsed: exp.Expression, rls_policy: dict, target_dialect: str):
        if not rls_policy:
            return parsed

        target_table = rls_policy["table"].lower()
        rls_col = rls_policy["column"]
        rls_val = rls_policy["value"].replace("'", "''")

        for select_node in parsed.find_all(exp.Select):
            tables_in_select = []
            from_clause = select_node.args.get("from_") or select_node.args.get("from")
            if from_clause and isinstance(from_clause.this, exp.Table):
                tables_in_select.append(from_clause.this)
            for join_clause in select_node.args.get("joins") or []:
                if isinstance(join_clause.this, exp.Table):
                    tables_in_select.append(join_clause.this)

            for tbl in tables_in_select:
                if tbl.name.lower() == target_table:
                    qualifier = f'"{tbl.alias}"' if tbl.alias else f'"{tbl.name}"'
                    condition_sql = f'{qualifier}."{rls_col}" = \'{rls_val}\''
                    select_node.where(condition_sql, dialect=target_dialect, copy=False)

        return parsed

    @classmethod
    def validate_and_sanitize(cls, raw_sql: str, target_dialect: str = "postgres", rls_policy: dict = None):
        try:
            statements = sqlglot.parse(raw_sql, read=target_dialect)
            if not statements or len(statements) != 1 or statements[0] is None:
                return None, "Security Violation: Multi-statement or empty SQL blocked by AST Guardrail."

            parsed = statements[0]

            for node in parsed.walk():
                if isinstance(node, cls.FORBIDDEN_OPERATIONS):
                    return None, f"Security Violation: Destructive statement '{node.key.upper()}' blocked by AST Guardrail."

            if not isinstance(parsed, (exp.Select, exp.Union)):
                return None, "Security Violation: Only SELECT/UNION statements are permitted."

            if rls_policy:
                parsed = cls._apply_rls_to_ast(parsed, rls_policy, target_dialect)

            sanitized_sql = parsed.sql(dialect=target_dialect)
            return sanitized_sql, None
        except sqlglot.errors.ParseError as e:
            return None, f"AST Syntax Parse Error: {str(e)}"
        except Exception as e:
            return None, f"Guardrail Error: {str(e)}"


class PIIMasker:
    SENSITIVE_COL_PATTERN = re.compile(
        r"(email|e_mail|phone|mobile|ssn|social_security|password|passwd|secret|token|credit_card|card_num|address)",
        re.IGNORECASE,
    )
    EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
    PHONE_REGEX = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b")

    @classmethod
    def mask_dataframe(cls, df: pd.DataFrame):
        if df is None or df.empty:
            return df, []

        masked_df = df.copy()
        masked_cols = []

        for col in masked_df.columns:
            if cls.SENSITIVE_COL_PATTERN.search(str(col)):
                masked_df[col] = "[REDACTED]"
                masked_cols.append(col)
                continue

            if pd.api.types.is_string_dtype(masked_df[col]) or masked_df[col].dtype == "object":
                series_str = masked_df[col].astype(str)
                has_email = series_str.str.contains(cls.EMAIL_REGEX, na=False).any()
                has_phone = series_str.str.contains(cls.PHONE_REGEX, na=False).any()
                if has_email or has_phone:
                    series_str = series_str.str.replace(cls.EMAIL_REGEX, "[EMAIL_REDACTED]", regex=True)
                    series_str = series_str.str.replace(cls.PHONE_REGEX, "[PHONE_REDACTED]", regex=True)
                    masked_df[col] = series_str
                    masked_cols.append(col)

        return masked_df, masked_cols

# ---------------------------------------------------------
# Semantic Business Rules & Golden Query Memory
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


class GoldenQueryStore:
    STOP_WORDS = {"what", "is", "our", "the", "a", "an", "for", "of", "by", "in", "to", "show", "me", "all", "from"}

    def __init__(self, filepath: str = "verified_queries.json"):
        self.filepath = filepath
        self.entries = self._load()

    def _load(self) -> list:
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data if isinstance(data, list) else []
            except Exception:
                return []
        return []

    def _persist(self):
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(self.entries, f, indent=2)

    def save(self, question: str, sql: str) -> bool:
        q_clean = question.strip()
        for item in self.entries:
            if item.get("question", "").lower() == q_clean.lower():
                item["sql"] = sql
                item["saved_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
                self._persist()
                return True
        self.entries.append({
            "question": q_clean,
            "sql": sql,
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        })
        self._persist()
        return True

    def clear(self):
        self.entries = []
        self._persist()

    def _tokenize(self, text_str: str) -> set:
        tokens = set(re.findall(r"[a-zA-Z0-9_]+", text_str.lower()))
        return tokens - self.STOP_WORDS

    def retrieve_context(self, user_prompt: str, top_k: int = 3) -> str:
        if not self.entries:
            return "No verified golden queries saved yet."

        q_tokens = self._tokenize(user_prompt)
        scored = []
        for entry in self.entries:
            e_tokens = self._tokenize(entry.get("question", ""))
            overlap = len(q_tokens & e_tokens)
            union = len(q_tokens | e_tokens) or 1
            score = overlap / union
            if score > 0:
                scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        selected = [item[1] for item in scored[:top_k]]
        if not selected:
            return "No closely matching verified queries found."

        blocks = []
        for idx, ex in enumerate(selected, 1):
            blocks.append(f"Verified Example {idx}:\nQuestion: {ex['question']}\nSQL: {ex['sql']}")
        return "\n\n".join(blocks)

# ---------------------------------------------------------
# Statistical Diagnostics Engine
# ---------------------------------------------------------
class StatisticalDiagnosticsEngine:
    @staticmethod
    def run_diagnostics(df: pd.DataFrame) -> dict:
        diagnostics = {}
        if df.empty or len(df) < 3:
            return diagnostics

        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return diagnostics

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
# Database Engine, Schema Inspector & DuckDB Federation
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_db_engine(uri: str):
    return create_engine(
        uri,
        pool_pre_ping=True,
        connect_args={
            "connect_timeout": 10,
            "options": "-c statement_timeout=15000",
        },
    )

def fetch_schema_context(engine, prefix: str = "") -> str:
    inspector = inspect(engine)
    schema_details = []
    for table_name in inspector.get_table_names():
        columns = inspector.get_columns(table_name)
        col_strs = [f"{col['name']} ({str(col['type'])})" for col in columns]
        qualified_name = f'{prefix}"{table_name}"' if prefix else f'"{table_name}"'
        schema_details.append(f"PostgreSQL Table {qualified_name}: {', '.join(col_strs)}")
    return "\n".join(schema_details)


class FederatedEngine:
    def __init__(self, pg_uri: str):
        self.con = duckdb.connect(database=":memory:")
        self.pg_attached = False
        self.uploaded_tables = {}

        if pg_uri:
            try:
                self.con.execute("INSTALL postgres; LOAD postgres;")
                self.con.execute(
                    f"ATTACH '{pg_uri}' AS pg (TYPE postgres, READ_ONLY, SCHEMA 'public');"
                )
                self.pg_attached = True
            except Exception as e:
                st.sidebar.warning(f"DuckDB Postgres attach warning: {e}")

    def register_dataframe(self, filename: str, df: pd.DataFrame) -> str:
        clean_name = re.sub(r"[^a-zA-Z0-9_]", "_", filename.rsplit(".", 1)[0]).lower()
        self.con.register(clean_name, df)
        self.uploaded_tables[clean_name] = df
        return clean_name

    def get_uploaded_schema(self) -> str:
        if not self.uploaded_tables:
            return ""
        lines = []
        for name, df in self.uploaded_tables.items():
            cols = [f"{col} ({dtype})" for col, dtype in df.dtypes.items()]
            lines.append(f"Local Uploaded Table '{name}': {', '.join(cols)}")
        return "\n".join(lines)

    def execute(self, sql: str):
        try:
            df = self.con.execute(sql).df()
            return df, None
        except Exception as e:
            return None, str(e)


def execute_query(engine, sql: str, fed_engine: FederatedEngine = None):
    if fed_engine is not None and fed_engine.uploaded_tables:
        return fed_engine.execute(sql)
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
    def __init__(
        self,
        model,
        schema_context: str,
        semantic_context: str = "",
        golden_store: GoldenQueryStore = None,
        federated: bool = False,
        rls_policy: dict = None,
    ):
        self.model = model
        self.schema = schema_context
        self.semantic_context = semantic_context
        self.golden_store = golden_store
        self.federated = federated
        self.rls_policy = rls_policy

    def generate_and_verify(self, user_prompt: str, engine, history: list, fed_engine=None, max_retries: int = 3):
        if not self.model:
            return None, "", "Gemini API client is not initialized.", {}

        target_dialect = "duckdb" if self.federated else "postgres"
        dialect_note = (
            "Target DuckDB SQL syntax. Reference PostgreSQL tables with the `pg.` prefix (e.g., `pg.\"SalesData\"`) "
            "and local uploaded tables directly by their table name."
            if self.federated
            else "Target PostgreSQL syntax. Wrap case-sensitive identifiers in double quotes."
        )

        golden_context = (
            self.golden_store.retrieve_context(user_prompt)
            if self.golden_store
            else "No verified golden queries."
        )

        error_context = ""
        history_context = ""
        if history:
            for turn in history[-3:]:
                history_context += f"User: {turn.get('question', '')}\nGenerated SQL: {turn.get('sql', '')}\n"

        for attempt in range(max_retries):
            prompt = f"""You are an enterprise SQL senior architect.
Available Schema:
{self.schema}

Semantic Business Rules (MUST BE OBEYED):
{self.semantic_context}

Verified Golden Query Examples (Reference Patterns):
{golden_context}

Conversational History:
{history_context}

User Task: "{user_prompt}"
{error_context}

Rules:
1. Write a read-only SELECT or UNION query that strictly solves the user task and obeys the Semantic Business Rules.
2. {dialect_note}
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

                sanitized_sql, guardrail_err = SQLGuardrail.validate_and_sanitize(
                    raw_sql, target_dialect=target_dialect, rls_policy=self.rls_policy
                )
                if guardrail_err:
                    error_context = f"\nPrevious attempt blocked by AST Guardrail: {guardrail_err}. Fix the SQL."
                    continue

                if self.rls_policy:
                    metadata.setdefault("assumptions", []).append(
                        f"RBAC AST Policy Enforced: {self.rls_policy['role']} "
                        f"({self.rls_policy['column']} = '{self.rls_policy['value']}')"
                    )

                df, db_err = execute_query(engine, sanitized_sql, fed_engine=fed_engine)
                if not db_err:
                    return df, sanitized_sql, None, metadata
                else:
                    error_context = f"\nPrevious attempt failed with Execution Error: {db_err}. Fix the SQL."

            except Exception as e:
                error_context = f"\nAPI or Pipeline Exception: {str(e)}"

        return None, "", f"Failed after {max_retries} attempts. Last issue: {error_context}", {}


class VisualizationAgent:
    def __init__(self, model):
        self.model = model

    def generate_figure(self, user_prompt: str, df: pd.DataFrame):
        if not self.model or df.empty or len(df.columns) < 2 or df.select_dtypes(include=[np.number]).empty:
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
    def __init__(self, model, schema_context: str, federated: bool = False):
        self.model = model
        self.schema = schema_context
        self.federated = federated

    def generate_insight_stream(self, user_prompt: str, original_sql: str, df: pd.DataFrame):
        if not self.model or df.empty:
            yield "No tabular data returned for analytical evaluation."
            return

        stats_data = StatisticalDiagnosticsEngine.run_diagnostics(df)
        stats_str = f"\nEmpirical Statistical Diagnostics:\n{stats_data}" if stats_data else ""

        synthesis_prompt = f"""You are a Lead Quantitative Data Scientist.
User Query: "{user_prompt}"
Executed SQL: {original_sql}
Primary Data Sample:
{df.head(5).to_string(index=False)}
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
    def __init__(
        self,
        model,
        schema_context: str,
        semantic_context: str = "",
        federated: bool = False,
        rls_policy: dict = None,
    ):
        self.model = model
        self.schema = schema_context
        self.semantic_context = semantic_context
        self.federated = federated
        self.rls_policy = rls_policy

    def investigate(self, user_prompt: str, original_sql: str, df: pd.DataFrame, engine, fed_engine=None):
        if not self.model:
            return [], "Gemini API client is not initialized."

        target_dialect = "duckdb" if self.federated else "postgres"
        plan_prompt = f"""You are a Senior Root-Cause Data Analyst.
Available Schema:
{self.schema}

Semantic Business Rules:
{self.semantic_context}

Original User Question: "{user_prompt}"
Original SQL: {original_sql}
Primary Result Preview:
{df.head(5).to_string(index=False)}

Task: Propose 2 to 3 diagnostic SELECT queries to explain *why* the primary result looks the way it does (e.g., slicing by Country, Product Description, CustomerID, or comparing cancellations/outliers).
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
                sanitized_sql, err = SQLGuardrail.validate_and_sanitize(
                    raw_sql, target_dialect=target_dialect, rls_policy=self.rls_policy
                )
                if not err:
                    slice_df, db_err = execute_query(engine, sanitized_sql, fed_engine=fed_engine)
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
# Automated Evaluation & Security Benchmark Engine
# ---------------------------------------------------------
class BenchmarkEngine:
    @staticmethod
    def run_security_suite() -> pd.DataFrame:
        records = []

        destructive_prompts = [
            "Delete all rows from SalesData where Country is France",
            "Drop table SalesData immediately",
            "Update SalesData set UnitPrice = 0",
        ]
        for p in destructive_prompts:
            is_safe, _ = SQLGuardrail.check_prompt_intent(p)
            records.append({
                "Category": "Pre-Flight Security",
                "Test Case": p,
                "Status": "PASS" if not is_safe else "FAIL",
                "Detail": "Blocked before LLM & DB" if not is_safe else "Bypassed filter",
            })

        malicious_sqls = [
            ('Multi-Statement Injection', 'SELECT * FROM "SalesData"; DROP TABLE "SalesData"'),
            ('Direct DELETE AST', 'DELETE FROM "SalesData" WHERE "Country" = \'France\''),
            ('DDL ALTER AST', 'ALTER TABLE "SalesData" DROP COLUMN "UnitPrice"'),
        ]
        for label, bad_sql in malicious_sqls:
            sanitized, err = SQLGuardrail.validate_and_sanitize(bad_sql, target_dialect="postgres")
            records.append({
                "Category": "AST Guardrail",
                "Test Case": label,
                "Status": "PASS" if (sanitized is None and err) else "FAIL",
                "Detail": err if err else "Allowed unsafe SQL",
            })

        uk_policy = RBAC_POLICIES["UK Manager (United Kingdom Only)"]
        rbac_sql, rbac_err = SQLGuardrail.validate_and_sanitize(
            'SELECT "Country", SUM("Quantity") FROM "SalesData" GROUP BY "Country"',
            target_dialect="postgres",
            rls_policy=uk_policy,
        )
        rbac_ok = (rbac_err is None) and ("United Kingdom" in (rbac_sql or ""))
        records.append({
            "Category": "RBAC AST Rewrite",
            "Test Case": "Inject UK filter on unrestricted SELECT",
            "Status": "PASS" if rbac_ok else "FAIL",
            "Detail": rbac_sql or rbac_err,
        })

        inj_sql, inj_err = SQLGuardrail.validate_and_sanitize(
            'SELECT * FROM "SalesData" WHERE "Country" = \'France\'',
            target_dialect="postgres",
            rls_policy=uk_policy,
        )
        inj_ok = (inj_err is None) and ("United Kingdom" in (inj_sql or "")) and ("AND" in (inj_sql or "").upper())
        records.append({
            "Category": "RBAC Injection Defense",
            "Test Case": "Force UK filter even when SQL requests France",
            "Status": "PASS" if inj_ok else "FAIL",
            "Detail": inj_sql or inj_err,
        })

        sample_pii = pd.DataFrame({
            "Country": ["UK", "France"],
            "ManagerEmail": ["alice@corp.com", "bob@corp.com"],
            "Phone": ["+44-20-7946-0958", "+33-1-4268-5300"],
            "Notes": ["Contact john@test.org asap", "Standard order"],
        })
        masked_df, masked_cols = PIIMasker.mask_dataframe(sample_pii)
        pii_ok = (
            (masked_df["ManagerEmail"] == "[REDACTED]").all()
            and (masked_df["Phone"] == "[REDACTED]").all()
            and ("[EMAIL_REDACTED]" in masked_df["Notes"].iloc[0])
        )
        records.append({
            "Category": "PII Redaction",
            "Test Case": "Column + Inline Email/Phone Masking",
            "Status": "PASS" if pii_ok else "FAIL",
            "Detail": f"Redacted columns: {', '.join(masked_cols)}",
        })

        return pd.DataFrame(records)

    @staticmethod
    def run_live_llm_suite(sql_agent: SQLAgent, engine) -> pd.DataFrame:
        tests = [
            {
                "name": "Semantic Net Revenue Compliance",
                "prompt": "What is our total Net Revenue across all valid orders?",
                "check": lambda sql, df: (
                    df is not None
                    and not df.empty
                    and ("C%" in sql or "c%" in sql)
                    and "Quantity" in sql
                    and "UnitPrice" in sql
                ),
            },
            {
                "name": "Top-3 Country Aggregation",
                "prompt": "What is our total Net Revenue by Country for the top 3 countries?",
                "check": lambda sql, df: df is not None and len(df) == 3,
            },
        ]

        records = []
        for t in tests:
            start = time.time()
            df, final_sql, err, _ = sql_agent.generate_and_verify(t["prompt"], engine, history=[])
            latency = round(time.time() - start, 2)
            passed = (err is None) and t["check"](final_sql, df)
            records.append({
                "Category": "Live LLM & Semantic",
                "Test Case": t["name"],
                "Status": "PASS" if passed else "FAIL",
                "Latency (s)": latency,
                "Detail": final_sql if not err else err,
            })
        return pd.DataFrame(records)

# ---------------------------------------------------------
# UI Helpers (Metadata Pills & Executive Brief Builder)
# ---------------------------------------------------------
def render_message_badges(meta: dict = None, masked_cols: list = None):
    pills = []
    if meta:
        conf = meta.get("confidence", "Medium")
        dot = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}.get(conf, "🟡")
        pills.append(f'<span class="meta-pill">{dot} Confidence: {conf}</span>')
    if masked_cols:
        pills.append(f'<span class="meta-pill meta-pill-pii">🔒 PII Redacted: {", ".join(masked_cols)}</span>')
    if pills:
        st.markdown(f'<div class="meta-row">{"".join(pills)}</div>', unsafe_allow_html=True)

    if meta and meta.get("assumptions"):
        with st.expander("Semantic Rules & Assumptions Applied", expanded=False):
            for item in meta["assumptions"]:
                st.markdown(f"- {item}")


def build_executive_brief(msg: dict) -> str:
    lines = [
        "# AI Data Studio — Executive Analytical Brief",
        f"**Question:** {msg.get('question', 'N/A')}",
        "",
    ]

    meta = msg.get("meta", {})
    if meta:
        lines.append(f"**Query Confidence:** {meta.get('confidence', 'Medium')}")
        assumptions = meta.get("assumptions", [])
        if assumptions:
            lines.append("**Semantic Rules & Assumptions Applied:**")
            for a in assumptions:
                lines.append(f"- {a}")
        lines.append("")

    if msg.get("sql"):
        lines.extend(["## Executed SQL", "```sql", msg["sql"], "```", ""])

    if msg.get("df") is not None and not msg["df"].empty:
        lines.extend([
            "## Data Snapshot (Top 10 Rows)",
            msg["df"].head(10).to_markdown(index=False),
            "",
        ])

    lines.extend(["## Analytical Briefing", msg.get("content", ""), ""])

    if "root_cause" in msg and msg["root_cause"].get("brief"):
        lines.extend(["## Root-Cause Attribution", msg["root_cause"]["brief"], ""])

    return "\n".join(lines)

# ---------------------------------------------------------
# Session & Persistent History Initialization
# ---------------------------------------------------------
history_store = ChatHistoryStore("chat_history.json")
golden_store = GoldenQueryStore("verified_queries.json")
gemini_model, model_display_name = initialize_gemini()

if "db_connected" not in st.session_state:
    st.session_state.db_connected = False
if "db_uri" not in st.session_state:
    st.session_state.db_uri = os.getenv("SUPABASE_DATABASE_URL", "")
if "sec_bench_df" not in st.session_state:
    st.session_state.sec_bench_df = None
if "live_bench_df" not in st.session_state:
    st.session_state.live_bench_df = None
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

# Load most recent persisted chat session on startup (or create one)
if "active_session_id" not in st.session_state:
    existing = history_store.list_sessions()
    if existing:
        st.session_state.active_session_id = existing[0]["id"]
        st.session_state.messages = history_store.get_messages(existing[0]["id"])
    else:
        sid = history_store.create_session("New Chat")
        st.session_state.active_session_id = sid
        st.session_state.messages = []

if "messages" not in st.session_state:
    st.session_state.messages = history_store.get_messages(st.session_state.active_session_id)

# ---------------------------------------------------------
# Professional Sidebar Layout
# ---------------------------------------------------------
uploaded_files = []
mask_pii = True
selected_role_label = "Admin (All Data)"
workspace_mode = "💬 Analytics Studio"

with st.sidebar:
    pill_html = (
        '<span class="status-pill-on">● LIVE</span>'
        if st.session_state.db_connected
        else '<span class="status-pill-off">○ OFFLINE</span>'
    )
    st.markdown(
        f"""
        <div class="sidebar-brand">
            <div>
                <p class="sidebar-brand-title">⚡ AI Data Studio</p>
                <p class="sidebar-brand-sub">{model_display_name}</p>
            </div>
            {pill_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("🔌 Database Connection", expanded=not st.session_state.db_connected):
        uri_input = st.text_input(
            "PostgreSQL URI",
            value=st.session_state.db_uri,
            type="password",
            placeholder="postgresql://user:pass@host:5432/postgres",
            label_visibility="collapsed",
        )
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Connect", use_container_width=True):
                if uri_input:
                    st.session_state.db_uri = uri_input.strip()
                    try:
                        engine = get_db_engine(st.session_state.db_uri)
                        with engine.connect() as conn:
                            conn.execute(text("SELECT 1"))
                        st.session_state.db_connected = True
                        st.rerun()
                    except Exception as e:
                        st.session_state.db_connected = False
                        st.error(f"Error: {str(e)}")
                else:
                    st.warning("Enter a valid URI.")
        with c2:
            if st.button("Disconnect", use_container_width=True, disabled=not st.session_state.db_connected):
                st.session_state.db_connected = False
                st.rerun()

    if st.session_state.db_connected:
        st.markdown('<div class="sidebar-section-label">Workspace</div>', unsafe_allow_html=True)
        workspace_mode = st.radio(
            "Workspace Mode",
            ["💬 Analytics Studio", "📊 Evaluation & Benchmark"],
            label_visibility="collapsed",
        )

        # Persistent Chat History Panel
        st.markdown('<div class="sidebar-section-label">Chat History</div>', unsafe_allow_html=True)
        h_col1, h_col2 = st.columns([3, 1])
        with h_col1:
            if st.button("➕ New Chat", use_container_width=True):
                new_sid = history_store.create_session("New Chat")
                st.session_state.active_session_id = new_sid
                st.session_state.messages = []
                st.rerun()
        with h_col2:
            if st.button("🗑️", help="Delete current chat", use_container_width=True):
                history_store.delete_session(st.session_state.active_session_id)
                remaining = history_store.list_sessions()
                if remaining:
                    st.session_state.active_session_id = remaining[0]["id"]
                    st.session_state.messages = history_store.get_messages(remaining[0]["id"])
                else:
                    new_sid = history_store.create_session("New Chat")
                    st.session_state.active_session_id = new_sid
                    st.session_state.messages = []
                st.rerun()

        sessions_list = history_store.list_sessions()
        if sessions_list:
            with st.expander(f"🕒 Recent Chats ({len(sessions_list)})", expanded=True):
                for sess in sessions_list[:8]:
                    is_active = sess["id"] == st.session_state.active_session_id
                    prefix = "▶ " if is_active else "💬 "
                    btn_label = f"{prefix}{sess.get('title', 'New Chat')}"
                    if st.button(btn_label, key=f"sess_{sess['id']}", use_container_width=True):
                        st.session_state.active_session_id = sess["id"]
                        st.session_state.messages = history_store.get_messages(sess["id"])
                        st.rerun()

        st.markdown('<div class="sidebar-section-label">Security & Governance</div>', unsafe_allow_html=True)
        selected_role_label = st.selectbox(
            "RBAC Role",
            list(RBAC_POLICIES.keys()),
            label_visibility="collapsed",
        )
        mask_pii = st.toggle("🔒 Mask PII Data", value=True)

        st.markdown('<div class="sidebar-section-label">Data & Memory</div>', unsafe_allow_html=True)
        with st.expander("📂 Local File Federation (DuckDB)", expanded=False):
            uploaded_files = st.file_uploader(
                "Upload CSV or Excel",
                type=["csv", "xlsx"],
                accept_multiple_files=True,
                label_visibility="collapsed",
            )

        with st.expander(f"⭐ Golden Memory ({len(golden_store.entries)})", expanded=False):
            if golden_store.entries:
                for item in golden_store.entries:
                    st.caption(f"**Q:** {item['question']}")
                    st.code(item["sql"], language="sql")
                if st.button("Clear Memory", use_container_width=True):
                    golden_store.clear()
                    st.rerun()
            else:
                st.caption("Click **👍 Verify Query** on any response to save reusable SQL patterns.")

    st.markdown("---")
    st.caption(
        "Shehbaz Makhmoor Ali • "
        "[GitHub](https://github.com/Shehbaz-Makhmoor-Ali) • "
        "[LinkedIn](https://linkedin.com/in/shehbaz-makhmoor-ali-5b8643235)"
    )

# ---------------------------------------------------------
# Main Canvas
# ---------------------------------------------------------
if not st.session_state.db_connected:
    st.markdown(
        """
        <div class="hero-container">
            <div class="hero-title">Ask your <span class="hero-accent">database.</span> ⚡</div>
            <div class="hero-subtitle">Connect your PostgreSQL database in the sidebar to launch your governed AI Data Studio copilot.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    engine = get_db_engine(st.session_state.db_uri)
    semantic_layer = SemanticLayer("semantic_layer.json")
    active_rls_policy = RBAC_POLICIES[selected_role_label]

    fed_engine = None
    is_federated = False
    if uploaded_files:
        fed_engine = FederatedEngine(st.session_state.db_uri)
        loaded_names = []
        for f in uploaded_files:
            try:
                local_df = pd.read_csv(f) if f.name.endswith(".csv") else pd.read_excel(f)
                tbl_name = fed_engine.register_dataframe(f.name, local_df)
                loaded_names.append(tbl_name)
            except Exception as e:
                st.sidebar.error(f"Failed to load {f.name}: {e}")
        if loaded_names:
            is_federated = True
            st.sidebar.success(f"🦆 Federated: {', '.join(loaded_names)}")

    pg_schema = fetch_schema_context(engine, prefix="pg." if is_federated else "")
    schema_context = (
        f"{pg_schema}\n\n{fed_engine.get_uploaded_schema()}"
        if is_federated
        else pg_schema
    )

    sql_agent = SQLAgent(
        gemini_model,
        schema_context,
        semantic_layer.get_prompt_context(),
        golden_store=golden_store,
        federated=is_federated,
        rls_policy=active_rls_policy,
    )
    viz_agent = VisualizationAgent(gemini_model)
    insight_agent = InsightAgent(gemini_model, schema_context, federated=is_federated)

    # =====================================================
    # MODE 1: EVALUATION & SECURITY BENCHMARK SUITE
    # =====================================================
    if workspace_mode == "📊 Evaluation & Benchmark":
        st.markdown("## 📊 System Evaluation & Security Benchmark")
        st.caption(
            "Quantitative verification of AST guardrails, deterministic RBAC enforcement, "
            "PII redaction, and live semantic SQL execution accuracy."
        )

        b_col1, b_col2 = st.columns(2)
        with b_col1:
            if st.button("🛡️ Run Security & Governance Suite (Instant)", use_container_width=True):
                st.session_state.sec_bench_df = BenchmarkEngine.run_security_suite()
        with b_col2:
            if st.button("⚡ Run Live LLM & Semantic Suite (2 API Calls)", use_container_width=True):
                with st.spinner("Executing live semantic accuracy tests against Supabase..."):
                    eval_agent = SQLAgent(
                        gemini_model,
                        fetch_schema_context(engine),
                        semantic_layer.get_prompt_context(),
                        golden_store=golden_store,
                        federated=False,
                        rls_policy=None,
                    )
                    st.session_state.live_bench_df = BenchmarkEngine.run_live_llm_suite(eval_agent, engine)

        if st.session_state.sec_bench_df is not None:
            sec_df = st.session_state.sec_bench_df
            pass_rate = round((sec_df["Status"] == "PASS").mean() * 100, 1)
            m1, m2, m3 = st.columns(3)
            m1.metric("Security & Governance Pass Rate", f"{pass_rate}%")
            m2.metric("Tests Passed", f"{(sec_df['Status'] == 'PASS').sum()} / {len(sec_df)}")
            m3.metric("API Cost", "0 Calls (Deterministic)")
            st.dataframe(sec_df, use_container_width=True)

        if st.session_state.live_bench_df is not None:
            live_df = st.session_state.live_bench_df
            live_rate = round((live_df["Status"] == "PASS").mean() * 100, 1)
            avg_lat = round(live_df["Latency (s)"].mean(), 2)
            l1, l2 = st.columns(2)
            l1.metric("Live Semantic & Execution Accuracy", f"{live_rate}%")
            l2.metric("Average End-to-End Latency", f"{avg_lat}s")
            st.dataframe(live_df, use_container_width=True)

        if st.session_state.sec_bench_df is not None or st.session_state.live_bench_df is not None:
            report_parts = [
                "# AI Data Studio — Evaluation & Security Benchmark Report",
                f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                f"**Model:** `{model_display_name}`",
                "",
            ]
            if st.session_state.sec_bench_df is not None:
                report_parts.extend([
                    "## 1. Deterministic Security & Governance Suite",
                    st.session_state.sec_bench_df.to_markdown(index=False),
                    "",
                ])
            if st.session_state.live_bench_df is not None:
                report_parts.extend([
                    "## 2. Live LLM Execution & Semantic Compliance Suite",
                    st.session_state.live_bench_df.to_markdown(index=False),
                    "",
                ])
            st.download_button(
                "📄 Export Benchmark Report (.md)",
                data="\n".join(report_parts),
                file_name="ai_data_studio_benchmark.md",
                mime="text/markdown",
            )

    # =====================================================
    # MODE 2: PROFESSIONAL INTERACTIVE ANALYTICS STUDIO
    # =====================================================
    else:
        current_title = history_store.sessions.get(st.session_state.active_session_id, {}).get("title", "New Chat")
        role_badge = active_rls_policy["role"] if active_rls_policy else "Admin"
        st.markdown(
            f"""
            <div class="chat-header-bar">
                <span class="chat-header-title">💬 {current_title}</span>
                <span class="chat-header-badge">🛡️ Role: {role_badge} • {"🔒 PII Mask On" if mask_pii else "🔓 PII Mask Off"}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if active_rls_policy:
            st.info(
                f"🛡️ **RBAC Active ({active_rls_policy['role']}):** All SQL queries on `{active_rls_policy['table']}` "
                f"are deterministically rewritten via AST to enforce `{active_rls_policy['column']} = '{active_rls_policy['value']}'`."
            )

        # Empty-state quick-start prompts for a fresh chat
        if not st.session_state.messages:
            st.markdown(
                """
                <div class="hero-container">
                    <div class="hero-title">What would you like to <span class="hero-accent">analyze?</span></div>
                    <div class="hero-subtitle">Ask in plain English. Every query is validated by AST guardrails, semantic rules, and statistical diagnostics.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            q1, q2, q3 = st.columns(3)
            starter_prompts = [
                "What is our total Net Revenue by Country for the top 5 countries?",
                "Which 5 products generate the highest Net Revenue?",
                "What is our monthly Net Revenue trend across all valid orders?",
            ]
            for col, sp in zip((q1, q2, q3), starter_prompts):
                with col:
                    if st.button(f"✨ {sp}", use_container_width=True):
                        st.session_state.pending_prompt = sp
                        st.rerun()

        # Render conversation history
        for idx, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"]):
                if msg["role"] == "user":
                    st.markdown(msg["content"])
                else:
                    render_message_badges(msg.get("meta"), msg.get("masked_cols"))

                    if "sql" in msg:
                        with st.expander("Executed SQL Query", expanded=False):
                            st.code(msg["sql"], language="sql")
                    if "df" in msg and msg["df"] is not None:
                        st.dataframe(msg["df"], use_container_width=True)
                    if "figure" in msg and msg["figure"] is not None:
                        st.plotly_chart(msg["figure"], use_container_width=True)

                    if msg.get("content"):
                        st.markdown('<div class="briefing-header">Analytical Briefing</div>', unsafe_allow_html=True)
                        st.markdown(msg["content"])

                    # 4-Column Action Bar
                    if msg.get("df") is not None and not msg["df"].empty:
                        if "root_cause" in msg:
                            with st.expander("🔍 Root-Cause Attribution Brief", expanded=True):
                                st.markdown(msg["root_cause"]["brief"])
                                for diag in msg["root_cause"]["slices"]:
                                    st.caption(f"**Slice:** {diag['dimension']}")
                                    if diag.get("df") is not None:
                                        st.dataframe(diag["df"], use_container_width=True)

                        act1, act2, act3, act4 = st.columns(4)
                        with act1:
                            if "root_cause" not in msg:
                                if st.button("🔍 Root Causes", key=f"rca_{idx}", use_container_width=True):
                                    with st.spinner("Running multi-dimensional root-cause analysis..."):
                                        rca_agent = RootCauseAgent(
                                            gemini_model,
                                            schema_context,
                                            semantic_layer.get_prompt_context(),
                                            federated=is_federated,
                                            rls_policy=active_rls_policy,
                                        )
                                        slices, brief = rca_agent.investigate(
                                            msg.get("question", ""),
                                            msg.get("sql", ""),
                                            msg["df"],
                                            engine,
                                            fed_engine=fed_engine,
                                        )
                                        msg["root_cause"] = {"slices": slices, "brief": brief}
                                        history_store.save_messages(st.session_state.active_session_id, st.session_state.messages)
                                        st.rerun()
                        with act2:
                            if msg.get("verified"):
                                st.button("✅ Verified", key=f"ver_{idx}", disabled=True, use_container_width=True)
                            else:
                                if st.button("👍 Verify Query", key=f"ver_{idx}", use_container_width=True):
                                    golden_store.save(msg.get("question", ""), msg.get("sql", ""))
                                    msg["verified"] = True
                                    history_store.save_messages(st.session_state.active_session_id, st.session_state.messages)
                                    st.rerun()
                        with act3:
                            brief_md = build_executive_brief(msg)
                            st.download_button(
                                "📄 Brief (.md)",
                                data=brief_md,
                                file_name=f"executive_brief_{idx}.md",
                                mime="text/markdown",
                                key=f"dl_md_{idx}",
                                use_container_width=True,
                            )
                        with act4:
                            csv_data = msg["df"].to_csv(index=False).encode("utf-8")
                            st.download_button(
                                "📊 Data (.csv)",
                                data=csv_data,
                                file_name=f"query_results_{idx}.csv",
                                mime="text/csv",
                                key=f"dl_csv_{idx}",
                                use_container_width=True,
                            )

        # Chat input (also handles clickable starter prompts)
        typed_prompt = st.chat_input("Ask a question about your data...")
        user_prompt = typed_prompt or st.session_state.pending_prompt
        if st.session_state.pending_prompt:
            st.session_state.pending_prompt = None

        if user_prompt:
            st.session_state.messages.append({"role": "user", "content": user_prompt})
            history_store.save_messages(st.session_state.active_session_id, st.session_state.messages)
            with st.chat_message("user"):
                st.markdown(user_prompt)

            with st.chat_message("assistant"):
                is_safe, block_msg = SQLGuardrail.check_prompt_intent(user_prompt)
                if not is_safe:
                    st.warning(block_msg)
                    st.session_state.messages.append({"role": "assistant", "content": block_msg})
                    history_store.save_messages(st.session_state.active_session_id, st.session_state.messages)
                    st.stop()

                with st.spinner("Synthesizing query with AST guardrails, RBAC & semantic rules..."):
                    df, final_sql, sql_err, sql_meta = sql_agent.generate_and_verify(
                        user_prompt=user_prompt,
                        engine=engine,
                        history=st.session_state.messages,
                        fed_engine=fed_engine,
                    )

                if sql_err:
                    st.error(f"Execution Error: {sql_err}")
                    st.session_state.messages.append({"role": "assistant", "content": f"Failed to execute query: {sql_err}"})
                    history_store.save_messages(st.session_state.active_session_id, st.session_state.messages)
                else:
                    if df is not None and not df.empty:
                        masked_cols = []
                        if mask_pii:
                            df, masked_cols = PIIMasker.mask_dataframe(df)

                        render_message_badges(sql_meta, masked_cols)

                        with st.expander("Executed SQL Query", expanded=False):
                            st.code(final_sql, language="sql")

                        st.dataframe(df, use_container_width=True)

                        fig = viz_agent.generate_figure(user_prompt, df)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)

                        st.markdown('<div class="briefing-header">Analytical Briefing</div>', unsafe_allow_html=True)
                        insight_placeholder = st.empty()
                        accumulated_insight = ""

                        for chunk in insight_agent.generate_insight_stream(user_prompt, final_sql, df):
                            accumulated_insight += chunk
                            insight_placeholder.markdown(accumulated_insight + "▌")
                        insight_placeholder.markdown(accumulated_insight)

                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": accumulated_insight,
                                "sql": final_sql,
                                "meta": sql_meta,
                                "masked_cols": masked_cols,
                                "df": df,
                                "figure": fig,
                                "question": user_prompt,
                                "verified": False,
                            }
                        )
                        history_store.save_messages(st.session_state.active_session_id, st.session_state.messages)
                        st.rerun()
                    else:
                        render_message_badges(sql_meta, [])
                        with st.expander("Executed SQL Query", expanded=False):
                            st.code(final_sql, language="sql")
                        st.info("Query returned an empty result set.")
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": "The query executed successfully but returned 0 rows.",
                                "sql": final_sql,
                                "meta": sql_meta,
                                "masked_cols": [],
                                "df": df,
                                "figure": None,
                                "question": user_prompt,
                                "verified": False,
                            }
                        )
                        history_store.save_messages(st.session_state.active_session_id, st.session_state.messages)
                        st.rerun()