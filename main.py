import os
from dotenv import load_dotenv

load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "").strip().strip('"').strip("'")
print(f"🔑 Debug: Loaded HF token (Length: {len(os.environ['HF_TOKEN'])})")

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from typing import Annotated
from langgraph.prebuilt import ToolNode
from tools import run_sqlite_query

class State(TypedDict):
    messages: Annotated[list, add_messages]

llm_endpoint = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
)
llm = ChatHuggingFace(llm=llm_endpoint)
tools = [run_sqlite_query]
llm_with_tools = llm.bind_tools(tools)

system_prompt = """You are a Data Analyst AI. 
You have access to a SQLite database with a single table named 'sales'.
The 'sales' table has the following columns: InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country.

CRITICAL ANALYTICAL RULES:
1. For merchandise/product rankings, unit price extremes, or top-selling queries, EXCLUDE non-inventory and administrative adjustment rows (e.g., add WHERE conditions ignoring descriptions like 'MANUAL', 'POSTAGE', 'ADJUST', 'BAD DEBT', 'CARRIAGE', 'BANK CHARGES').
2. Write clean SQLite queries, execute via `run_sqlite_query`, and translate results into a concise, natural English summary."""

def agent_node(state: State):
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)

def route_tools(state: State):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END

graph_builder = StateGraph(State)
graph_builder.add_node("agent", agent_node)
graph_builder.add_node("tools", tool_node)
graph_builder.add_edge(START, "agent")
graph_builder.add_conditional_edges("agent", route_tools, {"tools": "tools", END: END})
graph_builder.add_edge("tools", "agent")
app = graph_builder.compile()

print("\n" + "="*50)
print("🤖 Autonomous SQL Agent Initialized (Hugging Face Mode)")
print("Type 'exit' or 'quit' to stop.")
print("="*50 + "\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit', 'q']:
        print("Agent shutting down. Goodbye!")
        break
        
    try:
        final_state = app.invoke({"messages": [("user", user_input)]})
        msg_content = final_state["messages"][-1].content
        if isinstance(msg_content, list):
            clean_text = " ".join([b.get("text", str(b)) if isinstance(b, dict) else str(b) for b in msg_content])
        else:
            clean_text = str(msg_content)
        print(f"\nAgent:\n{clean_text}\n")
    except Exception as e:
        print(f"\nAgent: Error occurred -> {e}\n")