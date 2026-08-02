import os
import json
import base64
import redis
import psycopg
import litellm
from dotenv import load_dotenv
from typing import Annotated, List, TypedDict
import operator
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

# Load API keys
load_dotenv()

# ==========================================
# Pydantic Schema for Structured Outputs
# ==========================================
class AgentAction(BaseModel):
    action: str = Field(description="Must be one of: 'web_search', 'ltm_lookup', or 'finalize'")
    query: str = Field(description="Search keywords if calling a tool, otherwise leave empty")
    answer: str = Field(description="Your final conversational answer if action is finalize, otherwise leave empty")

# Setup config
MODEL_NAME = "gemini/gemini-3.5-flash"
REDIS_HOST = "localhost"
REDIS_PORT = 6379
DB_DSN = "dbname=research_db user=postgres password=postgres host=localhost port=5433"

# Initialize Redis
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True)

# ==========================================
# 1. State Definition
# ==========================================
class ReActState(TypedDict):
    user_query: str
    session_id: str
    
    # ReAct variables
    tool_action: str       # 'web_search', 'ltm_lookup', or 'finalize'
    tool_query: str        # The query to pass to the tool
    tool_result: str       # The output from the tool
    
    # Accumulated context
    context: Annotated[List[str], operator.add]
    final_answer: str
    
    # Injected user preferences
    user_preferences: str

# ==========================================
# 2. Tool Definitions
# ==========================================
def dummy_web_search(query: str) -> str:
    print(f"   [Tool Executing] DuckDuckGo Web Search: '{query}'")
    # Simulate a web search result
    return f"Search result for '{query}': Gemini 3.5 Flash was released in mid-2026 with native speed and reasoning features."

def database_ltm_lookup(query: str) -> str:
    print(f"   [Tool Executing] PostgreSQL pgvector lookup: '{query}'")
    # Query LTM table if it exists, otherwise return a simulated database match
    try:
        # In a real run, we'd use pgvector cosine similarity.
        # Let's return a clean match to show LTM retrieval.
        return f"Database LTM match for '{query}': Report from last week states that Agentic AI loop latency averages 4.2 seconds in local docker environments."
    except Exception as e:
        return f"Database search failed: {e}"

# ==========================================
# 3. Node Definitions
# ==========================================
def agent_node(state: ReActState):
    print("\n[Node: ReAct Agent]")
    query = state["user_query"]
    past_context = "\n".join(state.get("context", []))
    preferences = state.get("user_preferences", "None")
    
    # System prompt describing the ReAct loop
    system_prompt = (
        "You are an advanced autonomous ReAct Agent.\n"
        "Your goal is to answer the user query accurately.\n"
        "You have access to two tools:\n"
        "1. 'web_search': Use this if you need current information from the web.\n"
        "2. 'ltm_lookup': Use this to query past research reports stored in the database.\n"
        "\n"
        f"User Preferences to follow:\n{preferences}\n\n"
        "Review your user query, past chat history, and any tool results you have already collected.\n"
        "Decide your next action. You MUST respond strictly in JSON format:\n"
        "{\n"
        '  "action": "web_search" | "ltm_lookup" | "finalize",\n'
        '  "query": "search keywords if calling a tool, otherwise leave empty",\n'
        '  "answer": "your final conversational answer if action is finalize, otherwise leave empty"\n'
        "}"
    )
    
    user_prompt = f"User Query: {query}\n\nAccumulated Tool Results:\n{past_context}"
    
    try:
        response = litellm.completion(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format=AgentAction
        )
        
        content = response.choices[0].message.content.strip()
        result = AgentAction.model_validate_json(content)
        
        action = result.action
        tool_query = result.query
        answer = result.answer
        
        print(f"-> Agent Decision: Action='{action}' | Query='{tool_query}'")
        if answer:
            print(f"-> Agent Final Answer Drafted.")
            
        return {
            "tool_action": action,
            "tool_query": tool_query,
            "final_answer": answer
        }
    except Exception as e:
        print(f"-> Agent failed: {e}. Finalizing.")
        return {
            "tool_action": "finalize",
            "final_answer": "Failed to run agent loop."
        }

def tool_executor_node(state: ReActState):
    print("\n[Node: Tool Executor]")
    action = state["tool_action"]
    query = state["tool_query"]
    
    if action == "web_search":
        result = dummy_web_search(query)
    elif action == "ltm_lookup":
        result = database_ltm_lookup(query)
    else:
        result = "No tool run requested."
        
    return {
        "context": [f"Tool ({action}) output: {result}"]
    }

# ==========================================
# 4. Routing Logic
# ==========================================
def route_react(state: ReActState):
    action = state["tool_action"]
    if action == "finalize":
        return "end"
    else:
        return "execute_tool"

# ==========================================
# 5. Build Graph with Checkpointer (MemorySaver)
# ==========================================
builder = StateGraph(ReActState)

builder.add_node("agent", agent_node)
builder.add_node("tools", tool_executor_node)

builder.add_edge(START, "agent")

builder.add_conditional_edges(
    "agent",
    route_react,
    {
        "execute_tool": "tools",
        "end": END
    }
)

builder.add_edge("tools", "agent")  # Dynamic ReAct loop!

# We instantiate MemorySaver to serve as our checkpointer (State Persistence)
checkpointer = MemorySaver()

# Compile the graph passing the checkpointer
react_graph = builder.compile(checkpointer=checkpointer)

# ==========================================
# 6. Simulation & Verification Run
# ==========================================
if __name__ == "__main__":
    print("--- Starting Phase 8 Advanced Verification ---")
    session_id = "session_react_99"
    thread_config = {"configurable": {"thread_id": "thread_react_001"}}
    
    # ==========================================
    # C. Testing Episodic Memory Loader
    # ==========================================
    print("\n1. Injecting user preference into Episodic Memory (Redis)...")
    # Simulate the user previously telling the system they want short answers with bullet points
    redis_key = f"episodic_memory:{session_id}"
    r.set(redis_key, "User prefers very brief answers (under 2 sentences) and likes using emojis.")
    
    # Retrieve it
    user_prefs = r.get(redis_key)
    print(f"Loaded User Preferences: '{user_prefs}'")
    
    # ==========================================
    # A & D. Running the Graph (ReAct + Checkpointing)
    # ==========================================
    print("\n2. Executing ReAct Graph with Checkpointing...")
    
    # Query: Requires calling the web search tool
    initial_input = {
        "user_query": "When was Gemini 3.5 Flash released?",
        "session_id": session_id,
        "tool_action": "",
        "tool_query": "",
        "tool_result": "",
        "context": [],
        "final_answer": "",
        "user_preferences": user_prefs
    }
    
    # We invoke the graph passing our thread configuration
    # LangGraph will checkpoint the state automatically at every step under this thread_id
    final_state = react_graph.invoke(initial_input, config=thread_config)
    
    print("\n==============================================")
    print("              FINAL RESULTS                   ")
    print("==============================================")
    print(f"User Query:   {final_state['user_query']}")
    print(f"Final Answer: {final_state['final_answer']}")
    
    # ==========================================
    # Verify State Checkpoint History
    # ==========================================
    print("\n3. Verifying Checkpoint Time-Travel Log...")
    # Let's inspect the saved checkpoints for this thread!
    state_history = list(react_graph.get_state_history(thread_config))
    print(f"Total saved checkpoints: {len(state_history)}")
    
    for i, state_record in enumerate(state_history):
        print(f"\nCheckpoint {i+1} (Node: {state_record.metadata.get('step', 'START')}):")
        print(f"  Next Action in state: {state_record.values.get('tool_action')}")
        print(f"  Final Answer in state: {state_record.values.get('final_answer')}")
