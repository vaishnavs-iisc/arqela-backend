import operator
from typing import Annotated, List, TypedDict
from langgraph.graph import StateGraph, START, END

# ==========================================
# 1. State Definition (The Plate)
# ==========================================
class KitchenState(TypedDict):
    ingredients: str
    status: str
    is_delicious: bool
    # We use Annotated and operator.add as a REDUCER.
    # Any node returning a list for 'chef_history' will APPEND to it
    # instead of overwriting it.
    chef_history: Annotated[List[str], operator.add]

# ==========================================
# 2. Node Definitions (The Chefs)
# ==========================================
def chopper_chef(state: KitchenState):
    print("\n[Node: Chopper Chef]")
    print(f"-> Reading ingredients: {state['ingredients']}")
    print("-> Action: Chopping...")
    return {
        "status": "chopped",
        "chef_history": ["Chopper Chef"]
    }

def cooking_chef(state: KitchenState):
    print("\n[Node: Cooking Chef]")
    print(f"-> Reading status: {state['status']}")
    print("-> Action: Cooking...")
    return {
        "status": "cooked",
        "chef_history": ["Cooking Chef"]
    }

def head_chef(state: KitchenState):
    print("\n[Node: Head Chef]")
    print(f"-> Tasting the dish with status: {state['status']}")
    
    # Taste test logic
    if state["status"] == "cooked":
        print("-> Decision: Perfect!")
        is_delicious = True
    else:
        print("-> Decision: Under-cooked! Needs to go back.")
        is_delicious = False
        
    return {
        "is_delicious": is_delicious,
        "chef_history": ["Head Chef"]
    }

# ==========================================
# 3. Router Definition (Conditional Edge)
# ==========================================
def route_dish(state: KitchenState):
    print("\n[Routing Decision]")
    if state["is_delicious"]:
        print("-> Routing to: Serve Customer (END)")
        return "serve"
    else:
        print("-> Routing to: Cook More (cooking_chef)")
        return "cook_more"

# ==========================================
# 4. Building the Graph
# ==========================================
# Initialize the graph builder with our KitchenState
builder = StateGraph(KitchenState)

# Register our nodes (chefs)
builder.add_node("chopper", chopper_chef)
builder.add_node("cooker", cooking_chef)
builder.add_node("head_chef", head_chef)

# Connect normal edges (flow of plate)
builder.add_edge(START, "chopper")
builder.add_edge("chopper", "cooker")
builder.add_edge("cooker", "head_chef")

# Connect the conditional edge from head_chef
builder.add_conditional_edges(
    "head_chef",
    route_dish,
    {
        "serve": END,
        "cook_more": "cooker"
    }
)

# Compile the graph into a runnable application
kitchen_app = builder.compile()

# ==========================================
# 5. Executing the Graph
# ==========================================
if __name__ == "__main__":
    print("--- Starting Kitchen Simulation ---")
    
    # Initialize our plate
    initial_plate = {
        "ingredients": "potatoes",
        "status": "raw",
        "is_delicious": False,
        "chef_history": []
    }
    
    # Run the graph synchronously
    final_state = kitchen_app.invoke(initial_plate)
    
    print("\n--- Final State of the Serving Plate ---")
    print(f"Ingredients: {final_state['ingredients']}")
    print(f"Status:      {final_state['status']}")
    print(f"Delicious?  {final_state['is_delicious']}")
    print(f"Chefs who worked: {final_state['chef_history']}")
