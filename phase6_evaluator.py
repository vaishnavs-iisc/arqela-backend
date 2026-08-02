import json
import re
from dotenv import load_dotenv
import litellm
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List

load_dotenv()

# We use Gemini 3.5 Flash for both writing and judging to keep it fast
MODEL_NAME = "gemini/gemini-3.5-flash"

# ==========================================
# 1. State Definition
# ==========================================
class EvaluatorState(TypedDict):
    original_sources: List[str]
    draft_report: str
    accuracy_score: int
    critique: str
    loop_count: int
    is_finalized: bool

# ==========================================
# 2. Node Definitions
# ==========================================
def writer_node(state: EvaluatorState):
    print("\n[Node: Writer Agent]")
    loop = state.get("loop_count", 0)
    sources = "\n".join(state["original_sources"])
    
    if loop == 0:
        # First attempt: We will simulate a hallucination/mistake by explicitly
        # prompting the LLM to make a factual error, so we can see the Judge catch it.
        prompt = (
            f"You are a research writer. Based on these sources:\n{sources}\n"
            "Write a brief 2-sentence summary of when Project Apollo was initiated and by whom.\n"
            "CRITICAL WRITING ERROR (for testing): Intentionally write that Project Apollo was started "
            "in the year 1950 by President Dwight D. Eisenhower (even if the sources say otherwise)."
        )
    else:
        # Correction attempt: LLM reads the Judge's feedback and rewrites
        print(f"-> Correction Attempt {loop}. Reading Critique: '{state['critique']}'")
        prompt = (
            f"You are a research writer. Your previous draft had factual errors.\n"
            f"Original Sources:\n{sources}\n\n"
            f"Previous Draft:\n{state['draft_report']}\n\n"
            f"Judge's Critique:\n{state['critique']}\n\n"
            "Please rewrite the summary, correcting all errors pointed out by the Judge. "
            "Ensure it is 100% accurate according to the Original Sources."
        )

    response = litellm.completion(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}]
    )
    
    report_text = response.choices[0].message.content.strip()
    print("-> Draft Report Generated:")
    print(f"   \"{report_text}\"")
    
    return {
        "draft_report": report_text,
        "loop_count": loop + 1
    }

def judge_node(state: EvaluatorState):
    print("\n[Node: Judge Agent]")
    sources = "\n".join(state["original_sources"])
    draft = state["draft_report"]
    
    # We instruct the Judge LLM to check for any claims in the draft not supported by the sources
    system_prompt = (
        "You are an expert factual Auditor. Compare the Draft Report against the Original Sources.\n"
        "Verify every date, name, and fact.\n"
        "Evaluate factual accuracy on a scale of 1 to 5 (5 being perfect, 1 being completely false).\n"
        "If there are errors, explain them in the critique.\n"
        "Output your response strictly as JSON:\n"
        "{\n"
        '  "accuracy_score": int,\n'
        '  "critique": "string"\n'
        "}"
    )
    
    user_prompt = f"Original Sources:\n{sources}\n\nDraft Report:\n{draft}"
    
    try:
        response = litellm.completion(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        # Parse JSON
        content = response.choices[0].message.content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
            
        result = json.loads(content.strip())
        score = int(result.get("accuracy_score", 5))
        critique = result.get("critique", "")
        
        print(f"-> Score: {score}/5")
        if critique:
            print(f"-> Critique: {critique}")
            
        return {
            "accuracy_score": score,
            "critique": critique,
            "is_finalized": (score >= 4)
        }
    except Exception as e:
        print(f"[Error] Judge failed: {e}. Defaulting to safe.")
        return {
            "accuracy_score": 5,
            "critique": "",
            "is_finalized": True
        }

# ==========================================
# 3. Routing Edge
# ==========================================
def route_correction(state: EvaluatorState):
    # Route back to writer if accuracy score is low, but cap at 2 loops to prevent infinite runs
    if state["is_finalized"]:
        print("\n-> Routing: Approved! Ending process.")
        return "end"
    elif state["loop_count"] >= 3:
        print("\n-> Routing: Max loops reached. Ending process with warning.")
        return "end"
    else:
        print("\n-> Routing: Score too low! Routing back to Writer for correction.")
        return "re-write"

# ==========================================
# 4. Build and Compile Graph
# ==========================================
builder = StateGraph(EvaluatorState)

builder.add_node("writer", writer_node)
builder.add_node("judge", judge_node)

builder.add_edge(START, "writer")
builder.add_edge("writer", "judge")

builder.add_conditional_edges(
    "judge",
    route_correction,
    {
        "end": END,
        "re-write": "writer"
    }
)

evaluator_app = builder.compile()

# ==========================================
# 5. Run loop test
# ==========================================
if __name__ == "__main__":
    # Ground truth data
    facts = [
        "Fact Sheet: In May 1961, President John F. Kennedy announced the goal of landing a man on the moon.",
        "Apollo Program History: The Apollo program was officially initiated by NASA in 1961 under the Kennedy administration, following the successful Mercury and Gemini programs."
    ]
    
    initial_state = {
        "original_sources": facts,
        "draft_report": "",
        "accuracy_score": 0,
        "critique": "",
        "loop_count": 0,
        "is_finalized": False
    }
    
    print("--- Starting Evaluator & Self-Correction Loop ---")
    final_output = evaluator_app.invoke(initial_state)
    
    print("\n==============================================")
    print("              FINAL RESULTS                   ")
    print("==============================================")
    print(f"Final Report:     {final_output['draft_report']}")
    print(f"Final Accuracy:   {final_output['accuracy_score']}/5")
    print(f"Total Iterations: {final_output['loop_count']}")
