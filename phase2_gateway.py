import os
import logging
from dotenv import load_dotenv
from litellm import Router, completion

# Load API keys from .env file
load_dotenv()

# We configure logging to see when LiteLLM triggers a fallback
logging.basicConfig(level=logging.INFO)

# ==========================================
# 1. Gateway Configuration using LiteLLM Router
# ==========================================
# We define a list of models.
# Model 1 (Primary): gemini-1.5-pro, but we'll force it to fail by passing a bad key.
# Model 2 (Fallback): gemini-1.5-flash, which will use the real GEMINI_API_KEY.

gemini_api_key = os.getenv("GEMINI_API_KEY", "your-api-key-here")

model_list = [
    {
        "model_name": "research-pro-model",
        "litellm_params": {
            "model": "gemini/gemini-1.5-pro",
            "api_key": "BAD_EXPIRED_KEY_123", # Force fail!
        }
    },
    {
        "model_name": "research-flash-model",
        "litellm_params": {
            "model": "gemini/gemini-1.5-flash",
            "api_key": gemini_api_key,
        }
    }
]

# Initialize the LiteLLM Router
router = Router(model_list=model_list)

# ==========================================
# 2. Resilient Completion Call with Fallbacks
# ==========================================
def call_llm_with_resilience(prompt: str):
    print("\n--- Sending request to the LLM Gateway ---")
    print(f"Prompt: {prompt}")
    
    # We call the primary pro model, but list the flash model as a fallback.
    # If the primary fails, LiteLLM catches the exception and routes to the fallback.
    try:
        response = router.completion(
            model="research-pro-model",
            messages=[{"role": "user", "content": prompt}],
            fallbacks=["research-flash-model"]
        )
        
        # Determine if a fallback was triggered
        # LiteLLM includes model metadata in the response object
        actual_model_used = response.get("_response_ms", {}).get("model", "")
        # If the actual model used does not contain 'pro', then fallback occurred.
        # Note: LiteLLM also adds a headers field or we can check response.model
        model_name_used = response.model
        
        print("\n--- Gateway Response ---")
        print(f"Response: {response.choices[0].message.content.strip()}")
        print(f"Model actually used: {model_name_used}")
        
        if "pro" not in model_name_used:
            print("\n[WARNING to User/System]: Primary model 'gemini-1.5-pro' failed.")
            print("Fallback triggered: 'gemini-1.5-flash' was used instead.")
            print("Note: The quality or depth of this summary might be slightly reduced.")
            
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"\n[CRITICAL ERROR]: Both primary and fallback models failed!")
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    if gemini_api_key == "your-api-key-here" or not gemini_api_key:
        print("[WARNING]: GEMINI_API_KEY not found in environment or .env file.")
        print("Please create a .env file with GEMINI_API_KEY=your_key to run this script fully.")
        print("Simulating local test run...")
    
    call_llm_with_resilience("Write a one-sentence definition of Agentic AI.")
