import os
import json
import textwrap
from typing import List, Optional
from openai import OpenAI
from dotenv import load_dotenv 

# Load environment variables
load_dotenv() 

from env import TalentArbitrageEnv, Action

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-70B-Instruct") 
BENCHMARK = "QuantHire"
MAX_STEPS = 15

SYSTEM_PROMPT = textwrap.dedent("""
    You are an autonomous sourcing agent utilizing a Talent Graph network.
    Reply ONLY with a strictly valid JSON object representing your next action.

    Valid commands: "search", "shortlist", "submit", "graph_recommend"
    Schema:
    {
        "command": "string",
        "search_skill": "string",
        "search_region": "string",
        "target_skill": "string",
        "candidate_id": 0
    }
""").strip()

# ==========================================
# EXACT GRADER FORMATTING HELPER FUNCTIONS
# ==========================================
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)

# ==========================================
# MAIN LOOP
# ==========================================
def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    tasks = ["easy", "medium", "hard"]
    
    for task_name in tasks:
        env = TalentArbitrageEnv(task_level=task_name)
        obs = env.reset()
        
        history = []
        rewards = []
        steps_taken = 0
        score = 0.0
        success = False
        
        log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)
        
        try:
            for step in range(1, MAX_STEPS + 1):
                user_prompt = f"Observation: {obs.model_dump_json()}"
                
                try:
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt}
                        ],
                        response_format={"type": "json_object"},
                        temperature=0.1,
                        max_tokens=200
                    )
                    raw_response = response.choices[0].message.content
                    action_data = json.loads(raw_response)
                    action = Action(**action_data)
                    action_str = action.command
                    error = None
                except Exception as e:
                    action = Action(command="submit")
                    action_str = "submit"
                    error = str(e).replace('\n', ' ') # Ensure error has no newlines

                obs, reward, done, info = env.step(action)
                
                rewards.append(reward)
                steps_taken = step
                
                log_step(step=step, action=action_str, reward=reward, done=done, error=error)
                
                if done:
                    break
                    
            # Because our intermediate steps are 0.0, the final reward is our total score
            score = rewards[-1] if rewards else 0.01
            score = min(max(score, 0.01), 0.99) # Extra safety clamp
            success = score >= 0.5
            
        except Exception as e:
            print(f"[DEBUG] Error: {e}", flush=True)
        finally:
            log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    main()