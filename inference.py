import os
import json
from openai import OpenAI
from dotenv import load_dotenv 

# Load environment variables
load_dotenv() 

from env import TalentArbitrageEnv, Action

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-70B-Instruct") 

SYSTEM_PROMPT = """
You are an autonomous sourcing agent utilizing a Talent Graph network.
Reply ONLY with a strictly valid JSON object representing your next action.

Valid commands: "search", "shortlist", "submit", "graph_recommend"
Schema:
{
    "command": "string",
    "search_skill": "string",
    "search_region": "string",
    "target_skill": "string",
    "candidate_id": integer
}
"""

def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    tasks = ["easy", "medium", "hard"]
    
    for task_name in tasks:
        print(f"[START] Task: {task_name}")
        
        env = TalentArbitrageEnv(task_level=task_name)
        obs = env.reset()
        
        step_count = 0
        done = False
        last_reward = 0.01  # Use this to report the final task score
        
        while not done:
            step_count += 1
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
            except Exception as e:
                action = Action(command="submit")
                
            obs, reward, done, info = env.step(action)
            last_reward = reward  # Update with the most recent step's reward
            
            print(f"[STEP] {step_count} | Action: {action.command} | Step Reward: {reward:.2f} | Done: {done}")
            
            if step_count >= 15: break

        print(f"[END] Task: {task_name} | Final Task Score: {last_reward:.2f}")

if __name__ == "__main__":
    main()