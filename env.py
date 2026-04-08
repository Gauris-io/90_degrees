import logging
from typing import List, Tuple, Dict, Any, Literal
from pydantic import BaseModel, Field

# ==========================================
# 1. OPENENV SPEC: PYDANTIC MODELS
# ==========================================
class Candidate(BaseModel):
    id: int
    skill: str
    region: str
    asking_price: float

class Observation(BaseModel):
    client_brief: str = Field(description="The goal the agent must achieve.")
    system_feedback: str = Field(description="Feedback from the last action.")
    search_results: List[Candidate] = Field(default_factory=list)
    shortlist: List[Candidate] = Field(default_factory=list)

class Action(BaseModel):
    command: Literal["search", "shortlist", "submit", "graph_recommend"] = Field(
        description="The action to take."
    )
    search_skill: str = Field(default="", description="Skill to search for.")
    search_region: str = Field(default="", description="Region to search in.")
    candidate_id: int = Field(default=-1, description="ID of candidate to shortlist.")
    target_skill: str = Field(default="", description="Target skill to find structural equivalents for using GNN.")

# ==========================================
# 2. REAL-WORLD TASK: SOURCING TERMINAL
# ==========================================
class TalentArbitrageEnv:
    def __init__(self, task_level: Literal["easy", "medium", "hard"] = "easy"):
        self.task_level = task_level
        self.max_steps = 15
        self.current_step = 0
        
        # Mock Database representing the Global Graph
        self.database = [
            Candidate(id=1, skill="Madhubani", region="Bihar", asking_price=500),
            Candidate(id=2, skill="Madhubani", region="Rajasthan", asking_price=2500),
            Candidate(id=3, skill="React", region="Bangalore", asking_price=4000),
            Candidate(id=4, skill="React", region="Remote_Tier3", asking_price=1200),
            Candidate(id=5, skill="Vue", region="Bihar", asking_price=900), # Hidden Gem
            Candidate(id=6, skill="Warli", region="Maharashtra", asking_price=600), # Hidden Gem
        ]
        
        self._search_results: List[Candidate] = []
        self._shortlist: List[Candidate] = []
        self._feedback = "System initialized. Awaiting commands."

    def _get_client_brief(self) -> str:
        if self.task_level == "easy":
            return "Find 1 Madhubani artist with an asking price under 1000."
        elif self.task_level == "medium":
            return "Find 1 React equivalent developer under 1000 using the GNN graph_recommend tool."
        else: # hard
            return "Find 1 Madhubani artist < 1000 AND 1 React equivalent dev < 1000."

    def state(self) -> Observation:
        return Observation(
            client_brief=self._get_client_brief(),
            system_feedback=self._feedback,
            search_results=self._search_results,
            shortlist=self._shortlist
        )

    def reset(self) -> Observation:
        self.current_step = 0
        self._search_results = []
        self._shortlist = []
        self._feedback = "System ready."
        return self.state()

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        self.current_step += 1
        # Use 0.01 as a base non-zero floor
        reward = 0.01
        done = False
        
        if action.command == "search":
            self._search_results = [
                c for c in self.database 
                if (not action.search_skill or action.search_skill.lower() in c.skill.lower())
                and (not action.search_region or action.search_region.lower() in c.region.lower())
            ]
            self._feedback = f"Search returned {len(self._search_results)} results."
            if len(self._search_results) > 0: reward += 0.1

        elif action.command == "graph_recommend":
            self._feedback = f"Running Graph Message Passing for: {action.target_skill}"
            graph_embeddings = {
                "React": ["Vue", "Angular"],
                "Madhubani": ["Warli", "Pattachitra"]
            }
            adjacent_skills = graph_embeddings.get(action.target_skill, [])
            if adjacent_skills:
                hidden_talent = [c for c in self.database if c.skill in adjacent_skills]
                self._search_results = hidden_talent
                self._feedback += f" | Found {len(hidden_talent)} candidates with adjacent graph embeddings."
                reward += 0.3 
            else:
                self._feedback += " | No adjacent nodes found."

        elif action.command == "shortlist":
            candidate = next((c for c in self._search_results if c.id == action.candidate_id), None)
            if candidate:
                if candidate not in self._shortlist:
                    self._shortlist.append(candidate)
                    self._feedback = f"Added Candidate {candidate.id} to shortlist."
                    reward += 0.2
                else:
                    self._feedback = "Candidate already shortlisted."
            else:
                self._feedback = "Invalid candidate ID."

        elif action.command == "submit":
            final_score = self._grade_submission()
            reward += final_score
            done = True
            self._feedback = f"Shortlist submitted. Score: {final_score}"

        if self.current_step >= self.max_steps:
            done = True
            self._feedback = "Max steps reached."
            reward += self._grade_submission()

        # CLAMP: Ensure reward is strictly between 0 and 1 (0.01 to 0.99)
        reward = max(0.01, min(0.99, reward))

        return self.state(), reward, done, {"task_level": self.task_level}

    def _grade_submission(self) -> float:
        # Use 0.01 for absolute failures and 0.99 for absolute success
        if not self._shortlist: return 0.01
        
        if self.task_level == "easy":
            if len(self._shortlist) == 1 and self._shortlist[0].skill == "Madhubani" and self._shortlist[0].asking_price < 1000: 
                return 0.99
            return 0.5 if len(self._shortlist) == 1 else 0.01

        elif self.task_level == "medium":
            has_react_alt = any(c.skill in ["Vue", "Angular"] and c.asking_price < 1000 for c in self._shortlist)
            return 0.99 if has_react_alt and len(self._shortlist) == 1 else 0.01

        elif self.task_level == "hard":
            has_madhubani = any(c.skill == "Madhubani" and c.asking_price < 1000 for c in self._shortlist)
            has_react_alt = any(c.skill in ["Vue", "Angular"] and c.asking_price < 1000 for c in self._shortlist)
            if has_madhubani and has_react_alt and len(self._shortlist) == 2: 
                return 0.99
            elif has_madhubani or has_react_alt: 
                return 0.5
            return 0.01