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
    target_skill: str = Field(default="", description="Target skill to find equivalents.")

# ==========================================
# 2. REAL-WORLD TASK: SOURCING TERMINAL
# ==========================================
class TalentArbitrageEnv:
    def __init__(self, task_level: Literal["easy", "medium", "hard"] = "easy"):
        self.task_level = task_level
        self.max_steps = 15
        self.current_step = 0
        
        self.database = [
            Candidate(id=1, skill="Madhubani", region="Bihar", asking_price=500),
            Candidate(id=2, skill="Madhubani", region="Rajasthan", asking_price=2500),
            Candidate(id=3, skill="React", region="Bangalore", asking_price=4000),
            Candidate(id=4, skill="React", region="Remote_Tier3", asking_price=1200),
            Candidate(id=5, skill="Vue", region="Bihar", asking_price=900),
            Candidate(id=6, skill="Warli", region="Maharashtra", asking_price=600),
        ]
        
        self._search_results: List[Candidate] = []
        self._shortlist: List[Candidate] = []
        self._feedback = "System initialized."

    def _get_client_brief(self) -> str:
        if self.task_level == "easy":
            return "Find 1 Madhubani artist with an asking price under 1000."
        elif self.task_level == "medium":
            return "Find 1 React equivalent developer under 1000 using graph_recommend."
        else:
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
        reward = 0.05  # Standard non-zero step reward
        done = False
        
        if action.command == "search":
            self._search_results = [
                c for c in self.database 
                if (not action.search_skill or action.search_skill.lower() in c.skill.lower())
                and (not action.search_region or action.search_region.lower() in c.region.lower())
            ]
            self._feedback = f"Search returned {len(self._search_results)} results."
            reward = 0.1

        elif action.command == "graph_recommend":
            graph_embeddings = {"React": ["Vue", "Angular"], "Madhubani": ["Warli", "Pattachitra"]}
            adjacent_skills = graph_embeddings.get(action.target_skill, [])
            self._search_results = [c for c in self.database if c.skill in adjacent_skills]
            self._feedback = f"Found {len(self._search_results)} graph-adjacent candidates."
            reward = 0.2

        elif action.command == "shortlist":
            candidate = next((c for c in self._search_results if c.id == action.candidate_id), None)
            if candidate and candidate not in self._shortlist:
                self._shortlist.append(candidate)
                self._feedback = f"Added Candidate {candidate.id} to shortlist."
                reward = 0.3
            else:
                self._feedback = "Invalid or duplicate candidate."

        elif action.command == "submit":
            reward = self._grade_submission()
            done = True
            self._feedback = f"Submitted. Score: {reward}"

        if self.current_step >= self.max_steps:
            done = True
            reward = self._grade_submission()

        # CLAMP: Forces output strictly between 0.01 and 0.99
        final_reward = float(max(0.01, min(0.99, reward)))

        return self.state(), final_reward, done, {"task_level": self.task_level}

    def _grade_submission(self) -> float:
        if not self._shortlist: return 0.01
        
        score = 0.01
        if self.task_level == "easy":
            if any(c.skill == "Madhubani" and c.asking_price < 1000 for c in self._shortlist):
                score = 0.99
        
        elif self.task_level == "medium":
            if any(c.skill in ["Vue", "Angular"] and c.asking_price < 1000 for c in self._shortlist):
                score = 0.99
                
        elif self.task_level == "hard":
            has_madhu = any(c.skill == "Madhubani" and c.asking_price < 1000 for c in self._shortlist)
            has_alt = any(c.skill in ["Vue", "Angular"] and c.asking_price < 1000 for c in self._shortlist)
            if has_madhu and has_alt: score = 0.99
            elif has_madhu or has_alt: score = 0.5

        return score