---
title: QuantHire
emoji: 🌎
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---
# GeoTalent Arbitrage Agent

Built for the OpenEnv Hackathon.

GeoTalent steps outside traditional gaming environments to solve complex real-world logic problems. It is an autonomous sourcing recruiter designed to execute geographic talent arbitrage. It interacts with a simulated terminal to parse client briefs, search a talent database, and build a candidate shortlist while adhering to strict budget constraints. 

### The Arbitrage Angle (Macro-to-Micro)
Rather than relying solely on traditional keyword matching, this agent treats human capital acquisition as a quantitative analysis problem. By observing failure rates and budget constraints in high-demand regions, it identifies market saturation. It then uses a graph-recommendation tool to identify emerging trends in more accessible regions, successfully executing the arbitrage.

### The Alt-Data Pipeline
To build a highly predictive model for production, our data pipeline looks beyond historical job board data and ingests Alternative Data (Alt-Data) as leading indicators:
* **Future Supply:** We track regional NGO upskilling programs, coding bootcamps, and government grants to predict talent influxes in Tier-3 regions before they become saturated.
* **Future Demand:** We monitor early-stage startup registrations and localized seed funding to map out emerging tech hubs.
* **The Graph Engine:** This Alt-Data feeds our Graph Neural Network (GNN), giving the LLM agent a predictive heatmap to secure talent in emerging markets proactively.

### OpenEnv Spec Compliance
We built this strictly to the hackathon rubric:
- **Pydantic Environments:** `env.py` uses strict Pydantic models (`Observation`, `Action`) to manage state safely.
- **Forced JSON Output:** The agent (`inference.py`) forces `response_format={"type": "json_object"}`. It only outputs valid API commands, eliminating conversational hallucination.
- **Grading:** Includes `easy`, `medium`, and `hard` tasks with deterministic graders returning float rewards (0.0 to 1.0).
- **Logging:** Emits the exact `[START]`, `[STEP]`, and `[END]` stdout logs required for the automated evaluation suite.

### Tech Stack
- Python 3.10
- meta-llama/Meta-Llama-3-70B-Instruct (via Hugging Face)
- OpenAI Python Client
- Pydantic
- Docker

---

### How to Run Locally

1. Install dependencies:
   ```bash
   pip install openai pydantic python-dotenv
2. Set your environment
    HF_TOKEN=your_token_here
    API_BASE_URL=[https://router.huggingface.co/v1](https://router.huggingface.co/v1)
    MODEL_NAME=meta-llama/Meta-Llama-3-70B-Instruct
3. Run the agent (Root Executable):
    python inference.py
4. This repository includes a Dockerfile ready for the automated validation suite.
   in your terminal-
   docker build -t geotalent-agent .
   docker run geotalent-agent

