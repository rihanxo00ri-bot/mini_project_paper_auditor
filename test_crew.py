from pathlib import Path
import os
import time
from dotenv import load_dotenv
from crewai import LLM
from crewai.project.crew_loader import load_crew

load_dotenv()

BASE_DIR = Path(__file__).parent
CREW_PATH = BASE_DIR / "crew.jsonc"

def free_tier_cooldown(agent_step):
    print("\n ⏳ Free tier rate-limit cooldown: Sleeping for 20 seconds...")
    time.sleep(20)
    

groq_llm = LLM(
    model="openai/gpt-oss-20b",
    provider="openai",
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)
groq_llm_2 = LLM(
    model="groq/llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    max_tokens=2048
)
gemini_llm = LLM(
    model="gemini/gemini-3.6-flash", 
    api_key=os.getenv("GEMINI_API_KEY"), 
)

omniroute_llm = LLM(
    model="openai/auto",
    api_key="OMNIROUTE_API_KE",
    base_url="http://localhost:20128/v1",
)

crew, default_inputs = load_crew(CREW_PATH)

# Use Groq GPT-OSS 20B for every agent
crew.agents[0].llm = omniroute_llm
crew.agents[1].llm = omniroute_llm
crew.agents[2].llm = groq_llm
crew.agents[3].llm = groq_llm

for agent in crew.agents:
    agent.step_callback = free_tier_cooldown


crew.max_rpm = 3

print("\nLLM ASSIGNMENT")
print("==============================")

for agent in crew.agents:
    print(f"{agent.role} -> {agent.llm.model}")

print("==============================\n")

paper_path = Path(
    
    r"C:\Users\FARMAN\OneDrive\Documents\Downloads\NIPS-2017-attention-is-all-you-need-Paper.pdf"
)

if not paper_path.exists():
    raise FileNotFoundError(
        f"Research paper not found: {paper_path}"
    )


if __name__ == "__main__":

    result = crew.kickoff(
        inputs={
            "paper_path": str(paper_path.resolve())
        }
    )

    print("\n\n==============================")
    print("FINAL RESEARCH AUDIT")
    print("==============================\n")

    print(result.raw)