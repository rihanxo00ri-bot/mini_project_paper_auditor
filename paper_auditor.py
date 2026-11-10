
import os
import argparse
import json
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from crewai import LLM, Crew, Task, Process
from crewai.project.crew_loader import load_crew
from intent_normalizer import normalize_user_query

# -----------------------------
# Configuration
# -----------------------------

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
CREW_PATH = BASE_DIR / "crew.jsonc"
OUTPUT_DIR = BASE_DIR / "test_outputs"

OMNI_KEY = os.getenv("OMNIROUTE_API_KEY")
OMNI_BASE_URL = os.getenv(
    "OMNIROUTE_BASE_URL",
    "http://localhost:20128/v1",
)
OMNI_MODEL = os.getenv("OMNIROUTE_MODEL", "auto")


# Map logical names to the exact agent roles.
AGENT_ROLES = {
    "comprehension_mind": "Scientific Research Comprehension Specialist",
    "research_exploration_mind": "Research Exploration Specialist",
    "critical_evaluation_mind": "Scientific Critical Evaluation Specialist",
}


def build_agents():
    if not CREW_PATH.exists():
        raise FileNotFoundError(f"Missing config: {CREW_PATH}")

    if not OMNI_KEY:
        raise ValueError("OMNIROUTE_API_KEY is missing from .env")

    loaded_crew, _ = load_crew(CREW_PATH)

    if len(loaded_crew.agents) != 4:
        raise ValueError(
            f"Expected 4 agents, found {len(loaded_crew.agents)}"
        )

    llm = LLM(
        model=OMNI_MODEL,
        provider="openai",
        api_key=OMNI_KEY,
        base_url=OMNI_BASE_URL,
        max_tokens=4096,
    )

    for agent in loaded_crew.agents:
        agent.llm = llm

    agents_by_role = {
        agent.role: agent for agent in loaded_crew.agents
    }

    return agents_by_role


# -----------------------------
# Stage 1: Query routing
# -----------------------------



def route_query(question, intent_summary, agents):
    router_role = "Research Question Answering and Reasoning Specialist"
    router = agents.get(router_role)

    if router is None:
        raise ValueError("Question Answerer agent was not found.")

    routing_task = Task(
        description="""
        Analyze the user's research-paper question.

        NORMALIZED USER QUESTION:
        {user_question}

        UNDERSTOOD USER INTENT:
        {intent_summary}

        Your responsibility is ONLY to select the single most relevant
        specialist agent.

        Available specialists:

        1. comprehension_mind:
           Concepts, terminology, prerequisites, equations,
           methodology, architecture, and how something works.

        2. research_exploration_mind:
           Research gaps, novelty, prior research directions,
           unresolved problems, assumptions, and future work.

        3. critical_evaluation_mind:
           Reliability, experimental evidence, datasets, baselines,
           metrics, reproducibility, weaknesses, and limitations.

        Use the normalized question as the primary source.
        Use the intent summary as additional context.

        Select exactly ONE specialist.

        Do not answer the user's question.
        Do not analyze the PDF.
        Do not select yourself.

        Return ONLY valid JSON in this exact format:

        {
            "selected_agent": "comprehension_mind",
            "reason": "Short explanation for this selection."
        }
        """,
        expected_output=(
            "A valid JSON object containing selected_agent and reason."
        ),
        agent=router,
    )

    router_crew = Crew(
        agents=[router],
        tasks=[routing_task],
        process=Process.sequential,
        verbose=True,
    )
    router_crew.max_rpm = 3

    result = router_crew.kickoff(
        inputs={
            "user_question": question,
            "intent_summary": intent_summary,
        }
    )

    raw = str(result.raw).strip()

    start = raw.find("{")
    end = raw.rfind("}")

    if start == -1 or end == -1:
        raise ValueError(f"Router did not return valid JSON:\n{raw}")

    decision = json.loads(raw[start:end + 1])

    selected_key = decision.get("selected_agent")
    reason = decision.get("reason", "")

    if selected_key not in AGENT_ROLES:
        raise ValueError(
            f"Router selected an invalid agent: {selected_key}"
        )

    selected_role = AGENT_ROLES[selected_key]
    selected_agent = agents.get(selected_role)

    if selected_agent is None:
        raise ValueError(f"Agent not found: {selected_role}")

    return selected_key, reason, selected_agent, raw


# -----------------------------
# Stage 2: Specialist execution
# -----------------------------

def execute_specialist(agent, paper_path, question):
    answer_task = Task(
        description="""
        Answer the user's question using your specialist expertise.

        Original research paper:
        {paper_path}

        User question:
        {user_question}

        Read the original paper and examine relevant evidence.

        Requirements:
        - Answer the actual question directly.
        - Use evidence from the original paper.
        - Include page references wherever available.
        - Distinguish paper statements from your own interpretation.
        - Never invent evidence or unsupported claims.
        - State clearly if the paper lacks sufficient information.
        """,
        expected_output=(
            "A direct, evidence-grounded specialist answer with "
            "supporting page references and relevant reasoning."
        ),
        agent=agent,
    )

    specialist_crew = Crew(
        agents=[agent],
        tasks=[answer_task],
        process=Process.sequential,
        verbose=True,
    )
    specialist_crew.max_rpm = 3

    return specialist_crew.kickoff(
        inputs={
            "paper_path": str(paper_path),
            "user_question": question,
        }
    )


# -----------------------------
# Save results
# -----------------------------

def save_outputs(route_raw, selected_key, reason, answer):
    OUTPUT_DIR.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"research_run_{timestamp}.txt"

    with output_file.open("w", encoding="utf-8") as file:
        file.write("QUERY ROUTING\n")
        file.write("=" * 70 + "\n")
        file.write(route_raw + "\n\n")

        file.write(f"SELECTED AGENT: {selected_key}\n")
        file.write(f"REASON: {reason}\n\n")

        file.write("SPECIALIST ANSWER\n")
        file.write("=" * 70 + "\n")
        file.write(str(answer.raw))

    return output_file


# -----------------------------
# Main
# -----------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Query-routed research paper auditor"
    )

    parser.add_argument("--pdf", required=True)
    parser.add_argument("--question", required=True)

    args = parser.parse_args()

    pdf_path = Path(args.pdf).resolve()

    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Stage 0: Understand and normalize the user's question
    original_question = args.question

    print("\nStage 0: Groq is understanding the user's query...")

    normalized = normalize_user_query(original_question)

    question = normalized["normalized_question"]
    intent_summary = normalized["intent_summary"]

    print("\n--- QUERY NORMALIZATION ---")
    print("Original:", original_question)
    print("Normalized:", question)
    print("Intent:", intent_summary)

    if normalized["needs_clarification"]:
        print("\nClarification required:")
        print(normalized["clarification_question"])
        return

    # Stage 1: Load existing CrewAI agents
    print("\nLoading four agents...")
    agents = build_agents()

    print("\nStage 1: Question Answerer is analyzing the query...")

    selected_key, reason, selected_agent, route_raw = route_query(
        question,
        intent_summary,
        agents,
    )

    print("\n" + "=" * 60)
    print("ROUTING DECISION")
    print("=" * 60)
    print(f"Selected agent: {selected_key}")
    print(f"Reason: {reason}")

    # Stage 2: Execute selected specialist
    print("\nStage 2: Executing selected specialist...")

    answer = execute_specialist(
        selected_agent,
        pdf_path,
        question,
    )

    print("\n" + "=" * 60)
    print("FINAL SPECIALIST ANSWER")
    print("=" * 60)
    print(answer.raw)

    output_file = save_outputs(
        route_raw,
        selected_key,
        reason,
        answer,
    )

    print(f"\nOutput saved to: {output_file}")


if __name__ == "__main__":
    main()



   
