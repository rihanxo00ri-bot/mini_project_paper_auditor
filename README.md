# how to RUN 
if u use omniroute apikey so:
  1. type omniroute in first terminal 
  2. in other terminal write streamlit run test_crew2.py

# AI-Assisted Research Paper Auditing System

A multi-agent system for analyzing research papers using AI agents and Retrieval-Augmented Generation (RAG).

## Features

- Research-paper analysis using multiple AI agents
- RAG-based retrieval from documents
- Streamlit interface
- Separate scripts for different agent and RAG workflows

## Project Files

- `test_crew2.py` — RAG + multi-agent workflow
- `test_crew.py` — multi-agent workflow
- `main.py` — main Streamlit application and agent workflow
- `crew.jsonc` — CrewAI configuration
- `agents/` — agent-related code
- `tools/` — project tools
- `knowledge/` — knowledge resources
- `skills/` — project skills

## Requirements

- Python version compatible with `pyproject.toml`
- uv package manager
- Your own API keys for the services used by the application
- OmniRoute, if required by your selected workflow

## Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd paper_auditor