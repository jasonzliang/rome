# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is the `ecology` module within the `rome` framework - a Python library for building intelligent AI agents. The ecology module specifically contains **CaesarAgent**, a web exploration agent that uses graph-based navigation to gather and synthesize insights from web content.

## Project Structure

```
ecology/
├── caesar_agent.py       # Main CaesarAgent class - web exploration with perceive/think/act loop
├── caesar_config.py      # Configuration constants and default settings (CAESAR_CONFIG)
├── artifact_synthesis.py # ArtifactSynthesizer - generates final research artifacts from insights
├── brave_search.py       # BraveSearch - web search integration with retry logic
├── run_agent.py          # CLI runner for CaesarAgent
├── analysis/             # Evaluation tools (LLM-as-judge)
│   ├── llm_as_judge.py   # Multi-LLM judge (GPT, Claude, Gemini) for answer evaluation
│   └── judge_analysis.py # Analysis utilities for judge results
└── config/               # YAML configuration files
    ├── creative/         # Creativity benchmark configs
    └── llm_as_judge/     # Judge rubrics and templates
```

## Running the Agent

```bash
# Basic usage
python run_agent.py <repository_path> <config_path>

# With iteration override
python run_agent.py ./my_project config/creative/openended_creativity.yaml --max-iterations 50
```

## Running the LLM Judge

```bash
# Evaluate agent answers with all judges
python analysis/llm_as_judge.py ./experiments

# JSON output with 3 trials per judge
python analysis/llm_as_judge.py ./experiments -j -t 3

# Individual evaluation per answer, reasoning mode enabled
python analysis/llm_as_judge.py ./experiments -i -j -R

# Select specific judges
python analysis/llm_as_judge.py ./experiments -J claude gpt
```

## Architecture

### CaesarAgent Exploration Loop

The agent follows a three-phase perceive/think/act loop:

1. **Perceive** (`perceive()`): Fetches HTML/PDF content, extracts text and links from current URL
2. **Think** (`think()`): Analyzes content via LLM, generates insights, stores in knowledge base
3. **Act** (`act()`): Uses LLM to select next URL from available links based on exploration strategy

### Key Components

- **Knowledge Base**: Uses `ChromaClientManager` (ChromaDB + LlamaIndex) for storing/querying insights
- **Agent Memory**: Tracks navigation history via `AgentMemory` for pattern recognition
- **Graph Structure**: NetworkX DiGraph tracks visited URLs and relationships
- **Checkpointing**: Automatic save/resume via JSON checkpoints in log directory

### Synthesis Pipeline

`ArtifactSynthesizer` generates final artifacts through:
1. Iterative Q&A queries against the knowledge base
2. Multi-draft refinement with query evolution
3. Optional draft merging for comprehensive output
4. ELI5 post-processing for accessibility

## Configuration

Configurations use YAML and cascade: `DEFAULT_CONFIG` -> `CAESAR_CONFIG` -> custom config -> constructor args.

Key configuration sections:
- `CaesarAgent`: Exploration parameters (max_iterations, max_depth, allowed_domains)
- `ArtifactSynthesizer`: Synthesis settings (iterations, drafts, max_length)
- `OpenAIHandler`: LLM model and cost limits
- `AgentMemory`: Memory enable/disable and type (vector vs graph)
- `BraveSearch`: Search API settings

## Environment Variables

```bash
export OPENAI_API_KEY=...      # Required for LLM calls
export BRAVE_API_KEY=...       # Required for web search
export ANTHROPIC_API_KEY=...   # For Claude judge
export GOOGLE_API_KEY=...      # For Gemini judge
```

## Dependencies

Core dependencies are in the parent `rome/requirements.txt`. Caesar-specific:
- `curl_cffi`: HTTP requests with browser impersonation
- `beautifulsoup4`: HTML parsing
- `PyPDF2`: PDF text extraction
- `chromadb` + `llama-index`: Vector store knowledge base
- `networkx`: Graph-based URL tracking
- `pygraphviz`: Graph visualization (optional)

## Parent Module

The ecology module imports from the parent `rome` package:
- `rome.base_agent.BaseAgent`: Base class with LLM integration and logging
- `rome.config`: Configuration loading and merging utilities
- `rome.kb_client.ChromaClientManager`: Knowledge base client
- `rome.logger`: Logging infrastructure
