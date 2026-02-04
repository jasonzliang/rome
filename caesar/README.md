# Caesar: Autonomous Web Exploration Agent

Caesar is an LLM-powered autonomous agent that explores the web to discover, synthesize, and generate novel insights. It uses a **Perceive-Think-Act** loop to navigate web pages, extract knowledge, build a knowledge graph, and produce synthesis artifacts.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Exploration Modes](#exploration-modes)
- [Synthesis & Artifacts](#synthesis--artifacts)
- [Output Files](#output-files)
- [Example Configs](#example-configs)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)

## Overview

Caesar operates as an "Insight Hunter" rather than a traditional search engine. Its core philosophy:

- **Break information filter bubbles** - Explore beyond obvious results
- **Recursive curiosity** - Follow unexpected connections
- **Find novel connections** - Discover relationships across domains
- **Stochastic drifting** - Embrace controlled randomness in exploration
- **Cross-domain synthesis** - Combine insights from disparate sources

### Key Features

- **Autonomous web exploration** with LLM-guided link selection
- **Knowledge graph construction** tracking exploration paths
- **Vector database** for semantic insight storage and retrieval
- **Brave Search integration** for web search during exploration
- **Multi-draft artifact synthesis** with citation support
- **Checkpoint/resume** support for long explorations
- **Configurable role adaptation** based on exploration content

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         CaesarAgent                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐        │
│  │   PERCEIVE   │ → │    THINK     │ → │     ACT      │        │
│  │              │   │              │   │              │        │
│  │ Fetch HTML   │   │ Analyze with │   │ Select next  │        │
│  │ Extract text │   │ LLM          │   │ link         │        │
│  │ Parse links  │   │ Extract      │   │ Navigate or  │        │
│  │              │   │ insights     │   │ backtrack    │        │
│  └──────────────┘   └──────────────┘   └──────────────┘        │
│         │                  │                  │                 │
│         └──────────────────┼──────────────────┘                 │
│                            ▼                                    │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    Knowledge Base                           ││
│  │  ┌─────────────────┐  ┌─────────────────┐                  ││
│  │  │  Vector Store   │  │ Knowledge Graph │                  ││
│  │  │  (ChromaDB)     │  │  (NetworkX)     │                  ││
│  │  └─────────────────┘  └─────────────────┘                  ││
│  └─────────────────────────────────────────────────────────────┘│
│                            │                                    │
│                            ▼                                    │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                 ArtifactSynthesizer                         ││
│  │  Generate Q&A pairs → Build citations → Synthesize artifact ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites

- Python 3.10+
- The `rome` parent framework (should be in parent directory)

### Install Dependencies

```bash
cd /path/to/rome
pip install -r requirements.txt
```

### Environment Variables

Set the following environment variables:

```bash
# Required
export OPENAI_API_KEY="your-openai-api-key"

# Required for web search mode
export BRAVE_API_KEY="your-brave-search-api-key"

# Optional (for LLM-as-judge evaluation)
export ANTHROPIC_API_KEY="your-anthropic-api-key"
export GOOGLE_API_KEY="your-google-api-key"
```

### Optional: Graph Visualization

For exploration graph visualization, install pygraphviz:

```bash
# macOS
brew install graphviz
pip install pygraphviz

# Ubuntu/Debian
sudo apt-get install graphviz graphviz-dev
pip install pygraphviz
```

## Quick Start

### Basic Usage

Run from the **`caesar/` directory** (config paths are relative to this location):

```bash
cd /path/to/caesar
python run_agent.py <repository_path> <config_path> [--max-iterations N]
```

**Arguments:**
- `repository_path` - Directory to store exploration results (created if needed)
- `config_path` - Path to YAML configuration file (relative to caesar/)
- `--max-iterations` - Override max iterations from config (optional)

### Example: Simple Test Run

```bash
cd /path/to/caesar

# Quick test run (5 pages)
python run_agent.py ./results/test config/single_agent_test.yaml
```

### Example: Deep Web Search

```bash
cd /path/to/caesar

# Creative exploration starting from web search
python run_agent.py ./results/creativity config/creative/openended_creativity.yaml
```

## Configuration

Configuration files are YAML documents that override default settings. Only specify settings that differ from defaults.

### Configuration Structure

```yaml
# Agent identity
Agent:
  name: CaesarExplorer

# Exploration settings
CaesarAgent:
  starting_url: "https://example.com"     # OR
  starting_query: "Your search query"     # (mutually exclusive)
  max_iterations: 100
  max_depth: 10000
  # ... more settings

# Synthesis settings
ArtifactSynthesizer:
  synthesis_iterations: 20
  synthesis_drafts: 1
  # ... more settings

# LLM configuration
OpenAIHandler:
  model: gpt-5.2
  cost_limit: 50.0
  temperature: 0.1

# Memory/knowledge base
AgentMemory:
  enabled: true
  use_graph: true

# Logging
Logger:
  level: "info"
```

For a full configurations used to run actual experiments, see YAML files under config/creative

### Key Configuration Parameters

#### CaesarAgent Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `starting_url` | None | Initial URL to begin exploration |
| `starting_query` | None | Initial search query (alternative to URL) |
| `additional_starting_queries` | 0 | Generate N related queries from initial query |
| `max_iterations` | 10 | Total pages to explore |
| `max_depth` | 10000 | Max tree depth before backtracking |
| `max_web_searches` | 0 | Web searches allowed during exploration |
| `allowed_domains` | [] | Domain whitelist; empty = starting domain only; `["*"]` = all |
| `max_allowed_revisits` | 20 | Max times to revisit same page |
| `checkpoint_interval` | 1 | Save checkpoint every N iterations |
| `save_graph_interval` | 1 | Save graph every N iterations |
| `draw_graph` | false | Generate PNG visualizations (requires pygraphviz) |
| `adapt_role` | false | Adapt agent role based on content |
| `overwrite_role_file` | None | Path to custom role definition |
| `exploration_llm_config.model` | gpt-5.2 | Model for exploration decisions |
| `exploration_llm_config.temperature` | 0.9 | Temperature for exploration (higher = more creative) |

#### ArtifactSynthesizer Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `synthesis_classic_mode` | false | Ask all queries at once (vs iterative) |
| `synthesis_drafts` | 1 | Number of synthesis drafts |
| `synthesis_iterations` | 20 | Q&A iterations per draft |
| `synthesis_max_length` | None | Max words for artifact (None = unlimited) |
| `synthesis_merge_artifacts` | false | Merge multiple drafts into one |
| `synthesis_eli5` | false | Generate "Explain Like I'm 5" summary |
| `synthesis_eli5_length` | None | Max words for ELI5 summary |
| `synthesis_iteration_filter` | None | Only use insights up to iteration N |

#### OpenAIHandler Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | gpt-5.2 | LLM model for synthesis |
| `cost_limit` | None | Max API cost in USD |
| `temperature` | 0.1 | Temperature for synthesis (lower = focused) |
| `timeout` | 300 | API timeout in seconds |
| `reasoning_effort` | medium | For reasoning models (low/medium/high) |

## Exploration Modes

### 1. URL-Based Exploration

Start from a specific URL and explore linked pages:

```yaml
CaesarAgent:
  starting_url: "https://en.wikipedia.org/wiki/Artificial_intelligence"
  allowed_domains: []  # Stay on starting domain
```

### 2. Query-Based Exploration (Web Search)

Start from a web search query:

```yaml
CaesarAgent:
  starting_query: "Novel approaches to solving ARC-AGI benchmark"
  additional_starting_queries: 5  # Generate 5 related queries
  max_web_searches: 30            # Allow 30 searches during exploration
```

### 3. Domain-Restricted Exploration

Limit exploration to specific domains:

```yaml
CaesarAgent:
  starting_url: "https://arxiv.org/abs/1234.5678"
  allowed_domains: ["arxiv.org", "openreview.net"]
```

### 4. Open Exploration

Allow exploration across any domain:

```yaml
CaesarAgent:
  starting_query: "Cross-domain creativity research"
  allowed_domains: ["*"]  # Allow all domains
```

## Synthesis & Artifacts

After exploration, Caesar synthesizes collected insights into artifacts.

### Synthesis Modes

**Iterative Mode** (default): Progressively builds understanding through Q&A:
```yaml
ArtifactSynthesizer:
  synthesis_classic_mode: false
  synthesis_iterations: 25
```

**Classic Mode**: Asks all queries at once:
```yaml
ArtifactSynthesizer:
  synthesis_classic_mode: true
```

### Multi-Draft Synthesis

Generate multiple drafts and merge:

```yaml
ArtifactSynthesizer:
  synthesis_drafts: 3
  synthesis_merge_artifacts: true
  synthesis_eli5: true
  synthesis_eli5_length: 500
```

### Output Format

Synthesis artifacts include:
- **Abstract**: 80-120 word summary
- **Main Content**: Synthesized insights with citations [1], [2], etc.
- **Sources**: Full citation list with URLs
- **ELI5** (optional): Simplified explanation

## Output Files

All outputs are saved to the repository directory:

```
repository/
├── {agent_id}.checkpoint.json      # Resumable state
├── {agent_id}.graph_iter{N}.json   # Knowledge graph at iteration N
├── {agent_id}.graph_iter{N}.png    # Graph visualization (if enabled)
├── {agent_id}.synthesis.{timestamp}/
│   ├── synthesis-1.txt             # Draft 1
│   ├── synth-eli5-1.txt            # Draft 1 ELI5 (if enabled)
│   ├── synthesis-2.txt             # Draft 2 (if multi-draft)
│   └── merged-final.txt            # Merged artifact (if enabled)
└── search_result/                   # Cached web search results
    └── {query}_{hash}.html
```

## Example Configs

### Wikipedia Exploration

Located in `config/wikipedia/`:
- `single_agent_ai.yaml` - Explore AI from Wikipedia
- `single_agent_evo.yaml` - Explore Evolution
- `single_agent_sci.yaml` - Explore Science

### Creative Exploration

Located in `config/creative/`:
- `openended_creativity.yaml` - Open-ended creative exploration
- `crossdomain_synthesis.yaml` - Cross-domain knowledge synthesis
- `counterfactual_reasoning.yaml` - Counterfactual reasoning tasks
- `constrained_creativity.yaml` - Constrained creative challenges
- `meta_creativity.yaml` - Meta-level creativity

### AI Research

Located in `config/creative_ai/`:
- `solve_arcagi.yaml` - Explore ARC-AGI benchmark solutions
- `improve_transformer.yaml` - Transformer architecture improvements
- `intelligence_bottleneck.yaml` - Intelligence bottleneck research

**Note:** These configs use outdated parameter names (`synthesis_rounds`, `synthesis_max_tokens`, `synthesis_eli5_tokens`). Update them to use `synthesis_drafts`, `synthesis_max_length`, and `synthesis_eli5_length` respectively.

### Test Configuration

`config/single_agent_test.yaml` - Quick 5-iteration test run

## Advanced Usage

### Resume from Checkpoint

Exploration automatically resumes from the last checkpoint if one exists in the repository directory. To start fresh, delete or rename the checkpoint file.

### Custom Role Definition

Create a custom role file and reference it (paths are relative to working directory):

```yaml
CaesarAgent:
  overwrite_role_file: config/role/my_custom_role.txt
```

Example role file (`config/role/self_transcend_role_v1.txt`):
```
Your role: You are an agent of recursive self-transcendence...
Your approach:
- Impermanent Identity: Treat your current knowledge as a hypothesis...
- Creative Dissatisfaction: Actively seek limitations and paradoxes...
- Willful Becoming: Deliberately seek paths that resolve dissonance...
```

### Role Adaptation

Let the agent adapt its role based on discovered content:

```yaml
CaesarAgent:
  adapt_role: true
  adapt_role_file: config/role/self_transcend_insights_v1.txt
```

### Verbose Logging

```bash
cd /path/to/rome
export PYTHONUNBUFFERED=1
python run_agent.py ./repo config/single_agent_test.yaml 2>&1 | tee exploration.log
```

### Cost Management

Set a cost limit to prevent runaway API usage:

```yaml
OpenAIHandler:
  cost_limit: 50.0  # Stop at $50
```

The agent will stop gracefully when the limit is approached.

## Troubleshooting

### Common Issues

**"ModuleNotFoundError: No module named 'ecology'"**
- Fix the import in `run_agent.py` line 23
- Change `from ecology.caesar_agent` to `from caesar.caesar_agent`
- See [Quick Start](#quick-start) for details

**"No BRAVE_API_KEY found"**
- Set the environment variable: `export BRAVE_API_KEY="your-key"`
- Get a key from: https://brave.com/search/api/

**"Rate limit exceeded"**
- The agent has exponential backoff retry logic
- For persistent issues, reduce `max_web_searches` or add delays

**"Checkpoint not loading"**
- Ensure the repository path matches the original run
- Check file permissions on the checkpoint JSON

**"pygraphviz not found"**
- Install graphviz and pygraphviz (see Installation)
- Or disable visualization: `draw_graph: false`

**"Cost limit reached"**
- Increase `cost_limit` in config
- Or reduce `max_iterations` / `synthesis_iterations`

### Performance Tips

1. **Start small**: Use `config/single_agent_test.yaml` for testing
2. **Tune iterations**: More iterations = more insights but higher cost
3. **Use checkpoints**: Set `checkpoint_interval: 10` for long runs
4. **Domain restriction**: Limit `allowed_domains` to avoid tangents
5. **Model selection**: Use `gpt-4o` for cost efficiency, `gpt-5.2` for quality

## API Cost Estimates

Rough estimates (varies by content and settings):

| Configuration | ~Cost per 100 iterations |
|---------------|-------------------------|
| gpt-4o, basic synthesis | $2-5 |
| gpt-4o, multi-draft synthesis | $5-15 |
| gpt-5.2, basic synthesis | $10-30 |
| gpt-5.2, multi-draft synthesis | $30-80 |

Always set a `cost_limit` when experimenting!

## License

See parent rome repository for license information.

## Syncing repo

### To Update (Pull changes from Rome to Caesar)

```bash
git fetch rome
git subtree pull --prefix caesar rome main --squash

```

### To Push (Send changes from Caesar back to Rome)

```bash
git subtree push --prefix caesar rome main

```