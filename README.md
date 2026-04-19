<p align="center">
  <img src="caesar/paper/caesar.png" alt="Caesar autonomous research agent architecture — Perceive-Think-Act exploration loop and Generator-Verifier adversarial synthesis" width="720"/>
</p>

<h1 align="center">Rome — Autonomous Research Agent Framework</h1>

<p align="center">
  <strong>Deep web exploration and creative answer synthesis that outperforms frontier deep-research agents</strong>
</p>

<p align="center">
  <a href="#quickstart"><img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-blue"></a>
  <a href="caesar/paper/caesar.pdf"><img alt="Paper" src="https://img.shields.io/badge/paper-PDF-red"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-green"></a>
  <a href="https://github.com/jasonzliang/caesar-agent/commits/main"><img alt="Commits" src="https://img.shields.io/github/last-commit/jasonzliang/caesar-agent"></a>
</p>

**Rome** is an open-source framework for building autonomous AI research agents — LLM agents that navigate the web, reason over a knowledge graph, and synthesize novel, grounded answers. Its flagship agent **Caesar** significantly outperforms the native deep-research modes of state-of-the-art frontier models on creative reasoning benchmarks (Mann–Whitney U, p < 0.001).

If you're looking for an agentic RAG system that goes beyond retrieval to actual discovery — graph-based exploration, adversarial verification, and multi-draft synthesis — this is it.

## Quickstart

```bash
git clone https://github.com/jasonzliang/caesar-agent.git
cd rome
pip install -e .

export OPENAI_API_KEY=your_key
export BRAVE_API_KEY=your_key      # for web search
export ANTHROPIC_API_KEY=your_key  # optional, for Claude-based agents

cd caesar
python run_agent.py config/config_preset/regular.yaml -q "your research question"
```

The agent explores the web, writes a knowledge graph to disk, synthesizes a final artifact, and emits `<id>.experiment_summary.json` with tokens used, wall time, pages visited, and paths to generated artifacts.

## Why Caesar?

Current deep-research agents (ChatGPT Deep Research, Perplexity, GPT Researcher, and similar) optimize for retrieval precision over a flat sequence of documents. They produce competent summaries but struggle to advance from passive retrieval to the active discovery of new ideas. They suffer from **navigational amnesia**, fall into local minima, and generate derivative, consensus-driven outputs.

Caesar is different:

| Capability | Caesar | ChatGPT Deep Research | Perplexity | GPT Researcher |
|---|:-:|:-:|:-:|:-:|
| Open source | ✅ | ❌ | ❌ | ✅ |
| Dynamic knowledge graph | ✅ | ❌ | ❌ | ❌ |
| Adversarial Generator–Verifier loop | ✅ | ❌ | ❌ | ❌ |
| Multi-draft synthesis with merge | ✅ | ❌ | ❌ | Partial |
| Episodic memory + backtracking | ✅ | ❌ | ❌ | ❌ |
| Pluggable LLM backend (OpenAI / Anthropic / local) | ✅ | ❌ | ❌ | ✅ |
| Reproducible experiment JSON | ✅ | ❌ | ❌ | Partial |

<p align="center">
  <img src="caesar/paper/caesar.png" alt="Caesar architecture diagram showing the Perceive-Think-Act exploration loop feeding a vector knowledge base, and the Generator-Verifier synthesis loop with adversarial query refinement" width="720"/>
</p>

## How It Works

Caesar operates in two cognitive phases:

### 1. Deep Web Exploration — stateful graph traversal

A recursive **Perceive–Think–Act** loop performs topological traversal of information spaces. Rather than isolating summaries, Caesar generates context-aware insights conditioned on the **local structure of the exploration graph** — analyzing how new content builds upon or contradicts neighboring nodes. A dynamic exploration policy, informed by a vector knowledge base and episodic memory, autonomously switches between depth-first expansion, strategic backtracking, and targeted web search to maximize information gain.

### 2. Adversarial Artifact Synthesis — Generator–Verifier loop

Rather than a single-pass summary, Caesar runs as a recursive self-correction environment. An independent adversarial module formulates **orthogonal queries** targeting logical weaknesses, missing citations, and contradictions in the current belief state. Multiple drafts are produced iteratively and merged, forcing the agent out of the consensus basin that traps single-pass LLMs.

## Architectural Innovations

- **Domain-Specific Role Adaptation** — the agent dynamically rewrites its own system prompt to adopt domain-tuned constraints, overriding the safety-biased, generic responses typical of RLHF models.
- **Graph-Augmented Insight Generation** — insights are conditioned on the exploration graph neighborhood, enabling online associative reasoning.
- **Knowledge-Guided Exploration Policy** — a decision policy uses exploration context and episodic memory to detect stagnation and force backtracking.
- **Adversarial Query Refinement** — orthogonal queries push the agent out of the basin of attraction of generic LLM consensus to discover novel, grounded facts.

## Benchmark Results

In a blinded **LLM-as-a-Judge** framework evaluating creative combinatorial reasoning, Caesar significantly outperformed the native deep-research modes of frontier models (Mann–Whitney U, **p < 0.001**). Ablation studies confirm that both deep topological exploration and the adversarial verifier loop are independently necessary: shallow retrieval cannot reach rare long-tail insights, and single-pass synthesis cannot escape derivative hallucination.

See the [paper](caesar/paper/caesar.pdf) for full methodology, metrics, and ablations.

## The Rome Framework

Rome is the **Finite State Machine infrastructure** underneath Caesar — the stateful scaffolding for Generator–Verifier–Reviser topologies, multi-agent coordination, episodic memory, dynamic policy routing, and verifiable code execution. Caesar is one of several research projects built on Rome. See the [Rome module README](rome/README.md) for full architectural documentation.

## Project Layout

```
rome/
├── caesar/          # Caesar: deep web exploration agent (this is where most users start)
│   ├── caesar_agent.py
│   ├── artifact_synthesis.py
│   ├── run_agent.py
│   ├── config/      # YAML configs and creativity benchmarks
│   └── paper/       # Caesar paper (PDF)
└── rome/            # Rome framework: FSM, memory, LLM handlers, knowledge base
```

## FAQ

**How is this different from LangGraph / CrewAI / AutoGen?**
Those are orchestration frameworks — they help you wire up agents. Rome is a runtime with an opinionated stance on *how* agents should reason (graph-structured exploration, adversarial verification, episodic memory). Caesar is a concrete research agent that demonstrates what the framework enables.

**Do I need GPUs?**
No. Caesar uses hosted LLM APIs (OpenAI, Anthropic). A local ChromaDB instance handles the vector store. Runs comfortably on a laptop.

**Which models are supported?**
OpenAI (GPT-5 family, o-series reasoning models), Anthropic (Claude 4.5, 4.6, 4.7), and any OpenAI-compatible endpoint. Model selection is per-subsystem (exploration, synthesis, judging) via YAML config.

**How much does a typical run cost?**
A 5-iteration exploration with Claude Haiku 4.5 runs at roughly $0.30 and 10 minutes. A 50-iteration deep run with GPT-5 is typically $5–15.

**Can I reproduce the benchmarks?**
Yes — configs, judge rubrics, and evaluation scripts are in `caesar/config/` and `caesar/analysis/`.

## Citation

If you use Rome or Caesar in your research, please cite:

```bibtex
@misc{caesar2026,
  title        = {Caesar: Deep Agentic Web Exploration for Creative Answer Synthesis},
  author       = {Liang, Jason},
  year         = {2026},
  howpublished = {\url{https://github.com/jasonzliang/caesar-agent}}
}
```

## License

See the [LICENSE](LICENSE) file.
