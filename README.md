<p align="center">
  <img src="caesar/paper/caesar.png" alt="Caesar autonomous research agent architecture — Perceive-Think-Act exploration loop and Generator-Verifier adversarial synthesis" width="720"/>
</p>

<h1 align="center">Caesar — Autonomous AI Research Agent</h1>

<p align="center">
  <strong>Deep web exploration and creative answer synthesis — the open-source alternative to ChatGPT Deep Research and Perplexity</strong>
</p>

<p align="center">
  <a href="#quickstart"><img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-blue"></a>
  <a href="caesar/paper/caesar.pdf"><img alt="Paper" src="https://img.shields.io/badge/paper-PDF-red"></a>
  <a href="https://doi.org/10.13140/RG.2.2.15118.22088"><img alt="DOI" src="https://img.shields.io/badge/DOI-10.13140/RG.2.2.15118.22088-blue"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-green"></a>
  <a href="https://github.com/jasonzliang/caesar-agent/commits/main"><img alt="Last commit" src="https://img.shields.io/github/last-commit/jasonzliang/caesar-agent"></a>
</p>

**Caesar** is an autonomous AI research agent that navigates the web, reasons over a dynamic knowledge graph, and synthesizes novel, grounded answers. In blinded LLM-as-a-Judge creativity evaluations, Caesar scored **25.29 / 30** — beating the runner-up (Gemini 3 Deep Research, 22.27) by **3.02 points**, and doubling the score of GPT-5.2 Deep Research (15.40). Significant across every setting at **p < 0.001**.

If you're looking for an **agentic RAG system that goes beyond retrieval** — graph-based exploration, adversarial verification, and multi-draft synthesis — this is it.

## Quickstart

```bash
git clone https://github.com/jasonzliang/caesar-agent.git
cd caesar-agent
pip install -e .

export OPENAI_API_KEY=your_key
export BRAVE_API_KEY=your_key      # for web search
export ANTHROPIC_API_KEY=your_key  # optional, for Claude-based agents

cd caesar
python run_agent.py config/config_preset/regular.yaml -q "your research question"
```

Agent explores the web, builds a knowledge graph on disk, synthesizes a final artifact, and emits `<id>.experiment_summary.json` with tokens, cost, wall-time, pages visited, and paths to generated artifacts.

## What It's Good For

Caesar shines on **open-ended, creative, cross-disciplinary** research — problems where retrieval alone won't work:

- **Hypothesis generation** — novel cross-domain connections (e.g., bridging materials science and biology)
- **Literature synthesis** — graph-grounded review that spots tensions and gaps between papers
- **Competitive intelligence** — deep mapping of a technical or market landscape
- **Counterfactual & meta-creative reasoning** — "what if X was different?" style inquiry
- **Novel solution ideation** — e.g., ARC-AGI–style problem exploration

It's **not** the right tool for quick factual lookups or latency-sensitive apps — Caesar is designed for depth, not speed.

## Why Caesar?

Current deep-research agents (ChatGPT Deep Research, Perplexity, GPT Researcher, Gemini Deep Research) optimize for **retrieval precision over a flat sequence of documents**. They produce competent summaries but suffer from **navigational amnesia**, fall into local minima, and generate derivative, consensus-driven outputs.

Caesar is different:

| Capability | Caesar | ChatGPT Deep Research | Perplexity | GPT Researcher |
|---|:-:|:-:|:-:|:-:|
| Open source | ✅ | ❌ | ❌ | ✅ |
| Dynamic knowledge graph | ✅ | ❌ | ❌ | ❌ |
| Adversarial Generator–Verifier loop | ✅ | ❌ | ❌ | ❌ |
| Multi-draft synthesis with merge | ✅ | ❌ | ❌ | Partial¹ |
| Episodic memory + backtracking | ✅ | ❌ | ❌ | ❌ |
| Pluggable LLM backend (OpenAI / Anthropic / local) | ✅ | ❌ | ❌ | ✅ |
| Reproducible experiment JSON | ✅ | ❌ | ❌ | Partial² |

<sub>¹ GPT Researcher supports multi-draft generation but not adversarial self-critique or merge. ² GPT Researcher logs cost per run but not the full reproducibility bundle (wall-time, page-level sources, draft provenance).</sub>

## How It Works

Caesar operates in two cognitive phases:

### 1. Deep Web Exploration — stateful graph traversal

A recursive **Perceive–Think–Act** loop performs topological traversal of information spaces. Rather than isolating summaries, Caesar generates context-aware insights conditioned on the **local structure of the exploration graph** — analyzing how new content builds upon or contradicts neighboring nodes. A dynamic policy, informed by a vector knowledge base and episodic memory, autonomously switches between depth-first expansion, strategic backtracking, and targeted web search.

### 2. Adversarial Artifact Synthesis — Generator–Verifier loop

Rather than a single-pass summary, Caesar runs as a recursive self-correction environment. An independent adversarial module formulates **orthogonal queries** targeting logical weaknesses, missing citations, and contradictions in the current belief state. Multiple drafts are produced iteratively and merged — forcing the agent out of the consensus basin that traps single-pass LLMs.

## Architectural Innovations

- **Domain-Specific Role Adaptation** — the agent rewrites its own system prompt per task, overriding the safety-biased generic responses typical of RLHF models.
- **Graph-Augmented Insight Generation** — insights are conditioned on the exploration graph neighborhood, enabling online associative reasoning.
- **Knowledge-Guided Exploration Policy** — detects navigational stagnation via episodic memory and forces backtracking.
- **Adversarial Query Refinement** — orthogonal queries push the agent out of generic LLM consensus toward novel, grounded facts.

## Benchmark Results

Evaluated with a blinded **3-model LLM-as-a-Judge panel** (Claude Sonnet 4.5, GPT-5.2, Gemini 3 Pro) across three creativity dimensions — **New**, **Useful**, **Surprising** — scored 0–10 each:

| Agent | New | Useful | Surprising | **Total** |
|---|:-:|:-:|:-:|:-:|
| **Caesar** | **8.64** | **8.38** | **8.27** | **25.29** |
| Gemini 3 Deep Research | 7.69 | 7.09 | 7.49 | 22.27 |
| Sonnet 4.5 Deep Research | 6.96 | 7.20 | 6.73 | 20.89 |
| GPT-5.2 Deep Research | 5.02 | 6.02 | 4.36 | 15.40 |

Mann–Whitney U, **p < 0.001** across all settings. Ablations confirm both graph exploration and the adversarial verifier loop are independently necessary — see the [paper](caesar/paper/caesar.pdf) for full methodology, exploration-budget ablation, and judge bias analysis.

## Example Output

After a run, Caesar writes a full artifact (abstract + body with citations) plus a structured summary:

```json
{
  "wall_time": 591.31,
  "tokens_used": 109873,
  "token_cost": 0.29,
  "api_calls": 20,
  "webpages_visited": 4,
  "iterations_elapsed": 5,
  "artifact_dir": "result/.../agent_CaesarExplorer.synthesis.04161850",
  "num_drafts": 2,
  "config_summary": { "..." : "..." }
}
```

The `artifact_dir` contains one `.txt` per synthesis draft, a final merged artifact, and a metadata file tracking sources cited in each draft. Knowledge graphs are saved as compressed JSON checkpoints for reproducibility or post-hoc analysis.

## Built on Rome

Caesar is built on **Rome**, a Finite State Machine framework for stateful AI agents — providing Generator–Verifier–Reviser topologies, episodic memory, dynamic policy routing, and verifiable code execution. See the [Rome framework docs](rome/README.md) if you want to build your own agent on top.

## Project Layout

```
caesar-agent/
├── caesar/          # Caesar: the research agent (start here)
│   ├── caesar_agent.py
│   ├── artifact_synthesis.py
│   ├── run_agent.py
│   ├── config/      # YAML configs and creativity benchmarks
│   └── paper/       # Caesar paper (PDF)
└── rome/            # Rome framework: FSM, memory, LLM handlers, KB client
```

## FAQ

**How is this different from LangGraph / CrewAI / AutoGen?**
Those are orchestration frameworks — they help you wire up agents. Rome is an opinionated runtime for *how* agents should reason (graph-structured exploration, adversarial verification, episodic memory). Caesar is a concrete research agent built on it.

**Do I need GPUs?**
No. Caesar uses hosted LLM APIs (OpenAI, Anthropic). A local ChromaDB instance handles the vector store. Runs on a laptop.

**Which models are supported?**
OpenAI (GPT-5 family, o-series reasoning models), Anthropic (Claude 4.5 / 4.6 / 4.7), and any OpenAI-compatible endpoint. Model selection is per-subsystem (exploration, synthesis, judging) via YAML config.

**How much does a typical run cost?**
A 5-iteration exploration with Claude Haiku 4.5 runs at roughly $0.30 and 10 minutes. A 250-iteration deep run with GPT-5.4-mini is typically $5–$10.

**Can I reproduce the benchmarks?**
Yes — configs, judge rubrics, and evaluation scripts are in `caesar/config/` and `caesar/analysis/`.

## Contributing & Community

- ⭐ **Star the repo** if Caesar is useful for your research
- 💬 **[Open a Discussion](https://github.com/jasonzliang/caesar-agent/discussions)** for ideas, questions, or use cases
- 🐛 **[File an Issue](https://github.com/jasonzliang/caesar-agent/issues)** for bugs or feature requests
- 🔧 **PRs welcome** — especially new exploration policies, synthesis strategies, and benchmark domains

## Citation

If you use Caesar in your research, please cite:

```bibtex
@techreport{liang26caesar,
  title       = "Caesar: Deep Agentic Web Exploration for Creative Answer Synthesis",
  author      = "Jason Liang and Elliot Meyerson and Risto Miikkulainen",
  year        = 2026,
  month       = mar,
  institution = "Cognizant AI Lab",
  number      = "2026-02",
  url         = "https://www.researchgate.net/publication/402554537_Caesar_Deep_Agentic_Web_Exploration_for_Creative_Answer_Synthesis",
  doi         = "10.13140/RG.2.2.15118.22088"
}
```

## License

MIT — see [LICENSE](LICENSE).
