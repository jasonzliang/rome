<p align="center">
  <img src="https://img.freepik.com/premium-photo/ancient-rome-sunset-scenery-buildings-ruins-summer-beautiful-sunny-panorama-historical-city-houses-sun-sky-concept-roman-empire-antique-painting-background_788189-12960.jpg" alt="Rome" width="500"/>
</p>

# Rome

Rome is a Python framework for building intelligent, autonomous AI agents. It provides shared infrastructure — LLM integration, configuration, logging, knowledge base management, and memory — on top of which specialized agent modules are built.

## Projects

### [Caesar](caesar/README.md) — Deep Agentic Web Exploration for Creative Answer Synthesis

Caesar is an LLM-powered autonomous agent that explores the web to discover, synthesize, and generate novel insights. Unlike standard retrieval-augmented generation (RAG) or linear web search agents that treat the internet as a flat sequence of documents, Caesar constructs a dynamic knowledge graph during traversal and uses it to foster associative reasoning — enabling the discovery of non-obvious connections between disparate concepts.

<p align="center">
  <img src="caesar/paper/caesar.png" alt="Caesar Architecture" width="700"/>
</p>

Caesar operates in two phases:

1. **Deep Web Exploration** — A recursive **Perceive-Think-Act** loop navigates web pages guided by a context-aware policy. At each step the agent fetches and parses content (Perceive), generates graph-augmented insights by conditioning on the local topology of visited pages (Think), and selects the next link, backtrack, or web search using dual-context retrieval from both the knowledge graph and a vector store (Act). The agent autonomously switches between depth-first expansion and strategic backtracking to maximize information gain.

2. **Adversarial Artifact Synthesis** — Rather than simply summarizing collected insights, Caesar employs an adversarial refinement loop that actively seeks novel perspectives. It performs recursive insight discovery via iterative Q&A against the knowledge base, generates recurrent drafts with citations, and between drafts formulates adversarial queries that target narrative weaknesses and contradictions in the current draft. Multiple drafts are then merged through a generative unification step, and an optional ELI5 post-processing stage distills the result into accessible language.

Key innovations include **domain-specific role adaptation** (the agent rewrites its own system prompt to match the task), **graph-augmented insight generation** (insights are conditioned on the exploration graph neighborhood rather than generated in isolation), **knowledge-guided exploration** (a dynamic policy informed by both the knowledge base and episodic navigation memory), and **adversarial query refinement** (queries are designed to break out of the local minima of generic LLM outputs).

In evaluations using a blinded LLM-as-a-Judge framework across creativity benchmarks, Caesar significantly outperformed all baselines — including deep research modes of GPT-5.2, Claude Sonnet 4.5, and Gemini 3 Pro — scoring highest in Novelty, Usefulness, and Surprise metrics.

**Paper:** [Caesar: Deep Agentic Web Exploration for Creative Answer Synthesis (PDF)](caesar/paper/caesar.pdf)

### [Rome](rome/README.md) — FSM-Based AI Agent Framework

Rome is the core framework providing Finite State Machine (FSM) architecture for building AI agents that autonomously analyze, edit, test, and execute code. It includes configurable FSM workflows, multi-agent coordination, smart action selection strategies, and built-in benchmarking against coding datasets like HumanEval+. See the [Rome README](rome/README.md) for full documentation.

## Getting Started

```bash
git clone https://github.com/yourusername/rome.git
cd rome
pip install -e .
```

```bash
export OPENAI_API_KEY=your_api_key_here
```

See individual project READMEs for detailed setup and usage instructions.

## License

See LICENSE file for details.
