<p align="center">
  <img src="https://img.freepik.com/premium-photo/ancient-rome-sunset-scenery-buildings-ruins-summer-beautiful-sunny-panorama-historical-city-houses-sun-sky-concept-roman-empire-antique-painting-background_788189-12960.jpg" alt="Rome" width="500"/>
</p>

# Rome

Rome is a foundational architecture for verifiable agentic reasoning and autonomous scientific discovery. It provides the stateful environment, episodic memory, and FSM-based scaffolding necessary to deploy Generator-Verifier loops and facilitate deep, multi-step inference. It serves as the infrastructure for the following active research projects:

## Projects

### [Caesar](caesar/README.md) — Deep Agentic Web Exploration for Creative Answer Synthesis

**Paper Link:** [Caesar: Deep Agentic Web Exploration for Creative Answer Synthesis](caesar/paper/caesar.pdf)

Current LLM research agents optimize for retrieval precision over a flat sequence of documents. While effective at synthesizing known information, they struggle to advance from passive retrieval to the active discovery of new ideas. They suffer from "navigational amnesia" and frequently fall into local minima, looping through redundant content to produce derivative, consensus-driven outputs. 

Project Caesar bridges this gap by shifting the objective from mechanical retrieval to graph-based discovery. By constructing a dynamic knowledge graph during web traversal, Caesar automates the rigorous combinatorial process required for "System 2" hypothesis generation—discovering statistically significant, non-obvious connections across disparate domains. 

<p align="center">
  <img src="caesar/paper/caesar.png" alt="Caesar Architecture" width="700"/>
</p>

Caesar operates in two primary cognitive phases:

1. **Deep Web Exploration (Stateful Traversal):** A recursive Perceive-Think-Act loop performs a topological traversal of information spaces. Rather than isolating summaries, Caesar generates context-aware insights conditioned on the local structure of the exploration graph—analyzing how new empirical content builds upon or contradicts neighboring nodes (Think). It uses a dynamic exploration policy informed by a vector knowledge base and episodic memory to autonomously switch between depth-first expansion, strategic backtracking, and targeted search to maximize information gain (Act).

2. **Adversarial Artifact Synthesis (Generator-Verifier Loop):** Rather than producing a single-pass summary, Caesar operates as a recursive self-correction environment. It performs rigorous insight discovery via iterative Q&A chains. Between draft generations, an independent adversarial module formulates orthogonal queries explicitly targeting logical weaknesses, missing citations, and contradictions in the current belief state. This active gap analysis forces the agent to explore directions that maximize empirical grounding before a generative merge consolidates the verified trajectories. 

**Architectural Innovations:**
* **Domain-Specific Role Adaptation:** The agent dynamically rewrites its own system prompt to adopt constraints tuned to the target domain, overriding the safety-biased, generic responses typical of RLHF models.
* **Graph-Augmented Insight Generation:** Information extraction is conditioned on the exploration graph neighborhood, enabling online associative reasoning that identifies verifiable connections relative to adjacent nodes.
* **Knowledge-Guided Exploration Policy:** A dynamic decision-making policy uses exploration context and episodic memory to detect navigational stagnation and force backtracking.
* **Adversarial Query Refinement:** Orthogonal queries designed to push the agent out of the basin of attraction of generic LLM consensus to discover novel, grounded facts.

**Evaluations:**
In a blinded LLM-as-a-Judge framework evaluating System 2 combinatorial reasoning, Caesar significantly outperformed the native deep research modes of state-of-the-art frontier models (Mann-Whitney U, p < 0.001). Ablation studies confirm that both deep topological exploration and the adversarial verifier loop are independently necessary: shallow retrieval cannot reach rare, long-tail insights, and single-pass synthesis cannot escape derivative hallucination.

### [Rome](rome/README.md) — Stateful Environment for Autonomous Reasoning

Rome is the core Finite State Machine (FSM) infrastructure that underpins these autonomous research loops. Moving beyond brittle, linear prompting pipelines, Rome provides the stateful scaffolding necessary to deploy rigorous Generator-Verifier-Reviser topologies. It handles multi-agent coordination, episodic memory management, dynamic policy routing, and verifiable code execution—serving as a high-fidelity environment for evaluating and scaling the reasoning capabilities of foundation models. See [Rome README](rome/README.md) for full architectural documentation.

## Getting Started

```bash
git clone [https://github.com/yourusername/rome.git](https://github.com/yourusername/rome.git)
cd rome
pip install -e .
export OPENAI_API_KEY=your_api_key_here
```

See individual project READMEs for detailed setup and usage instructions.

## License

See LICENSE file for details.
