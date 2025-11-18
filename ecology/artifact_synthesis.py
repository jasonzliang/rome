"""Caesar Synthesizer - Logic for synthesizing exploration artifacts"""
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from rome.config import set_attributes_from_config
from rome.logger import get_logger
from .caesar_config import (CAESAR_CONFIG, MAX_SYNTHESIS_QUERY_SOURCES,
                            MAX_SYNTHESIS_QA_CONTEXT)

class ArtifactSynthesizer:
    """Handles the synthesis of insights into final artifacts"""

    def __init__(self, agent, config: Dict = None):
        self.agent = agent
        self.logger = get_logger()

        # Load settings from config or defaults
        self.config = config or {}
        # We allow attributes to be set on 'self' from the config provided
        set_attributes_from_config(self, self.config, CAESAR_CONFIG['ArtifactSynthesizer'].keys())

        # Shortcuts to agent resources
        self.kb_manager = self.agent.kb_manager

    def synthesize_artifact(self, num_rounds: int = None) -> Dict[str, str]:
        """Generate final synthesis with optional multi-round refinement"""
        if not num_rounds: num_rounds = self.synthesis_rounds

        mode = f"iterative (n={self.synthesis_iterations})" if self.iterative_synthesis else "classic"
        self.logger.info(f"[SYNTHESIS] Using {mode} mode with {num_rounds} round(s)")

        if self.kb_manager.size() == 0:
            return {"abstract": "", "artifact": "No insights collected during exploration."}

        # Multi-round synthesis loop
        current_query = self.agent.starting_query
        all_rounds = []; previous_artifact = None

        for round_num in range(1, num_rounds + 1):
            self.logger.info(f"\n{'='*60}\n[SYNTHESIS ROUND {round_num}/{num_rounds}]\n{'='*60}")
            if round_num > 1 and current_query:
                self.logger.info(f"Refined query: {current_query}")

            # Generate synthesis for current round (with previous artifact context)
            result = self._synthesize_single_round(mode, current_query, previous_artifact)
            all_rounds.append(result)

            # Save current result as previous for next round
            previous_artifact = result

            # Refine query for next round (if not last round)
            if round_num < num_rounds:
                current_query = self._refine_query(current_query, result)
                if not current_query:
                    self.logger.error(f"Query refinement failed, stopping at round {round_num}")
                    break

        # Return final round's result with all rounds metadata
        final_result = all_rounds[-1]
        final_result["metadata"]["total_rounds"] = len(all_rounds)
        self._save_synthesis_outputs(final_result)

        return final_result

    def _synthesize_single_round(self, mode: str, current_query: Optional[str] = None,
                                 previous_artifact: Optional[Dict] = None) -> Dict[str, str]:
        """Execute a single synthesis round with optional query and previous artifact"""

        # Temporarily override starting_query for this round
        original_query = self.agent.starting_query
        if current_query:
            self.agent.starting_query = current_query

        qa_pairs = self._generate_qa_pairs(mode)

        # Restore original query
        self.agent.starting_query = original_query

        if not qa_pairs:
            return {"abstract": "", "artifact": "Unable to generate synthesis questions."}
        qa_list, source_list, source_map = self._build_answers_with_citations(qa_pairs)

        starting_query_task = f" that creatively answers this query: {self.agent.starting_query}" if self.agent.starting_query else ":"
        starting_query_role = f" and on how to creatively answer the query!" if self.agent.starting_query else "!"

        # Build context from previous artifact if available
        previous_context = ""
        if previous_artifact and previous_artifact.get("artifact"):
            previous_context = f"""
PREVIOUS ARTIFACT:
{previous_artifact["artifact"]}
--- END OF ARTIFACT ---
"""

        prompt = f"""You explored {len(self.agent.visited_urls)} sources and gathered {self.kb_manager.size()} insights.

KEY INSIGHTS (with source citations):
{qa_list}
--- END OF INSIGHTS ---

SOURCES:
{source_list}
--- END OF SOURCES ---
{previous_context}
YOUR TASK:
Drawing heavily upon the patterns that emerged from the key insights{', and building upon the previous artifact,' if previous_artifact else ''} create a novel, exciting, and thought provoking artifact{starting_query_task}

1. **Artifact Abstract** (100-150 tokens):
    - Summary of the artifact's core discovery and its significance
    - Include source citations [n] for key claims

2. **Artifact Main Text** (around {self.synthesis_max_tokens} tokens):
    - Some general suggests for artifact:
        a. Emergent patterns not visible in individual sources
        b. Novel discoveries, connections, or applications
        c. Surprising new directions or perspectives
        d. Interesting tensions, contradictions, or open questions
    - Cite sources using [n] notation after relevant claims (e.g., "This pattern emerged [1,3]")
    - Use one or more citations if necessary to support complex arguments
    {'- Build upon and extend the previous artifact, avoid repeating the same points - instead, deepen, extend, or challenge the previous artifact.' if previous_artifact else ''}

IMPORTANT: Avoid excessive jargon while keeping it logical, easy to understand, and convincing to a skeptical reader
IMPORTANT: Cite sources to support your claims and insights, but do NOT recreate the "Sources" list or provide a "References" section
IMPORTANT: Use your role as a guide on how to respond{starting_query_role}

Respond with valid JSON only:
{{
    "abstract": "<abstract text>",
    "artifact": "<artifact text>"
}}"""

        try:
            response = self.agent.chat_completion(prompt, response_format={"type": "json_object"})
            result = self.agent.parse_json_response(response)
            if not result or "abstract" not in result or "artifact" not in result:
                raise ValueError("Missing required keys in response")
        except Exception as e:
            self.logger.error(f"Synthesis generation failed: {e}")
            return {"abstract": "Synthesis failed.", "artifact": f"Error: {e}"}

        result["sources"] = dict(sorted(source_map.items(), key=lambda x: x[1]))
        result["metadata"] = {
            "pages_visited": len(self.agent.visited_urls),
            "insights_collected": self.kb_manager.size(),
            "sources_cited": len(source_map),
            "synthesis_mode": mode,
            "synthesis_queries": len(qa_pairs),
            "max_depth": self.agent.current_depth,
            "starting_url": self.agent.starting_url,
            "starting_query": self.agent.starting_query,
        }

        return result

    def _refine_query(self, current_query: Optional[str], artifact_result: Dict) -> Optional[str]:
        """Refine the synthesis query based on previous artifact"""
        artifact_text = artifact_result.get("artifact", "")

        if not artifact_text:
            self.logger.error("Cannot refine query: no artifact text")
            return None

        query_context = f"PREVIOUS QUERY: {current_query}\n\n" if current_query else ""

        prompt = f"""{query_context}PREVIOUS ARTIFACT:
{artifact_text}

YOUR TASK:
Based on the previous query and artifact above, identify the most promising direction for deeper exploration. What NEW question or angle would:
- Build on the insights already discovered
- Explore gaps, contradictions, or unexplored connections
- Lead to novel perspectives or applications
- Go deeper rather than broader

The refined query should be concise (1-2 sentences) and actionable for knowledge base retrieval.

IMPORTANT: Use your role as a guide on how to respond!

Respond with JSON:
{{
    "refined_query": "<your refined exploration query>",
    "reason": "<brief explanation of why this direction is promising>"
}}"""

        try:
            response = self.agent.chat_completion(prompt, response_format={"type": "json_object"})
            result = self.agent.parse_json_response(response)
            if result and "refined_query" in result:
                refined = result["refined_query"]
                reason = result.get("reason", "No reason provided")
                self.logger.info(f"[QUERY REFINEMENT] {refined}\nReason: {reason}")
                return refined
        except Exception as e:
            self.logger.error(f"Query refinement failed: {e}")

        return None


    def _generate_qa_pairs(self, mode: str) -> List[Tuple[str, str, List[Dict]]]:
        """Generate Q&A pairs with sources"""
        queries = [
            "What are the central themes and patterns discovered?",
            "What unexpected connections or insights emerged?",
            "What contradictions or tensions were revealed?",
            "What questions remain open or were raised?",
            "What novel perspective emerged from this exploration?"
        ]
        if self.agent.starting_query:
            queries = [self.agent.starting_query] + queries

        if mode == "classic":
            return self._generate_qa_pairs_classic(queries)

        # Iterative mode
        queries, answers, sources_list = [queries[0]], [], []

        for i in range(self.synthesis_iterations):
            answer, sources = self.kb_manager.query(
                queries[-1], top_k=self.synthesis_top_k, top_n=self.synthesis_top_n,
                return_sources=True)
            if not answer: break

            answers.append(answer)
            sources_list.append(sources)

            self.logger.info(f"[SYNTHESIS {i+1}/{self.synthesis_iterations}]\nQ: {queries[-1]}\nA: {answers[-1]}")

            if i < self.synthesis_iterations - 1:
                if not (next_query := self._generate_next_query(queries, answers)):
                    break
                queries.append(next_query)

        return list(zip(queries, answers, sources_list))

    def _generate_qa_pairs_classic(self, queries: List[str]) -> List[Tuple[str, str, List[Dict]]]:
        """Execute queries against KB with sources"""
        qa_pairs = []
        for query in queries:
            try:
                answer, sources = self.kb_manager.query(
                    query, top_k=self.synthesis_top_k, top_n=self.synthesis_top_n,
                    return_sources=True)
                if answer:
                    qa_pairs.append((query, answer, sources))
            except Exception as e:
                self.logger.error(f"KB query failed for '{query}': {e}")
        return qa_pairs

    def _generate_next_query(self, previous_queries: List[str],
                             previous_responses: List[str]) -> Optional[str]:
        """Generate next synthesis query"""
        recent_queries = previous_queries[-MAX_SYNTHESIS_QA_CONTEXT:]
        recent_responses = previous_responses[-MAX_SYNTHESIS_QA_CONTEXT:]

        context = "\n\n".join([f"Q: {q}\nA: {r}"
                              for q, r in zip(recent_queries, recent_responses)])

        prompt = f"""PREVIOUS INSIGHTS:
{context}

YOUR TASK:
Based on the insights gathered so far, what is the next most important question to ask
to deepen understanding and reveal emergent patterns? The question should:
- Build on previous insights rather than repeat them
- Seek connections between different themes
- Identify gaps or contradictions to explore
- Move toward synthesis and creation rather than enumeration

IMPORTANT: Use your role as a guide on how to respond!

Respond with JSON:
{{
    "query": "<your next question>",
    "reason": "<brief explanation of why this question deepens understanding>"
}}"""

        try:
            response = self.agent.chat_completion(prompt, response_format={"type": "json_object"})
            result = self.agent.parse_json_response(response)
            if result and "query" in result:
                return result["query"]
        except Exception as e:
            self.logger.error(f"Query generation failed: {e}")
        return None

    def _build_answers_with_citations(self, qa_pairs):
        """Build formatted answers with citations and source index from Q&A pairs."""
        source_map = {}; answers = []

        for i, (q, a, sources) in enumerate(qa_pairs):
            # Add new sources to index (exclude file:// URLs)
            for src in sources:
                if (url := src['url']) and not url.startswith('file://') and url not in source_map:
                    source_map[url] = len(source_map) + 1

            # Format with citations (only for URLs in source_map)
            refs = ", ".join([f"[{source_map[s['url']]}]"
                for s in sources[:MAX_SYNTHESIS_QUERY_SOURCES] if s.get('url') and s['url'] in source_map])
            answers.append(f"({i+1}) Question: {q}\n\nAnswer: {a} {refs}")

        qa_list = "\n\n\n".join(answers)
        source_list = "\n".join([f"[{idx}] {url}"
            for url, idx in sorted(source_map.items(), key=lambda x: x[1])])

        return qa_list, source_list, source_map

    def _save_synthesis_outputs(self, result: Dict) -> None:
        """Save synthesis with sources in JSON and text formats"""
        timestamp = datetime.now().strftime("%m%d%H%M")
        base_path = os.path.join(self.agent.get_repo(), f"{self.agent.get_id()}.synthesis.{timestamp}")

        try:
            with open(f"{base_path}.json", 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=4, ensure_ascii=False)

            with open(f"{base_path}.txt", 'w', encoding='utf-8') as f:
                f.write(f"ABSTRACT:\n{result['abstract']}\n\n")
                f.write(f"ARTIFACT:\n{result['artifact']}\n\n")
                if sources := result.get('sources'):
                    f.write("SOURCES:\n")
                    for url, idx in sorted(sources.items(), key=lambda x: x[1]):
                        f.write(f"[{idx}] {url}\n")
                    f.write("\n")
                f.write(f"METADATA:\n{json.dumps(result['metadata'], indent=4)}")
        except Exception as e:
            self.logger.error(f"Failed to save synthesis: {e}")