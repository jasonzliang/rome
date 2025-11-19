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
        num_rounds = max(num_rounds, 1)

        mode = f"iterative (n={self.synthesis_iterations})" if self.iterative_synthesis else "classic"
        self.logger.info(f"[SYNTHESIS] Using {mode} mode with {num_rounds} round(s)")

        if self.kb_manager.size() == 0:
            return {"abstract": "", "artifact": "No insights collected during exploration."}

        # Multi-round synthesis loop
        current_query = self.agent.starting_query
        all_rounds = []; previous_artifact = None
        base_dir = os.path.join(self.agent.get_repo(),
            f"{self.agent.get_id()}.synthesis.{datetime.now().strftime("%m%d%H%M")}")

        for round_num in range(1, num_rounds + 1):
            self.logger.info(f"\n{'='*80}\n[SYNTHESIS ROUND {round_num}/{num_rounds}]\n{'='*80}")
            if current_query: self.logger.info(f"Current query: {current_query}")

            # Generate synthesis for current round (with previous artifact context)
            result = self._synthesize_single_round(mode, current_query, previous_artifact)
            self._save_synthesis_outputs(result, base_dir=base_dir)
            all_rounds.append(result)

            # Save current result as previous for next round
            previous_artifact = result

            # Refine query for next round (if not last round)
            if round_num < num_rounds:
                current_query = self._refine_query(result)
                if not current_query:
                    self.logger.error(f"Query refinement failed, stopping at round {round_num}")
                    break

        # Merge artifacts if requested and multiple rounds exist
        if self.synthesis_merge_artifacts and len(all_rounds) > 1:
            self.logger.info(f"\n{'='*80}\n[MERGING {len(all_rounds)} ARTIFACTS]\n{'='*80}")
            final_result = self._merge_artifacts(all_rounds)
        else:
            final_result = all_rounds[-1]

        # final_result["metadata"]["total_rounds"] = len(all_rounds)
        self._save_synthesis_outputs(final_result, base_dir=base_dir)

        return final_result

    def _synthesize_single_round(self, mode: str, current_query: Optional[str] = None,
                                 previous_artifact: Optional[Dict] = None) -> Dict[str, str]:
        """Execute a single synthesis round with optional query and previous artifact"""

        # Temporarily override starting_query for this round
        qa_pairs = self._generate_qa_pairs(mode, current_query)

        if not qa_pairs:
            return {"abstract": "", "artifact": "Unable to generate synthesis questions."}
        qa_list, source_list, source_map = self._build_answers_with_citations(qa_pairs)

        query_context = f" that creatively answers this query: {self.agent.starting_query}" if self.agent.starting_query else ":"
        query_role = f" and on how to creatively answer the query!" if self.agent.starting_query else "!"

        # Build context from previous artifact if available
        previous_context = ""
        if self.synthesis_prev_artifact and previous_artifact and previous_artifact.get("artifact"):
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
Drawing heavily upon the patterns that emerged from the key insights{', and building upon the previous artifact,' if previous_artifact else ''} create a novel, exciting, and thought provoking artifact{query_context}

1. **Artifact Abstract** (100-150 tokens):
    - Summary of the artifact's core discovery and its significance

2. **Artifact Main Text** (around {self.synthesis_max_tokens} tokens):
    - Some general suggests for artifact:
        a. Emergent patterns not visible in individual sources
        b. Novel discoveries, connections, or applications
        c. Surprising new directions or perspectives
        d. Interesting tensions, contradictions, or open questions
    - Cite sources using [n] notation after relevant claims (e.g., "This pattern emerged [1,3]")
    - Use one or more citations if necessary to support complex arguments
    {'- Build upon the previous artifact by analyzing it for weaknesses in organization, arguments, or content, and then use key insights to deepen, improve, and extend the previous artifact\n' if previous_artifact else ''}

IMPORTANT: Avoid excessive jargon while keeping it logical, easy to understand, and convincing to a skeptical reader
IMPORTANT: Cite sources to support your claims and insights, but do NOT recreate the "Sources" list or provide a "References" section{', and do NOT mention or reference the previous artifact\n' if previous_artifact else ''}
IMPORTANT: Use your role as a guide on how to respond{query_role}

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

    def _refine_query(self, artifact_result: Dict, current_query: Optional[str] = None) -> Optional[str]:
        """Refine the synthesis query based on previous artifact"""

        artifact_text = artifact_result.get("artifact", "")
        if not artifact_text:
            self.logger.error("Cannot refine query: no artifact text")
            return None

        if not current_query: current_query = self.agent.starting_query
        query_context = f"PREVIOUS QUERY: {current_query}\n\n" if current_query else ""

        prompt = f"""{query_context}PREVIOUS ARTIFACT:
{artifact_text}
--- END OF ARTIFACT ---

YOUR TASK:
Based on the previous query and artifact above, identify the most promising direction for deeper exploration. What NEW question or angle would:
    - Build on the insights already discovered
    - Explore gaps, contradictions, or unexplored connections
    - Lead to novel perspectives or applications
    - Go deeper rather than broader

The refined query should be concise (1-2 sentences), straightforward, clear, and understandable.

IMPORTANT: Use your role as a guide on how to respond!

Respond with JSON:
{{
    "refined_query": "<your refined exploration query, posed as a question>",
    "reason": "<brief explanation of why the refined query improves upon the previous query>"
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

    def _merge_artifacts(self, all_rounds: List[Dict]) -> Dict[str, str]:
        """Merge artifacts from all rounds into a single comprehensive artifact"""
        # Build context from all rounds with their original sources
        artifacts_context = []

        for i, round_result in enumerate(all_rounds, 1):
            # Use original sources from this round
            round_sources = round_result.get('sources', {})

            # Build sources list for this specific round using original indices
            round_source_list = "\n".join([f"[{idx}] {url}"
                for url, idx in sorted(round_sources.items(), key=lambda x: x[1])])

            # Add artifact with its own sources section
            artifacts_context.append(
                f"ROUND {i} ARTIFACT:\n{round_result['artifact']}\n\n"
                f"ROUND {i} SOURCES:\n{round_source_list}\n--- END OF ROUND {i} ---"
            )

        artifacts_text = "\n\n" + "="*80 + "\n\n".join(artifacts_context)

        query_context = f" that creatively answers this query: {self.agent.starting_query}" if self.agent.starting_query else ""

        prompt = f"""You have explored {len(self.agent.visited_urls)} sources and synthesized insights across {len(all_rounds)} rounds of analysis.

LIST OF ARTIFACTS WITH SOURCES:
{artifacts_text}
--- END OF ARTIFACTS ---

YOUR TASK:
Synthesize all the artifacts into a single, comprehensive, and coherent final artifact{query_context}. The final artifact should:
    - Integrate the best insights from all rounds
    - Resolve any tensions or contradictions across rounds
    - Present a unified narrative that goes beyond simple concatenation
    - Build a more complete and nuanced understanding than any single round

1. **Artifact Abstract** (100-150 tokens):
    - Summary of the final artifact's core discovery and significance

2. **Artifact Main Text** (around {self.synthesis_max_tokens} tokens):
    - Synthesize insights across all rounds into a coherent whole
    - Identify and reconcile any contradictions between rounds
    - Emphasize emergent patterns that span multiple rounds
    - Ensure logical flow and clear organization
    - Cite sources using [n] notation after relevant claims or statements (e.g., "This pattern emerged [1,3]")
    - When citing, extract URLs from the round artifact sources and assign them your own NEW sequential numbers

3. **Sources** (dict of url->index for ALL sources cited in your artifact):
    - Extract the URLs of every source you cited in your merged artifact
    - Create NEW sequential numbering starting from 1 for these sources
    - Format: {{"url1": 1, "url2": 2, ...}}
    - Example: If you cited source [2] from Round 1 (url_a) and source [1] from Round 2 (url_b), return: {{"url_a": 1, "url_b": 2}}

IMPORTANT: Create a NEW synthesis that integrates all rounds, not just a summary of them
IMPORTANT: Avoid excessive jargon, keep it logical, clear, and convincing to a skeptical reader
IMPORTANT: Cite sources to support claims, but do NOT create a "Sources" or "References" section in the artifact main text
IMPORTANT: Do NOT reference or mention the previous list of artifacts

Respond with valid JSON only:
{{
    "abstract": "<abstract text>",
    "artifact": "<artifact text>",
    "sources": {{"<url>": <index>, ...}}
}}"""

        try:
            response = self.agent.chat_completion(prompt, response_format={"type": "json_object"})
            result = self.agent.parse_json_response(response)
            if not result or "abstract" not in result or "artifact" not in result:
                raise ValueError("Missing required keys in response")

            # Extract sources if provided, ensure indices are integers
            if "sources" in result and isinstance(result["sources"], dict):
                result["sources"] = {url: int(idx) for url, idx in result["sources"].items()}
                result["sources"] = dict(sorted(result["sources"].items(), key=lambda x: x[1]))
            else:
                result["sources"] = {}

        except Exception as e:
            self.logger.error(f"Artifact merging failed: {e}")
            return all_rounds[-1]

        # Combine metadata from all rounds
        total_queries = sum(r["metadata"]["synthesis_queries"] for r in all_rounds)
        result["metadata"] = {
            "insights_collected": self.kb_manager.size(),
            "max_depth": self.agent.current_depth,
            "pages_visited": len(self.agent.visited_urls),
            "sources_cited": len(result["sources"]),
            "starting_url": self.agent.starting_url,
            "starting_query": self.agent.starting_query,
            "synthesis_mode": all_rounds[-1]["metadata"]["synthesis_mode"],
            "synthesis_queries": total_queries,
            "merged_artifacts": len(all_rounds),
        }
        return result

    def _generate_qa_pairs(self, mode: str, starting_query: str = None) -> List[Tuple[str, str, List[Dict]]]:
        """Generate Q&A pairs with sources"""
        queries = [
            "What are the central themes and patterns discovered?",
            "What unexpected connections or insights emerged?",
            "What contradictions or tensions were revealed?",
            "What questions remain open or were raised?",
            "What novel perspective emerged from this exploration?"
        ]
        if starting_query: queries = [starting_query] + queries
        if mode == "classic": return self._generate_qa_pairs_classic(queries)

        # Iterative mode
        queries, answers, sources_list = [queries[0]], [], []

        for i in range(self.synthesis_iterations):
            answer, sources = self.kb_manager.query(
                queries[-1], top_k=self.synthesis_top_k, top_n=self.synthesis_top_n,
                return_sources=True)
            if not answer: break

            answers.append(answer)
            sources_list.append(sources)

            self.logger.info(f"[SYNTHESIS ITERATION {i+1}/{self.synthesis_iterations}]\nQ: {queries[-1]}\nA: {answers[-1]}")

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

    def _save_synthesis_outputs(self, result: Dict, base_dir: str = None, timestamp: str = None) -> None:
        """Save synthesis with sources in JSON and text formats"""
        if not base_dir: base_dir = self.agent.get_repo()
        if not timestamp: timestamp = datetime.now().strftime("%m%d%H%M")
        base_path = os.path.join(base_dir, f"{self.agent.get_id()}.synthesis.{timestamp}")
        os.makedirs(base_dir, exist_ok=True)

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
