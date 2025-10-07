import networkx as nx
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import requests
import json
import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rome.agent import Agent
from rome.config import DEFAULT_CONFIG, merge_with_default_config, set_attributes_from_config
from rome.config import SHORT_SUMMARY_LEN, SUMMARY_LENGTH, LONG_SUMMARY_LEN, LONGER_SUMMARY_LEN, LONGEST_SUMMARY_LEN
from rome.logger import get_logger

CAESAR_CONFIG = {
    **DEFAULT_CONFIG,

    "CaesarAgent": {
        "max_iterations": 10,
        "starting_url": "https://en.wikipedia.org/wiki/Main_Page",
        "allowed_domains": [],
        "max_depth": 1e12,
        "exploration_temperature": 0.8,
        "save_graph_interval": 5,
        "draw_graph": False,
        "same_page_links": False,
        "checkpoint_interval": 1,  # Checkpoint every N iterations
        "auto_resume": True,  # Automatically resume from checkpoint if found
    },

    "OpenAIHandler": {
        **DEFAULT_CONFIG["OpenAIHandler"],
        "model": "gpt-4o",
        "temperature": 0.1,
        "max_tokens": 4096,
    }
}


class CaesarAgent(Agent):
    """Veni, Vidi, Vici - Web exploration agent with checkpointing support."""

    def __init__(self, name: str = None, role: str = None,
                 repository: str = None, config: Dict = None,
                 starting_url: str = None, allowed_domains: List[str] = None,
                 resume_from_checkpoint: bool = None):

        self._setup_caesar_config_pre_init(config, starting_url, allowed_domains)

        if not role:
            role = self._get_default_role()

        super().__init__(name=name, role=role, repository=repository, config=self.config)

        self._validate_caesar_config()

        # Determine if we should try to resume
        should_resume = (resume_from_checkpoint if resume_from_checkpoint is not None
                        else self.auto_resume)

        # Try to load checkpoint or initialize fresh
        if should_resume and self._load_checkpoint():
            self.logger.info("Resumed from checkpoint")
        else:
            self._setup_exploration_state()
            self.logger.info("Starting fresh exploration")

        self._log_initialization()

    def _setup_caesar_config_pre_init(self, config: Dict = None,
                                      starting_url: str = None,
                                      allowed_domains: List[str] = None) -> None:
        """Setup Caesar config before parent init"""
        if config:
            self.config = merge_with_default_config(config)
            caesar_defaults = CAESAR_CONFIG.get('CaesarAgent', {})
            for key, value in caesar_defaults.items():
                if key not in self.config.get('CaesarAgent', {}):
                    self.config.setdefault('CaesarAgent', {})[key] = value
        else:
            self.config = CAESAR_CONFIG.copy()

        caesar_config = self.config.get('CaesarAgent', {})
        set_attributes_from_config(self, caesar_config,
            ['max_iterations', 'max_depth', 'exploration_temperature',
             'save_graph_interval', 'draw_graph', 'starting_url',
             'allowed_domains', 'same_page_links', 'checkpoint_interval',
             'auto_resume'])

        if starting_url:
            self.starting_url = starting_url
        if allowed_domains:
            self.allowed_domains = allowed_domains

        self._setup_allowed_domains()

    def _get_default_role(self) -> str:
        """Return default Caesar role"""
        return """Your role: You are an explorer seeking novel patterns and connections in information.

Your approach:
- Identify limitations in current understanding as opportunities for deeper exploration
- Seek non-obvious connections between seemingly unrelated concepts
- Question assumptions and explore alternative interpretations
- Synthesize insights from diverse sources into novel perspectives

You navigate through information space systematically yet creatively, always within defined boundaries, building a web of understanding that reveals emergent patterns."""

    def _setup_allowed_domains(self) -> None:
        """Configure allowed domains from starting URL or config"""
        if not hasattr(self, 'logger'):
            self.logger = get_logger()

        if not self.allowed_domains:
            if self.starting_url:
                self.allowed_domains = [urlparse(self.starting_url).netloc]
                self.logger.info(f"Auto-extracted domain: {self.allowed_domains[0]}")
            else:
                self.logger.assert_true(False,
                    "Must provide starting_url and/or allowed_domains")

        self.allow_all_domains = "*" in self.allowed_domains
        if self.allow_all_domains:
            self.logger.info("Wildcard '*' detected - ALL domains allowed!")

    def _validate_caesar_config(self) -> None:
        """Validate Caesar-specific configuration"""
        self.logger.assert_true(self.starting_url is not None,
            "starting_url must be provided")
        self.logger.assert_true(self.allowed_domains and len(self.allowed_domains) > 0,
            "allowed_domains must contain at least one domain")
        self.logger.assert_true(self.max_iterations > 0,
            "max_iterations must be positive")
        self.logger.assert_true(self.max_depth > 0,
            "max_depth must be positive")
        self.logger.assert_true(self.save_graph_interval > 0,
            "save_graph_interval must be positive")
        self.logger.assert_true(self.checkpoint_interval > 0,
            "checkpoint_interval must be positive")

    def _setup_exploration_state(self) -> None:
        """Initialize exploration-specific state"""
        self.graph = nx.DiGraph()
        self.visited_urls = set()
        self.url_stack = [self.starting_url]
        self.current_url = self.starting_url
        self.current_depth = 0
        self.current_iteration = 0

    def _log_initialization(self) -> None:
        """Log initialization summary"""
        self.logger.info(f"CaesarAgent '{self.name}' initialized")
        self.logger.info(f"Starting: {self.starting_url}")
        self.logger.info(f"Domains: {self.allowed_domains}")
        self.logger.info(f"Iterations: {self.max_iterations}, Depth: {self.max_depth}")
        self.logger.info(f"Graph save interval: {self.save_graph_interval}, Draw: {self.draw_graph}")
        self.logger.info(f"Checkpoint interval: {self.checkpoint_interval}")
        if hasattr(self, 'current_iteration') and self.current_iteration > 0:
            self.logger.info(f"Resuming from iteration: {self.current_iteration}")

    def _get_checkpoint_path(self) -> str:
        """Get the checkpoint file path in log directory"""
        return os.path.join(self.get_log_dir(), f"{self.get_id()}.checkpoint.json")

    def _save_checkpoint(self, iteration: int) -> None:
        """Save current exploration state to checkpoint file with validation"""
        # Validate state before saving
        if not self.url_stack:
            self.logger.error("Cannot save checkpoint: url_stack is empty")
            return

        if self.url_stack[-1] != self.current_url:
            self.logger.info(
                f"State inconsistency: url_stack[-1] ({self.url_stack[-1]}) != "
                f"current_url ({self.current_url})"
            )

        try:
            # Convert graph to JSON-serializable format
            graph_data = {
                'nodes': [
                    {
                        'url': node,
                        'depth': self.graph.nodes[node].get('depth', 0),
                        'insights': self.graph.nodes[node].get('insights', ''),
                    }
                    for node in self.graph.nodes()
                ],
                'edges': [
                    {
                        'source': u,
                        'target': v,
                        'reason': self.graph.edges[u, v].get('reason', '')
                    }
                    for u, v in self.graph.edges()
                ]
            }

            checkpoint_data = {
                'iteration': iteration,
                'current_url': self.current_url,
                'current_depth': self.current_depth,
                'visited_urls': list(self.visited_urls),
                'url_stack': self.url_stack,
                'graph': graph_data,
                'timestamp': datetime.now().isoformat(),
                'config': {
                    'starting_url': self.starting_url,
                    'allowed_domains': self.allowed_domains,
                    'max_iterations': self.max_iterations,
                    'max_depth': self.max_depth,
                }
            }

            checkpoint_path = self._get_checkpoint_path()
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=4, ensure_ascii=False)

            self.logger.info(f"Checkpoint saved at iteration {iteration}: {checkpoint_path}")

        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")

    def _load_checkpoint(self) -> bool:
        """
        Load exploration state from checkpoint file with validation.
        Returns True if successfully loaded and validated.
        """
        checkpoint_path = self._get_checkpoint_path()

        if not os.path.exists(checkpoint_path):
            self.logger.debug(f"No checkpoint found at {checkpoint_path}")
            return False

        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)

            # Validate checkpoint matches current config
            config = checkpoint_data.get('config', {})
            if (config.get('starting_url') != self.starting_url or
                config.get('allowed_domains') != self.allowed_domains):
                self.logger.error("Checkpoint config mismatch - starting fresh")
                return False

            # Restore iteration and visited URLs
            self.current_iteration = checkpoint_data['iteration']
            self.visited_urls = set(checkpoint_data['visited_urls'])
            self.url_stack = checkpoint_data['url_stack']

            # Validate url_stack is non-empty (critical check)
            if not self.url_stack:
                self.logger.error("Invalid checkpoint: empty url_stack")
                return False

            # Recompute derived values from url_stack (source of truth)
            self.current_depth = len(self.url_stack) - 1
            self.current_url = self.url_stack[-1]

            # Optional: warn if checkpoint stored values differ (data integrity check)
            checkpoint_url = checkpoint_data.get('current_url')
            checkpoint_depth = checkpoint_data.get('current_depth')
            if checkpoint_url != self.current_url:
                self.logger.info(
                    f"Checkpoint inconsistency: stored current_url={checkpoint_url}, "
                    f"but url_stack[-1]={self.current_url}. Using url_stack as source of truth."
                )
            if checkpoint_depth != self.current_depth:
                self.logger.info(
                    f"Checkpoint inconsistency: stored depth={checkpoint_depth}, "
                    f"but computed depth={self.current_depth}. Using computed value."
                )

            # Restore graph from JSON format
            graph_data = checkpoint_data['graph']
            self.graph = nx.DiGraph()

            for node_data in graph_data['nodes']:
                self.graph.add_node(
                    node_data['url'],
                    depth=node_data['depth'],
                    insights=node_data['insights']
                )

            for edge_data in graph_data['edges']:
                self.graph.add_edge(
                    edge_data['source'],
                    edge_data['target'],
                    reason=edge_data['reason']
                )

            timestamp = checkpoint_data.get('timestamp', 'unknown')
            self.logger.info(f"✓ Checkpoint loaded from {timestamp}")
            self.logger.info(f"✓ Resuming at iteration {self.current_iteration + 1}")
            self.logger.info(f"✓ Current URL: {self.current_url}")
            self.logger.info(f"✓ Depth: {self.current_depth}, Stack size: {len(self.url_stack)}")
            self.logger.info(f"✓ Visited {len(self.visited_urls)} pages, {self.graph.number_of_nodes()} nodes in graph")

            return True

        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            return False

    def _should_checkpoint(self, iteration: int) -> bool:
        """Determine if checkpoint should be saved this iteration"""
        return iteration % self.checkpoint_interval == 0

    def _is_allowed_url(self, url: str) -> bool:
        """Check if URL is within allowed domains"""
        if self.allow_all_domains:
            return True
        parsed = urlparse(url)
        return any(domain in parsed.netloc for domain in self.allowed_domains)

    def _fetch_html(self, url: str) -> Optional[str]:
        """Fetch HTML content with timeout"""
        try:
            response = requests.get(
                url,
                timeout=10,
                headers={'User-Agent': 'CaesarBot/1.0 (Research Agent)'},
                allow_redirects=True
            )
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to fetch {url}: {e}")
            return None

    def _extract_links(self, html: str, base_url: str) -> List[Tuple[str, str]]:
        """Extract links with anchor text"""
        try:
            soup = BeautifulSoup(html, 'html.parser')
        except Exception as e:
            self.logger.error(f"HTML parsing failed: {e}")
            return []

        links = []
        seen = set()

        base_parsed = urlparse(base_url)
        base_path = f"{base_parsed.scheme}://{base_parsed.netloc}{base_parsed.path}"

        for a in soup.find_all('a', href=True):
            try:
                url = urljoin(base_url, a['href'])
            except Exception:
                continue

            if not self.same_page_links:
                parsed = urlparse(url)
                url_without_fragment = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                if url_without_fragment == base_path:
                    continue

            text = a.get_text(strip=True)[:LONGER_SUMMARY_LEN] or "[no text]"

            if (url not in seen and
                self._is_allowed_url(url) and
                url not in self.visited_urls and
                url.startswith('http')):
                links.append((url, text))
                seen.add(url)

        return links

    def perceive(self) -> Tuple[str, List[Tuple[str, str]]]:
        """Phase 1: Extract content and links from current page"""
        self.logger.info(f"[PERCEIVE] {self.current_url}")

        try:
            html = self._fetch_html(self.current_url)
            if not html:
                return "", []

            soup = BeautifulSoup(html, 'html.parser')

            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()

            text = soup.get_text(separator=' ', strip=True)
            text = ' '.join(text.split())
            text = text[:LONGEST_SUMMARY_LEN*100]

            links = self._extract_links(html, self.current_url)

            self.logger.debug(f"Extracted {len(text)} chars, {len(links)} links")
            return text, links

        except Exception as e:
            self.logger.error(f"Perceive phase failed: {e}")
            return "", []

    def think(self, content: str) -> str:
        """Phase 2: Analyze content and extract insights"""
        self.logger.info("[THINK] Analyzing content")

        if not content:
            self.logger.error("No content to analyze")
            return ""

        prompt = f"""Analyze this content and extract key insights focusing on:
- Novel patterns or unexpected connections
- Assumptions being made and alternatives
- Questions raised by the content
- How this relates to or challenges previous insights

Content:
{content}

Provide 3-5 concise, substantive insights that are roughly 250-500 tokens in length total:"""

        try:
            insights = self.chat_completion(
                prompt,
                override_config={'temperature': self.exploration_temperature}
            )
        except Exception as e:
            self.logger.error(f"LLM call failed in think phase: {e}")
            return ""

        try:
            self.kb_manager.add_text(
                insights,
                metadata={
                    'url': self.current_url,
                    'depth': self.current_depth,
                    'iteration': self.current_iteration
                }
            )
        except Exception as e:
            self.logger.error(f"KB add_text failed: {e}")

        self.graph.add_node(
            self.current_url,
            insights=insights,
            depth=self.current_depth
        )

        self.logger.debug(f"Insights:\n{insights}")
        return insights

    def act(self, links: List[Tuple[str, str]]) -> Optional[str]:
        """Phase 3: Choose next URL based on accumulated knowledge"""

        if self.current_depth >= self.max_depth:
            self.logger.debug(f"[ACT] Max depth {self.max_depth} reached - backtracking")
            if len(self.url_stack) > 1:
                self.url_stack.pop()
                self.current_depth = len(self.url_stack) - 1
                return self.url_stack[-1]
            return None

        if not links:
            self.logger.debug("[ACT] No links - backtracking")
            if len(self.url_stack) > 1:
                self.url_stack.pop()
                self.current_depth = len(self.url_stack) - 1
                return self.url_stack[-1]
            self.logger.assert_true(False, "No links and cannot backtrack from starting page")

        kb_context = ""
        if self.kb_manager.size() > 0:
            try:
                kb_context = self.kb_manager.query(
                    "What patterns, gaps, or questions have emerged? What should we explore next?",
                    top_k=5
                )
            except Exception as e:
                self.logger.error(f"KB query failed: {e}")

        link_options = '\n'.join(
            f"{i+1}. [{text}] {url}"
            for i, (url, text) in enumerate(links)
        )

        prompt = f"""Current exploration context:
{kb_context if kb_context else "Beginning exploration"}

Available paths forward:
{link_options}

Based on your role of seeking novel patterns and deeper understanding, which link offers the most promising direction for exploration?

Respond with a JSON object in this exact format:
{{
    "choice": <number from 1 to {len(links)}>,
    "reason": "<brief explanation of why this path is promising>"
}}

Your response must be valid JSON only, nothing else."""

        try:
            response = self.chat_completion(
                prompt,
                override_config={'temperature': self.exploration_temperature},
                response_format={"type": "json_object"}
            )

            decision = self.parse_json_response(response)
            if decision and 'choice' in decision:
                choice = int(decision['choice']) - 1
                choice = max(0, min(choice, len(links) - 1))
                reason = decision.get('reason', 'No reason provided')
            else:
                self.logger.error("Invalid JSON response, using first link")
                choice = 0
                reason = "Fallback due to invalid response"

        except Exception as e:
            self.logger.error(f"LLM decision failed: {e}, using first link")
            choice = 0
            reason = "Fallback due to error"

        next_url = links[choice][0]

        self.graph.add_edge(self.current_url, next_url, reason=reason)

        self.logger.info(f"[ACT] Selected {choice + 1}: {next_url}")
        self.logger.info(f"Reason: {reason}")

        return next_url

    def _save_graph_data(self, iteration: int) -> None:
        """Save graph structure and insights to JSON"""
        try:
            graph_data = {
                'iteration': iteration,
                'nodes': [
                    {
                        'url': node,
                        'depth': self.graph.nodes[node].get('depth', 0),
                        'insights': self.graph.nodes[node].get('insights', ''),
                    }
                    for node in self.graph.nodes()
                ],
                'edges': [
                    {
                        'source': u,
                        'target': v,
                        'reason': self.graph.edges[u, v].get('reason', '')
                    }
                    for u, v in self.graph.edges()
                ]
            }

            filepath = os.path.join(self.get_repo(),
                f"{self.get_id()}.graph_iter{iteration}.json")
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(graph_data, f, indent=4, ensure_ascii=False)

            self.logger.info(f"Graph data saved: {filepath}")

        except Exception as e:
            self.logger.error(f"Failed to save graph data: {e}")

    def _draw_graph_visualization(self, iteration: int) -> None:
        """Create Graphviz visualization of the exploration graph"""
        if not self.draw_graph:
            return

        try:
            import pygraphviz as pgv
        except ImportError:
            self.logger.error("pygraphviz not available, skipping visualization")
            return

        try:
            viz = pgv.AGraph(directed=True, strict=False)
            viz.graph_attr.update(rankdir='TB', size='16,12', dpi='150')
            viz.node_attr.update(shape='box', style='rounded,filled',
                               fillcolor='lightblue', fontsize='10')
            viz.edge_attr.update(color='gray', fontsize='8')

            for node in self.graph.nodes():
                parsed = urlparse(node)
                path_parts = parsed.path.strip('/').split('/')
                label = path_parts[-1] if path_parts and path_parts[-1] else parsed.netloc
                label = label[:SHORT_SUMMARY_LEN] + '...' if len(label) > SHORT_SUMMARY_LEN else label

                insights_preview = self.graph.nodes[node].get('insights', '')[:SUMMARY_LENGTH]
                if insights_preview:
                    label += f"\n{insights_preview}..."

                depth = self.graph.nodes[node].get('depth', 0)
                color = f"0.{min(9, depth)} 0.3 1.0"

                viz.add_node(node, label=label, fillcolor=color)

            for u, v in self.graph.edges():
                reason = self.graph.edges[u, v].get('reason', '')[:SUMMARY_LENGTH] + "..."
                viz.add_edge(u, v, label=reason)

            filepath = os.path.join(self.get_repo(),
                f"{self.get_id()}.graph_iter{iteration}.png")
            viz.draw(filepath, prog='dot')
            self.logger.info(f"Graph visualization saved: {filepath}")

        except Exception as e:
            self.logger.error(f"Failed to create graph visualization: {e}")

    def _should_save_graph(self, iteration: int) -> bool:
        """Determine if graph should be saved this iteration"""
        return (iteration == 1 or
                iteration % self.save_graph_interval == 0 or
                iteration == self.max_iterations)

    def explore(self) -> str:
        """Execute the main exploration loop"""
        start_iteration = self.current_iteration + 1
        self.logger.info(f"Beginning exploration: iterations {start_iteration} to {self.max_iterations}")

        self.last_iter_saved = -1
        for iteration in range(start_iteration, self.max_iterations + 1):
            self.current_iteration = iteration

            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"Iteration {iteration}/{self.max_iterations}")
            self.logger.info(f"Depth: {self.current_depth}/{self.max_depth}")
            self.logger.info(f"URL: {self.current_url}")
            self.logger.info(f"{'='*80}")

            # Perceive
            content, links = self.perceive()
            if not content:
                if len(self.url_stack) > 1:
                    self.url_stack.pop()
                    self.current_url = self.url_stack[-1]
                    self.current_depth = len(self.url_stack) - 1
                    continue
                else:
                    self.logger.error("Cannot continue exploration")
                    break

            self.visited_urls.add(self.current_url)

            # Think
            self.think(content)

            # Save checkpoint periodically
            if self._should_checkpoint(iteration):
                self._save_checkpoint(iteration)

            # Save graph periodically
            if self._should_save_graph(iteration):
                self.logger.debug(f"Saving graph at iteration {iteration}")
                self._save_graph_data(iteration)
                self._draw_graph_visualization(iteration)
                self.last_iter_saved = iteration

            # Act (skip on last iteration)
            if iteration < self.max_iterations:
                next_url = self.act(links)
                if next_url:
                    self.current_url = next_url
                    self.url_stack.append(next_url)
                    self.current_depth = len(self.url_stack) - 1
                else:
                    break

        # Final saves
        if self.last_iter_saved < self.max_iterations:
            self.logger.debug("Final graph save")
            self._save_graph_data(self.max_iterations)
            self._draw_graph_visualization(self.max_iterations)
            self._save_checkpoint(self.max_iterations)

        self.logger.info(f"\nExploration complete: visited {len(self.visited_urls)} pages")
        return self._synthesize_artifact()

    def _synthesize_artifact(self) -> Dict[str, str]:
        """Generate final synthesis artifact as JSON with abstract and main content"""
        self.logger.info("[SYNTHESIS] Generating artifact")

        if self.kb_manager.size() == 0:
            self.logger.error("No insights in KB, cannot synthesize")
            return {"abstract": "", "artifact": "No insights collected during exploration."}

        queries = [
            "What are the central themes and patterns discovered?",
            "What unexpected connections or insights emerged?",
            "What contradictions or tensions were revealed?",
            "What questions remain open or were raised?",
            "What novel perspective emerged from this exploration?"
        ]

        perspectives = []
        for q in queries:
            try:
                resp = self.kb_manager.query(q, top_k=5)
                if resp:
                    perspectives.append(f"### {q}\n{resp}\n")
            except Exception as e:
                self.logger.error(f"KB query failed for '{q}': {e}")

        if not perspectives:
            return {"abstract": "", "artifact": "Unable to query knowledge base for synthesis."}
        perspectives = '\n'.join(perspectives)

        synthesis_prompt = f"""You have completed an exploration through {len(self.visited_urls)} interconnected sources, gathering {self.kb_manager.size()} insights.

{perspectives}

Create a synthesis with two parts:

1. **Abstract** (100-150 tokens): A concise summary capturing the core discovery, key patterns, and significance

2. **Artifact** (Up to 3000 tokens): A structured synthesis that:
   - Identifies emergent patterns not visible in individual sources
   - Highlights novel connections and unexpected relationships
   - Explores tensions, contradictions, or open questions
   - Proposes new directions or perspectives

Make both intellectually engaging and substantive.

Respond with a JSON object in this exact format:
{{
    "abstract": "<your abstract text here>",
    "artifact": "<your full synthesis text here>"
}}

Your response must be valid JSON only, nothing else."""

        try:
            response = self.chat_completion(
                synthesis_prompt,
                response_format={"type": "json_object"}
            )
            result = self.parse_json_response(response)

            if not result or "abstract" not in result or "artifact" not in result:
                self.logger.error("Invalid JSON response from LLM")
                raise ValueError("Missing required keys in response")
        except Exception as e:
            self.logger.error(f"Synthesis LLM call failed: {e}")
            result = {
                "abstract": "Synthesis generation failed.",
                "artifact": f"Synthesis generation failed: {e}"
            }

        result["metadata"] = {
            "pages_visited": len(self.visited_urls),
            "insights_collected": self.kb_manager.size(),
            "max_depth": self.current_depth,
            "starting_url": self.starting_url
        }

        try:
            filepath = os.path.join(self.get_repo(), f"{self.get_id()}.synthesis.json")
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=4, ensure_ascii=False)
            self.logger.info(f"Artifact saved as JSON: {filepath}")

            filepath = os.path.splitext(filepath)[0] + ".txt"
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"ABSTRACT:\n{result['abstract']}\n\n")
                f.write(f"ARTIFACT:\n{result['artifact']}\n\n")
                f.write(f"METADATA:\n{str(result['metadata'])}")
            self.logger.info(f"Artifact saved as txt: {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save artifact: {e}")

        self.logger.info("Artifact synthesis complete")
        return result
