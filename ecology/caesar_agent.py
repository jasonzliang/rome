import networkx as nx
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import requests
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rome.agent import Agent
from rome.config import DEFAULT_CONFIG, set_attributes_from_config
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
        "save_graph_interval": 5,  # Save graph every N iterations
        "draw_graph": True,  # Whether to visualize graph
    },

    "OpenAIHandler": {
        **DEFAULT_CONFIG["OpenAIHandler"],
        "model": "gpt-4o",
        "temperature": 0.1,
        "max_tokens": 4096,
    }
}


class CaesarAgent(Agent):
    """Veni, Vidi, Vici - Web exploration agent that systematically explores allowed domains."""

    def __init__(self, name: str = None, role: str = None,
                 repository: str = None, config: Dict = None,
                 starting_url: str = None, allowed_domains: List[str] = None):

        self.config = {**CAESAR_CONFIG, **(config or {})}
        self.logger = get_logger()

        # Core setup
        self._setup_config(self.config)

        if not role:
            role = """Your role: You are an explorer seeking novel patterns and connections in information.

Your approach:
- Identify limitations in current understanding as opportunities for deeper exploration
- Seek non-obvious connections between seemingly unrelated concepts
- Question assumptions and explore alternative interpretations
- Synthesize insights from diverse sources into novel perspectives

You navigate through information space systematically yet creatively, always within defined boundaries, building a web of understanding that reveals emergent patterns."""

        self._validate_name_role(name, role)
        self._setup_repository_and_logging(repository)
        self._setup_openai_handler()
        self._setup_knowledge_base()

        # Caesar-specific config
        caesar_config = self.config.get('CaesarAgent', {})
        set_attributes_from_config(self, caesar_config,
            ['max_iterations', 'max_depth', 'exploration_temperature',
             'save_graph_interval', 'draw_graph'],
            ['starting_url', 'allowed_domains'])

        # Override from constructor if provided
        if starting_url:
            self.starting_url = starting_url
        if allowed_domains:
            self.allowed_domains = allowed_domains

        # Auto-extract domain if needed
        if not self.allowed_domains:
            if self.starting_url:
                self.allowed_domains = [urlparse(self.starting_url).netloc]
                self.logger.info(f"Auto-extracted domain: {self.allowed_domains[0]}")
            else:
                self.logger.assert_true(False, "Must provide starting_url and/or allowed_domains")

        # Check for wildcard
        self.allow_all_domains = "*" in self.allowed_domains
        if self.allow_all_domains:
            self.logger.warning("Wildcard '*' detected - ALL domains allowed!")

        # Validation
        self.logger.assert_true(self.starting_url is not None,
            "starting_url must be provided")
        self.logger.assert_true(self.allowed_domains and len(self.allowed_domains) > 0,
            "allowed_domains must contain at least one domain")
        self.logger.assert_true(self.max_iterations > 0, "max_iterations must be positive")
        self.logger.assert_true(self.max_depth > 0, "max_depth must be positive")
        self.logger.assert_true(self.save_graph_interval > 0, "save_graph_interval must be positive")

        # Exploration state
        self.graph = nx.DiGraph()
        self.visited_urls = set()
        self.url_stack = [self.starting_url]
        self.current_url = self.starting_url
        self.current_depth = 0

        self.logger.info(f"CaesarAgent '{self.name}' initialized")
        self.logger.info(f"Starting: {self.starting_url}")
        self.logger.info(f"Domains: {self.allowed_domains}")
        self.logger.info(f"Iterations: {self.max_iterations}, Depth: {self.max_depth}")
        self.logger.info(f"Graph save interval: {self.save_graph_interval}, Draw: {self.draw_graph}")

    def _is_allowed_url(self, url: str) -> bool:
        """Check if URL is within allowed domains"""
        # If wildcard is set, allow all domains
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

        for a in soup.find_all('a', href=True):
            try:
                url = urljoin(base_url, a['href'])
            except Exception:
                continue

            text = a.get_text(strip=True)[:LONGEST_SUMMARY_LEN] or "[no text]"

            if (url not in seen and
                self._is_allowed_url(url) and
                url not in self.visited_urls and
                url.startswith('http')):
                links.append((url, text))
                seen.add(url)

        return links[:LONG_SUMMARY_LEN]

    def perceive(self) -> Tuple[str, List[Tuple[str, str]]]:
        """Phase 1: Extract content and links from current page"""
        self.logger.info(f"[PERCEIVE] {self.current_url}")

        try:
            html = self._fetch_html(self.current_url)
            if not html:
                return "", []

            soup = BeautifulSoup(html, 'html.parser')

            # Remove non-content
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()

            # Extract text
            text = soup.get_text(separator='\n', strip=True)
            lines = [line for line in text.split('\n') if line.strip()]
            text = '\n'.join(lines)[:LONGEST_SUMMARY_LEN*100]

            # Extract links
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

        # Store in KB
        try:
            self.kb_manager.add_text(
                insights,
                metadata={
                    'url': self.current_url,
                    'depth': self.current_depth,
                    'iteration': len(self.visited_urls) + 1
                }
            )
        except Exception as e:
            self.logger.error(f"KB add_text failed: {e}")

        # Add to graph
        self.graph.add_node(
            self.current_url,
            insights=insights,
            full_insights=insights,
            depth=self.current_depth
        )

        self.logger.debug(f"Insights:\n{insights}")
        return insights

    def act(self, links: List[Tuple[str, str]]) -> Optional[str]:
        """Phase 3: Choose next URL based on accumulated knowledge"""

        # Check depth limit
        if self.current_depth >= self.max_depth:
            self.logger.debug(f"[ACT] Max depth {self.max_depth} reached - backtracking")
            if len(self.url_stack) > 1:
                self.url_stack.pop()
                self.current_depth = len(self.url_stack) - 1
                return self.url_stack[-1]
            return None

        # No links available
        if not links:
            self.logger.debug("[ACT] No links - backtracking")
            if len(self.url_stack) > 1:
                self.url_stack.pop()
                self.current_depth = len(self.url_stack) - 1
                return self.url_stack[-1]
            self.logger.assert_true(False, "No links and cannot backtrack from starting page")

        # Query KB for context
        kb_context = ""
        if self.kb_manager.size() > 0:
            try:
                kb_context = self.kb_manager.query(
                    "What patterns, gaps, or questions have emerged? What should we explore next?",
                    top_k=5
                )
            except Exception as e:
                self.logger.error(f"KB query failed: {e}")

        # Format links
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

            # Parse JSON response
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

        # Add edge to graph
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
                        'full_insights': self.graph.nodes[node].get('full_insights', '')
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

            filepath = os.path.join(self.get_log_dir(),
                                   f"{self.get_id()}.graph_iter{iteration}.json")
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(graph_data, f, indent=2, ensure_ascii=False)

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
            # Create pygraphviz graph
            viz = pgv.AGraph(directed=True, strict=False)
            viz.graph_attr.update(rankdir='TB', size='16,12', dpi='150')
            viz.node_attr.update(shape='box', style='rounded,filled',
                               fillcolor='lightblue', fontsize='10')
            viz.edge_attr.update(color='gray', fontsize='8')

            # Add nodes with shortened labels
            for node in self.graph.nodes():
                # Create shortened label
                parsed = urlparse(node)
                path_parts = parsed.path.strip('/').split('/')
                label = path_parts[-1] if path_parts and path_parts[-1] else parsed.netloc
                label = label[:SHORT_SUMMARY_LEN] + '...' if len(label) > SHORT_SUMMARY_LEN else label

                insights_preview = self.graph.nodes[node].get('insights', '')[:LONG_SUMMARY_LEN]
                if insights_preview:
                    label += f"\n{insights_preview}..."

                depth = self.graph.nodes[node].get('depth', 0)
                color = f"0.{min(9, depth)} 0.3 1.0"  # HSV color by depth

                viz.add_node(node, label=label, fillcolor=color)

            # Add edges with reasons
            for u, v in self.graph.edges():
                reason = self.graph.edges[u, v].get('reason', '')[:LONG_SUMMARY_LEN]
                viz.add_edge(u, v, label=reason)

            # Save visualization
            filepath = os.path.join(self.get_log_dir(),
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
        self.logger.info(f"Beginning exploration: {self.max_iterations} iterations")

        for iteration in range(1, self.max_iterations + 1):
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

            # Save graph periodically
            if self._should_save_graph(iteration):
                self.logger.debug(f"Saving graph at iteration {iteration}")
                self._save_graph_data(iteration)
                self._draw_graph_visualization(iteration)

            # Act (skip on last iteration)
            if iteration < self.max_iterations:
                next_url = self.act(links)
                if next_url:
                    self.current_url = next_url
                    self.url_stack.append(next_url)
                    self.current_depth = len(self.url_stack) - 1
                else:
                    break

        # Final graph save
        self.logger.debug("Final graph save")
        self._save_graph_data(self.max_iterations)
        self._draw_graph_visualization(self.max_iterations)

        self.logger.info(f"\nExploration complete: visited {len(self.visited_urls)} pages")
        return self._synthesize_artifact()

    def _synthesize_artifact(self) -> str:
        """Generate final synthesis artifact"""
        self.logger.info("[SYNTHESIS] Generating artifact")

        if self.kb_manager.size() == 0:
            self.logger.error("No insights in KB, cannot synthesize")
            return "No insights collected during exploration."

        # Query KB from multiple perspectives
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
            return "Unable to query knowledge base for synthesis."

        # Generate synthesis
        prompt = f"""You have completed an exploration through {len(self.visited_urls)} interconnected sources, gathering {self.kb_manager.size()} insights.

{'\n'.join(perspectives)}

Create a structured synthesis that:
1. Identifies emergent patterns not visible in individual sources
2. Highlights novel connections and unexpected relationships
3. Explores tensions, contradictions, or open questions
4. Proposes new directions or perspectives

Make this synthesis intellectually engaging and substantive."""

        try:
            artifact = self.chat_completion(
                prompt
            )
        except Exception as e:
            self.logger.error(f"Synthesis LLM call failed: {e}")
            return f"Synthesis generation failed: {e}"

        self.logger.info("Artifact synthesis complete")
        return artifact
