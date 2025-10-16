"""Caesar Agent - Web exploration agent with graph-based navigation"""
import networkx as nx
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import copy
import requests
import json
import os
import sys
import time
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rome.base_agent import BaseAgent
from rome.config import (DEFAULT_CONFIG, merge_with_default_config, set_attributes_from_config,
                     SHORT_SUMMARY_LEN, SUMMARY_LENGTH, LONG_SUMMARY_LEN,
                     LONGER_SUMMARY_LEN, LONGEST_SUMMARY_LEN)
from rome.logger import get_logger
from rome.kb_client import ChromaClientManager

# Maximum number of links to consider when selecting next webpage to go to
MAX_NUM_LINKS = 2000
# Maximum number of visited links to consider when selecting next webpage
MAX_NUM_VISITED_LINKS = 1000

CAESAR_CONFIG = {
    "CaesarAgent": {
        # Total number of pages to explore before stopping
        "max_iterations": 10,
        # Maximum depth of exploration tree before backtracking
        "max_depth": 10000,
        # Initial URL to begin exploration
        "starting_url": "https://en.wikipedia.org/wiki/Main_Page",
        # Domains to allow exploration in; empty list uses starting_url domain; use ["*"] to allow any
        "allowed_domains": [],
        # Generate visual graph representations (requires pygraphviz)
        "draw_graph": False,
        # Save exploration graph every N iterations
        "save_graph_interval": 1,
        # Save checkpoint every N iterations for resumption
        "checkpoint_interval": 1,

        # Whether to follow links that point to the same page (different fragments)
        "same_page_links": False,
        # Whether to allow agent to return to page it has seen before
        "allow_revisit": True,
        # Whether to use new link display format or not
        "fancy_link_display": True,
        # Whether to dynamically determine to explore or go back to visited pages
        "use_explore_strategy": True,

        # False for classic mode
        "iterative_synthesis": True,
        # Number of q/a iterations
        "synthesis_iterations": 10,
        # KB docs per query
        "top_k": 10,
        # Temperature for agent's ACT/THINK phases to encourage exploration
        "exploration_temperature": 0.8,
    },

    "AgentMemory": {
        # Whether to enable agent memory
        "enabled": True,
        # Whether to use vector DB or graph DB
        "use_graph": False,
    },

    "OpenAIHandler": {
        # Model for exploration analysis and synthesis
        "model": "gpt-4o",
        # Base temperature for LLM (overridden by exploration_temperature for ACT/THINK)
        "temperature": 0.1,
        # Maximum tokens per LLM response
        "max_tokens": 4096,
        # API timeout in seconds
        "timeout": 120,
    }
}


class CaesarAgent(BaseAgent):
    """Veni, Vidi, Vici - Web exploration agent with checkpointing support"""
    def __init__(self, name: str = None, role: str = None,
             repository: str = None, config: Dict = None,
             starting_url: str = None, allowed_domains: List[str] = None):

        # Prepare merged config BEFORE calling super().__init__()
        merged_config = self._prepare_caesar_config(config, starting_url, allowed_domains)
        if not role: role = self._get_default_role()
        # Pass the merged config to BaseAgent (this configures the logger)
        super().__init__(name, role, repository, merged_config)

        # NOW setup Caesar-specific attributes (logger is configured)
        caesar_config = self.config.get('CaesarAgent', {})
        set_attributes_from_config(self, caesar_config, CAESAR_CONFIG['CaesarAgent'].keys())
        self._setup_allowed_domains()

        self._validate_caesar_config()
        self._setup_knowledge_base()

        if self._load_checkpoint():
            self.logger.info("Resumed from checkpoint")
        else:
            self._setup_exploration_state()
            self.logger.info("Starting fresh exploration")

        self._log_initialization()

    def _prepare_caesar_config(self, config: Dict = None,
                               starting_url: str = None,
                               allowed_domains: List[str] = None) -> Dict:
        """Prepare Caesar config before parent init"""
        def deep_merge(base: Dict, overlay: Dict) -> None:
            """Deep merge overlay into base (modifies base in place)"""
            for key, value in overlay.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    deep_merge(base[key], value)
                else:
                    base[key] = value

        merged_config = copy.deepcopy(DEFAULT_CONFIG)

        # Apply CAESAR_CONFIG (Caesar defaults)
        deep_merge(merged_config, CAESAR_CONFIG)

        # Apply custom config if provided (overrides Caesar defaults)
        if config: deep_merge(merged_config, config)

        # Apply constructor parameters (highest priority)
        if starting_url:
            merged_config.setdefault('CaesarAgent', {})['starting_url'] = starting_url
        if allowed_domains:
            merged_config.setdefault('CaesarAgent', {})['allowed_domains'] = allowed_domains
        return merged_config

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
        """Configure allowed domains (called before super().__init__)"""
        # Use get_logger() directly since self.logger not yet initialized
        logger = get_logger()

        if not self.allowed_domains:
            if self.starting_url:
                self.allowed_domains = [urlparse(self.starting_url).netloc]
                logger.info(f"Auto-extracted domain: {self.allowed_domains[0]}")
            else:
                raise ValueError("Must provide starting_url and/or allowed_domains")

        self.allow_all_domains = "*" in self.allowed_domains
        if self.allow_all_domains:
            logger.info("Wildcard '*' detected - ALL domains allowed!")

    def _validate_caesar_config(self) -> None:
        """Validate Caesar-specific configuration"""
        self.logger.assert_true(self.starting_url, "starting_url required")
        self.logger.assert_true(self.allowed_domains, "allowed_domains required")
        self.logger.assert_true(self.max_iterations >= 0, "max_iterations must be >= 0")
        self.logger.assert_true(self.max_depth > 0, "max_depth must be > 0")

        # Validate starting URL to make sure it can be reached
        try:
            parsed = urlparse(self.starting_url)
            if not all([parsed.scheme, parsed.netloc]):
                raise ValueError(f"Invalid starting URL: {self.starting_url}")
            if parsed.scheme not in ['http', 'https']:
                raise ValueError(f"URL must use http/https: {self.starting_url}")
        except Exception as e:
            raise ValueError(f"Invalid starting_url: {e}")


    def _setup_exploration_state(self) -> None:
        """Initialize exploration state"""
        self.graph = nx.DiGraph()
        self.visited_urls = {}
        self.failed_urls = set()
        self.url_stack = [self.starting_url]
        self.current_url = self.starting_url
        self.current_depth = 0
        self.current_iteration = 0

    def _setup_knowledge_base(self) -> None:
        """Setup knowledge base"""
        self.kb_manager = ChromaClientManager(agent=self)
        self.logger.info(f"Knowledge base initialized: {self.kb_manager.info()}")

    def _log_initialization(self) -> None:
        """Log initialization summary"""
        self.logger.info(f"CaesarAgent '{self.name}' initialized")
        # self.logger.info(f"Log file: {os.path.join(self.get_log_dir(), f'{self.get_id()}.console.log')}")
        self.logger.info(f"Starting: {self.starting_url}")
        self.logger.info(f"Domains: {self.allowed_domains}")
        self.logger.info(f"Iterations: {self.max_iterations}, Depth: {self.max_depth}")

    def _get_checkpoint_path(self) -> str:
        """Get checkpoint file path"""
        return os.path.join(self.get_log_dir(), f"{self.get_id()}.checkpoint.json")

    def _save_checkpoint(self, iteration: int) -> None:
        """Save exploration state to checkpoint"""
        if not self.url_stack:
            self.logger.error("Cannot save checkpoint: empty url_stack")
            return

        try:
            graph_data = {
                'nodes': [{'url': n, 'depth': self.graph.nodes[n].get('depth', 0),
                          'insights': self.graph.nodes[n].get('insights', '')}
                         for n in self.graph.nodes()],
                'edges': [{'source': u, 'target': v,
                          'reason': self.graph.edges[u, v].get('reason', '')}
                         for u, v in self.graph.edges()]
            }

            checkpoint_data = {
                'iteration': iteration,
                'current_url': self.current_url,
                'current_depth': self.current_depth,
                'visited_urls': self.visited_urls,
                'failed_urls': list(self.failed_urls),
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

            with open(self._get_checkpoint_path(), 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=4, ensure_ascii=False)

            self.logger.info(
                f"Checkpoint saved on iteration {iteration}: {self._get_checkpoint_path()}")

        except Exception as e:
            self.logger.error(f"Failed to save checkpoint on iteration {iteration}: {e}")

    def _load_checkpoint(self) -> bool:
        """Load exploration state from checkpoint"""
        checkpoint_path = self._get_checkpoint_path()
        if not os.path.exists(checkpoint_path):
            return False

        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)

            config = checkpoint_data.get('config', {})
            if (config.get('starting_url') != self.starting_url or
                config.get('allowed_domains') != self.allowed_domains):
                self.logger.error("Checkpoint config mismatch")
                return False

            self.current_iteration = checkpoint_data['iteration']
            self.visited_urls = checkpoint_data['visited_urls']
            self.failed_urls = set(checkpoint_data['failed_urls'])
            self.url_stack = checkpoint_data['url_stack']

            if not self.url_stack:
                self.logger.error("Invalid checkpoint: empty url_stack")
                return False

            self.current_depth = len(self.url_stack)
            self.current_url = self.url_stack[-1]

            graph_data = checkpoint_data['graph']
            self.graph = nx.DiGraph()

            for node_data in graph_data['nodes']:
                self.graph.add_node(node_data['url'], depth=node_data['depth'],
                                   insights=node_data['insights'])

            for edge_data in graph_data['edges']:
                self.graph.add_edge(edge_data['source'], edge_data['target'],
                                   reason=edge_data['reason'])

            self.logger.info(f"Checkpoint loaded from {checkpoint_data.get('timestamp')}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            return False

    def _is_allowed_url(self, url: str) -> bool:
        """Check if URL is within allowed domains"""
        if self.allow_all_domains: return True
        parsed = urlparse(url)
        return any(domain in parsed.netloc for domain in self.allowed_domains)

    def _fetch_html(self, url: str) -> Optional[str]:
        """Fetch HTML content"""
        try:
            response = requests.get(url,
                timeout=5,
                headers={'User-Agent': 'CaesarBot/1.0'},
                allow_redirects=True)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to fetch {url}: {e}")
            return None

    def _extract_links(self, html: str, base_url: str) -> List[Tuple[str, str]]:
        """Extract links with anchor text"""
        try:
            soup = BeautifulSoup(html, 'html.parser')
        except Exception:
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

            if (url not in seen and self._is_allowed_url(url) and
                url not in self.failed_urls and
                (url not in self.visited_urls or self.allow_revisit) and
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
                self.failed_urls.add(self.current_url)
                return "", []

            soup = BeautifulSoup(html, 'html.parser')
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()

            text = soup.get_text(separator=' ', strip=True)
            text = ' '.join(text.split())[:LONGEST_SUMMARY_LEN*100]
            links = self._extract_links(html, self.current_url)

            return text, links

        except Exception as e:
            self.logger.error(f"Perceive phase failed: {e}")
            self.failed_urls.add(self.current_url)
            return "", []

    def think(self, content: str) -> str:
        """Phase 2: Analyze content and extract insights"""
        self.logger.info("[THINK] Analyzing content")
        prev_insights = self.graph.nodes[self.current_url].get('insights', '') if self.current_url in self.graph.nodes else ''
        # if self.current_url in self.visited_urls:
        #     self.visited_urls[self.current_url] += 1
        #     self.logger.info(
        #         f"[THINK] Already analyzed ({self.visited_urls[self.current_url]} visits), skipping")
        #     return prev_insights
        if not content: return ""

        prompt = f"""Analyze this content and extract key insights focusing on:
- Novel patterns or unexpected connections
- Assumptions being made and alternatives
- Questions raised by the content
- How this relates to or challenges previous insights
- Build upon insights from previous analysis

CONTENT:
{content}

PREVIOUS INSIGHTS:
{prev_insights if prev_insights else 'No previous insights available'}

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
            self.kb_manager.add_text(insights, metadata={
                'url': self.current_url,
                'depth': self.current_depth,
                'iteration': self.current_iteration
            })
        except Exception as e:
            self.logger.error(f"KB add_text failed: {e}")

        self.visited_urls[self.current_url] = self.visited_urls.get(self.current_url, 0) + 1
        self.graph.add_node(self.current_url, insights=insights, depth=self.current_depth)
        return insights

    def _advance_to_url(self, url: str) -> None:
        """Advance exploration to new URL"""
        self.url_stack.append(url)
        self.current_url = url
        self.current_depth = len(self.url_stack)  # Derive from stack

    def _backtrack(self) -> bool:
        """Backtrack to parent URL"""
        if len(self.url_stack) > 1:
            self.url_stack.pop()
            self.current_url = self.url_stack[-1]
            self.current_depth = len(self.url_stack)  # Derive from stack
            self.logger.debug(f"Backtracked to parent link: {self.current_url}")
            return True
        return False

    def _get_parent_url(self):
        """Get url to parent page"""
        return self.url_stack[-2] if len(self.url_stack) > 1 else None

    def _determine_exploration_strategy(self, kb_context: str, memory_context: str) -> str:
            """Use LLM to determine optimal exploration strategy"""
            prompt = f"""Based on accumulated knowledge and navigation patterns, determine the optimal exploration strategy.

CURRENT KNOWLEDGE STATE:
{kb_context}

HISTORICAL NAVIGATION PATTERNS:
{memory_context}

CURRENT EXPLORATION CONTEXT:
- Current depth: {self.current_depth}/{self.max_depth}
- Pages visited: {len(self.visited_urls)}
- Current URL: {self.current_url}

Analyze whether the agent should:
1. **Revisit** previously seen pages to deepen understanding of known valuable areas
2. **Explore** new unexplored links to discover novel information
3. **Backtrack** to the previous page to try alternative paths

Consider:
- Knowledge gaps vs areas of saturation
- Depth of current exploration branch
- Success patterns from previous decisions
- Risk/reward of new exploration vs consolidation

Provide a strategic recommendation in 2-3 sentences."""

            try:
                return self.chat_completion(
                    prompt,
                    override_config={'temperature': self.exploration_temperature}
                )
            except Exception as e:
                self.logger.error(f"Exploration strategy determination failed: {e}")
                return ""

    def _format_link_options(self, links: List[Tuple[str, str]]) -> Tuple[List[str], List[str]]:
        """Format links into display options, returns (link_options, url_map)"""
        link_options, url_map = [], []

        if self.fancy_link_display:
            parent_url = self._get_parent_url()
            visited_links = [(url, text, self.visited_urls.get(url, 0))
                             for url, text in links
                             if url in self.visited_urls and url != self.current_url]
            new_links = [(url, text)
                         for url, text in links
                         if url not in self.visited_urls and url != self.current_url]

            link_options.extend([
                "- INITIAL STARTING LINK -",
                f"1. [Back to starting LINK] ({self.visited_urls.get(self.starting_url, 0)} visits) {self.starting_url}"
            ])
            url_map.append(self.starting_url)

            if parent_url:
                link_options.extend([
                    "\n- IMMEDIATE PREVIOUS LINK -",
                    f"2. [Back to previous link] ({self.visited_urls.get(parent_url, 0)} visits) {parent_url}"
                ])
                url_map.append(parent_url)

            if visited_links:
                link_options.append("\n- PREVIOUSLY VISITED LINKS -")
                for url, text, visit_count in visited_links[:MAX_NUM_VISITED_LINKS]:
                    link_options.append(f"{len(url_map)+1}. [{text}] ({visit_count} visits) {url}")
                    url_map.append(url)

            if new_links:
                link_options.append("\n- NEW UNEXPLORED LINKS -")
                for url, text in new_links[:MAX_NUM_LINKS]:
                    link_options.append(f"{len(url_map)+1}. [{text}] {url}")
                    url_map.append(url)
        else:
            for url, text in links[:MAX_NUM_LINKS]:
                if url == self.current_url:
                    continue
                visit_count = self.visited_urls.get(url, 0)
                visit_info = f" ({visit_count} visits so far) " if visit_count else " "
                link_options.append(f"{len(url_map)+1}. [{text}]{visit_info}{url}")
                url_map.append(url)

        return link_options, url_map

    def _select_next_link(self, links: List[Tuple[str, str]]) -> Tuple[Optional[str], str]:
        """Use LLM to select best link, returns (url, reason)"""
        kb_context = memory_context = explore_strategy = ""
        try:
            kb_context = self.kb_manager.query(
                "What patterns, gaps, or questions have emerged from our knowledge? What should we explore next?", top_k=self.top_k)
            memory_context = self.recall(
                f"What webpages have I frequently visited and what navigation patterns have emerged in relation to the following insights:\n{kb_context}")
            if self.use_explore_strategy:
                explore_strategy = self._determine_exploration_strategy(kb_context, memory_context)
        except Exception as e:
            self.logger.error(f"KB/memory/explore for link selection failed: {e}")

        link_options, url_map = self._format_link_options(links)

        prompt = f"""You are selecting the next webpage link to explore based on your role

CURRENT EXPLORATION INSIGHTS:
{kb_context if kb_context else "No exploration insights available."}

HISTORICAL EXPLORATION PATTERNS:
{memory_context if memory_context else "No exploration history available."}

EXPLORATION STRATEGY:
{explore_strategy if explore_strategy else "No exploration strategy available"}

AVAILABLE PATHS FORWARD:
{'\n'.join(link_options)}

TASK: Based on current insights and historical patterns, which page link offers the most promising direction to explore and deepen understanding? Use the exploration strategy to determine which link to select.

Respond with a JSON object in this exact format:
{{
    "choice": <number from 1 to {len(url_map)}>,
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
                choice_num = max(0, min(int(decision['choice']) - 1, len(url_map) - 1))
                url = url_map[choice_num]
                reason = decision.get('reason', 'No reason provided')
                return url, reason
        except Exception as e:
            self.logger.error(f"LLM decision failed: {e}")

        return url_map[0] if url_map else None, "Fallback due to error"

    def act(self, links: List[Tuple[str, str]]) -> Optional[str]:
        """Phase 3: Choose next URL based on accumulated knowledge"""
        if self.current_depth > self.max_depth or not links:
            self._backtrack(); return self.current_url

        next_url, reason = self._select_next_link(links)
        if next_url == self._get_parent_url():
            self._backtrack(); return self.current_url
        if next_url == self.starting_url:
            self._restart_exploration(); return self.current_url

        self.graph.add_edge(self.current_url, next_url, reason=reason)

        current_domain = urlparse(self.current_url).netloc
        next_domain = urlparse(next_url).netloc
        self.remember(
            f"Agent navigated from {self.current_url} to visit {next_url} "
            f"(from domain {current_domain} to domain {next_domain}). "
            f"Navigation performed on iteration {self.current_depth} at depth {self.current_depth}. "
            f"Agent selected new webpage from {len(links)} options because: {reason}",
            context="navigation",
            metadata={
                'from_url': self.current_url,
                'to_url': next_url,
                'from_domain': current_domain,
                'to_domain': next_domain,
                'reason': reason,
                'alternatives': len(links),
                'depth': self.current_depth
            }
        )

        self._advance_to_url(next_url)
        self.logger.info(f"[ACT] Selected link: {next_url}")
        self.logger.info(f"Reason: {reason}")

        return next_url

    def _save_graph_data(self, iteration: int) -> None:
        """Save graph structure to JSON"""
        try:
            graph_data = {
                'iteration': iteration,
                'nodes': [{'url': n, 'depth': self.graph.nodes[n].get('depth', 0),
                          'insights': self.graph.nodes[n].get('insights', '')}
                         for n in self.graph.nodes()],
                'edges': [{'source': u, 'target': v,
                          'reason': self.graph.edges[u, v].get('reason', '')}
                         for u, v in self.graph.edges()]
            }

            filepath = os.path.join(self.get_repo(),
                f"{self.get_id()}.graph_iter{iteration}.json")
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(graph_data, f, indent=4, ensure_ascii=False)

        except Exception as e:
            self.logger.error(f"Failed to save graph data: {e}")

    def _draw_graph_visualization(self, iteration: int) -> None:
        """Create Graphviz visualization"""
        if not self.draw_graph:
            return

        try:
            import pygraphviz as pgv
        except ImportError:
            return

        try:
            viz = pgv.AGraph(directed=True, strict=False)
            viz.graph_attr.update(rankdir='TB', size='16,12', dpi='150')
            viz.node_attr.update(shape='box', style='rounded,filled',
                               fillcolor='lightblue', fontsize='10')

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

        except Exception as e:
            self.logger.error(f"Failed to create graph visualization: {e}")

    def _restart_exploration(self, iteration: int) -> None:
        """Reset to starting URL while preserving accumulated knowledge"""
        self.logger.info(f"[RESTART] Returning to starting page: {self.starting_url}")
        self.url_stack = [self.starting_url]
        self.current_url = self.starting_url
        self.current_depth = 1

    def explore(self) -> str:
        """Execute main exploration loop"""
        start_iteration = self.current_iteration + 1
        self.logger.info(f"Beginning exploration: iterations {start_iteration} to {self.max_iterations}")

        for iteration in range(start_iteration, self.max_iterations + 1):
            if self.shutdown_called: break
            self.current_iteration = iteration

            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"Iteration {iteration}/{self.max_iterations}")
            self.logger.info(f"Depth: {self.current_depth}/{self.max_depth}")
            self.logger.info(f"URL: {self.current_url}")
            self.logger.info(f"{'='*80}")

            content, links = self.perceive()
            if not content:
                self._backtrack()
                if iteration % self.checkpoint_interval == 0:
                    self._save_checkpoint(iteration)
                continue

            insights = self.think(content)

            if iteration < self.max_iterations:
                next_url = self.act(links)

            if iteration % self.save_graph_interval == 0 or iteration == self.max_iterations:
                self._save_graph_data(iteration)
                self._draw_graph_visualization(iteration)

            if iteration % self.checkpoint_interval == 0 or iteration == self.max_iterations:
                self._save_checkpoint(iteration)

        self._save_graph_data(self.current_iteration)
        self._draw_graph_visualization(self.current_iteration)
        self._save_checkpoint(self.current_iteration)

        self.logger.info(f"\nExploration complete: visited {len(self.visited_urls)} pages")
        return self._synthesize_artifact()

    def _execute_queries(self, queries: List[str]) -> List[Tuple[str, str]]:
        """Execute queries against KB"""
        qa_pairs = []
        for query in queries:
            try:
                answer = self.kb_manager.query(query, top_k=self.top_k)
                if answer:
                    qa_pairs.append((query, answer))
            except Exception as e:
                self.logger.error(f"KB query failed for '{query}': {e}")
        return qa_pairs

    def _generate_next_query(self, previous_queries: List[str],
                             previous_responses: List[str]) -> Optional[str]:
        """Generate next synthesis query"""
        max_context_pairs = 20
        recent_queries = previous_queries[-max_context_pairs:]
        recent_responses = previous_responses[-max_context_pairs:]

        context = "\n\n".join([f"Q: {q}\nA: {r}"
                              for q, r in zip(recent_queries, recent_responses)])

        prompt = f"""Previous exploration queries and responses:
{context}

Based on the insights gathered so far, what is the next most important question to ask
to deepen understanding and reveal emergent patterns? The question should:
- Build on previous insights rather than repeat them
- Seek connections between different themes
- Identify gaps or contradictions to explore
- Move toward synthesis and creation rather than enumeration

Respond with JSON:
{{
    "query": "<your next question>",
    "reason": "<why this question deepens understanding>"
}}"""

        try:
            response = self.chat_completion(prompt, response_format={"type": "json_object"})
            result = self.parse_json_response(response)
            if result and "query" in result:
                return result["query"]
        except Exception as e:
            self.logger.error(f"Query generation failed: {e}")
        return None

    def _generate_synthesis_qa_pairs(self, mode: str) -> List[Tuple[str, str]]:
        """Generate Q&A pairs by querying KB"""
        if mode == "classic":
            queries = [
                "What are the central themes and patterns discovered?",
                "What unexpected connections or insights emerged?",
                "What contradictions or tensions were revealed?",
                "What questions remain open or were raised?",
                "What novel perspective emerged from this exploration?"
            ]
            return self._execute_queries(queries)

        # Iterative mode
        queries = ["What are the central themes and patterns discovered across all sources?"]
        answers = []

        for i in range(self.synthesis_iterations):
            self.logger.info(f"[SYNTHESIS {i+1}/{self.synthesis_iterations}] {queries[-1]}")

            answer = self.kb_manager.query(queries[-1], top_k=self.top_k)
            if not answer:
                break
            answers.append(answer)

            if i < self.synthesis_iterations - 1:
                next_query = self._generate_next_query(queries, answers)
                if not next_query:
                    break
                queries.append(next_query)

        return list(zip(queries, answers))

    def _synthesize_artifact(self) -> Dict[str, str]:
        """Generate final synthesis from accumulated insights"""
        mode = "iterative" if self.iterative_synthesis else "classic"
        self.logger.info(f"[SYNTHESIS] Using {mode} mode")

        if self.kb_manager.size() == 0:
            return {"abstract": "", "artifact": "No insights collected during exploration."}

        qa_pairs = self._generate_synthesis_qa_pairs(mode)
        if not qa_pairs:
            return {"abstract": "", "artifact": "Unable to generate synthesis questions."}

        perspectives = "\n\n\n".join([f"({i+1}) Question: {q}\n\nAnswer: {a}"
                                     for i, (q, a) in enumerate(qa_pairs)])

        prompt = f"""You explored {len(self.visited_urls)} sources and gathered {self.kb_manager.size()} insights.

Key patterns emerged through querying and analyzing gathered insights
{perspectives}

Drawing heavily upon the key patterns that emerged from the insights, create a novel, exciting, and thought provoking artifact with two parts:

1. **Abstract** (100-150 tokens):
    - Summary of core discovery and its significance

2. **Artifact** (1000-3000 tokens):
    - Emergent patterns not visible in individual sources
    - Novel discoveries and unexpected connections
    - Tensions, contradictions, or open questions
    - New directions or perspectives

IMPORTANT: Try to keep the artifact easy to understand and use simple English as much as possible

Respond with valid JSON only:
{{
    "abstract": "<abstract text>",
    "artifact": "<artifact text>"
}}"""

        try:
            response = self.chat_completion(prompt, response_format={"type": "json_object"})
            result = self.parse_json_response(response)
            if not result or "abstract" not in result or "artifact" not in result:
                raise ValueError("Missing required keys in response")
        except Exception as e:
            self.logger.error(f"Synthesis generation failed: {e}")
            return {"abstract": "Synthesis failed.", "artifact": f"Error: {e}"}

        result["metadata"] = {
            "pages_visited": len(self.visited_urls),
            "insights_collected": self.kb_manager.size(),
            "synthesis_mode": mode,
            "queries_used": len(qa_pairs),
            "max_depth": self.current_depth,
            "starting_url": self.starting_url
        }

        self._save_synthesis_outputs(result)
        return result

    def _save_synthesis_outputs(self, result: Dict) -> None:
        """Save synthesis in JSON and text formats"""
        timestamp = datetime.now().strftime("%m%d%H%M")
        base_path = os.path.join(self.get_repo(), f"{self.get_id()}.synthesis.{timestamp}")

        try:
            with open(f"{base_path}.json", 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=4, ensure_ascii=False)

            with open(f"{base_path}.txt", 'w', encoding='utf-8') as f:
                f.write(f"ABSTRACT:\n{result['abstract']}\n\n")
                f.write(f"ARTIFACT:\n{result['artifact']}\n\n")
                f.write(f"METADATA:\n{json.dumps(result['metadata'], indent=2)}")
        except Exception as e:
            self.logger.error(f"Failed to save synthesis: {e}")

    def shutdown(self) -> None:
        """Clean up CaesarAgent resources with immediate flag setting"""
        if self.shutdown_called:
            return

        # Set flags IMMEDIATELY to stop loops
        self.shutdown_called = True

        # Cleanup in reverse order of initialization
        if hasattr(self, 'kb_manager'):
            self.kb_manager.shutdown()

        # Call parent shutdown
        super().shutdown()

        self.logger.info("CaesarAgent shutdown completed")
