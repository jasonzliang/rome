"""Caesar Agent - Web exploration agent with graph-based navigation"""
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse
import copy
import io
import requests
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import networkx as nx
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rome.base_agent import BaseAgent
from rome.config import set_attributes_from_config, DEFAULT_CONFIG, SHORT_SUMMARY_LEN, SUMMARY_LENGTH, LONG_SUMMARY_LEN, LONGER_SUMMARY_LEN, LONGEST_SUMMARY_LEN
from rome.logger import get_logger
from rome.kb_client import ChromaClientManager

from .brave_search import BraveSearch
from .caesar_config import MAX_NUM_LINKS, MAX_NUM_VISITED_LINKS, MAX_NUM_NEIGHBORS, MAX_TEXT_LENGTH, REQUESTS_TIMEOUT, REQUESTS_HEADERS, CAESAR_CONFIG


class CaesarAgent(BaseAgent):
    """Veni, Vidi, Vici - Web exploration agent with checkpointing support"""
    def __init__(self, name: str = None, role: str = None,
             repository: str = None, config: Dict = None,
             starting_url: str = None, allowed_domains: List[str] = None):

        # Prepare merged config BEFORE calling super().__init__()
        merged_config = self._prepare_caesar_config(config, starting_url, allowed_domains)
        # Pass the merged config to BaseAgent (this configures the logger)
        if not role: role = self._get_default_role()
        super().__init__(name, role, repository, merged_config)

        # NOW setup Caesar-specific attributes (logger is configured)
        self.caesar_config = self.config.get('CaesarAgent', {})
        set_attributes_from_config(self, self.caesar_config, CAESAR_CONFIG['CaesarAgent'].keys())

        self._setup_allowed_domains()
        self._setup_brave_search()
        self._setup_knowledge_base()
        self._update_role()

        self._setup_exploration_state()
        if self._load_checkpoint():
            self.logger.info("Resumed from checkpoint")
        else:
            self.logger.info("Starting fresh exploration")

        self._validate_caesar_config()
        self._validate_starting_url()
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
        """Configure allowed domains"""
        if not self.allowed_domains:
            if self.starting_query:
                self.allowed_domains = ["*"]
                self.logger.info("Starting query detected - allowing ALL domains")
            elif self.starting_url:
                netloc = urlparse(self.starting_url).netloc
                self.allowed_domains = [netloc] if netloc else ["*"]
                self.logger.info(f"Auto-extracted domain: {netloc}" if netloc else "Cannot extract domain - allowing ALL domains")
            else:
                raise ValueError(
                    "Must provide starting_url, starting_query, and/or allowed_domains")

        self.allow_all_domains = "*" in self.allowed_domains
        if self.allow_all_domains:
            self.logger.info("Wildcard '*' detected - allowing ALL domains")

    def _setup_brave_search(self) -> None:
        """Setup brave search"""
        self.web_searches_used = 0
        if self.starting_query:
            old_starting_url = self.starting_url
            search_engine = BraveSearch(agent=self, config=self.config.get("BraveSearch", {}))

            # Generate additional queries if requested
            queries = [self.starting_query]
            if self.additional_starting_queries > 0:
                try:
                    prompt = f"""Given this query: "{self.starting_query}"

Generate anywhere from 0 to {self.additional_starting_queries} additional search queries that would help comprehensively answer the original query. These queries should:
- Explore different aspects or angles of the original query
- Cover related concepts that provide essential context
- Include specific technical or domain-specific variations
- Be concise (1-6 words each for optimal search results)

IMPORTANT: Use your role as a guide on how to respond!
IMPORTANT: If no additional queries are generated, return an empty list

Respond with valid JSON only:
{{
    "queries": ["query1", "query2", ...]
}}"""

                    response = self.chat_completion(
                        prompt,
                        response_format={"type": "json_object"}
                    )
                    result = self.parse_json_response(response)
                    if result and result.get("queries"):
                        additional = result["queries"][:self.additional_starting_queries]
                        queries.extend(additional)
                        self.logger.info(f"Generated {len(additional)} additional queries: {additional}")
                except Exception as e:
                    self.logger.error(f"Failed to generate additional queries: {e}")

            # Execute search with all queries
            self.starting_url = search_engine.search_and_save(queries)
            self.web_searches_used += len(queries)
            self.logger.info(f"Overwriting existing starting_url ({old_starting_url}) with {len(queries)} query search results")

    def _setup_knowledge_base(self) -> None:
        """Setup knowledge base"""
        self.kb_manager = ChromaClientManager(agent=self)
        self.logger.info(f"Knowledge base initialized: {self.kb_manager.info()}")

    def _update_role(self):
        """Adapt agent role based on starting URL content and optional insights"""
        try:
            # Check overwrite (highest priority)
            if self.overwrite_role_file and os.path.exists(self.overwrite_role_file):
                with open(self.overwrite_role_file, 'r', encoding='utf-8') as f:
                    if role := f.read().strip():
                        self.role = role
                        self.logger.info(f"[OVERWRITE ROLE] Using overwritten role:\n{self.role}")

            # Early return if adaptation disabled
            if not self.adapt_role: return

            # Check cached adapted role
            role_file = os.path.join(self.get_log_dir(), f"{self.get_id()}.updated_role.txt")
            if os.path.exists(role_file) and os.path.getsize(role_file) > 0:
                with open(role_file, 'r', encoding='utf-8') as f:
                    if cached_role := f.read().strip():
                        self.role = cached_role
                        self.logger.info(f"[ADAPT ROLE] Using cached adapted role:\n{self.role}")
                        return

            # Fetch and extract content
            self.logger.info(f"[ADAPT ROLE] Analyzing {self.starting_url}")
            html = self._fetch_html(self.starting_url)
            content = self._extract_text_from_html(html) if html else ""
            if not content: return

            # Load insights if available
            insights = ""
            if self.adapt_role_file and os.path.exists(self.adapt_role_file):
                with open(self.adapt_role_file, 'r', encoding='utf-8') as f:
                    insights = f.read().strip()

            # Generate adapted role
            starting_query = f"\nSTARTING QUERY:\n{self.starting_query}\n" if self.starting_query else ""
            starting_query_task = " based on the starting query" if self.starting_query else ""

            insights_section = f"\nPRIOR INSIGHTS:\n{insights}\n" if insights else ""
            insights_info = " and prior insights" if insights else ""
            insights_task = " - Builds upon themes and gaps identified in prior insights"

            prompt = f"""You are adapting your current role based on the following starting content{insights_info}.
{starting_query}
STARTING URL:
{self.starting_url}

STARTING CONTENT:
{content}
{insights_section}
CURRENT ROLE:
{self.role}

YOUR TASK:
Analyze the page content{insights_info} to create a specialized role that:
 - Improves upon core exploration philosophy
 - Creates an overall goal for the agent to strive for{starting_query_task}
 - Focuses exploration toward most promising areas revealed by the page content
{insights_task}

Provide an adapted role description (300-500 tokens) that is creative, innovative, and original!

IMPORATNT: Your response must start with "Your role:" followed by the adapted role description."""

            if not (adapted_role := self.chat_completion(prompt).strip()) or len(adapted_role) < 50:
                self.logger.error("[ADAPT ROLE] Invalid LLM response, keeping default role")
                return

            # Save and apply
            self.role = adapted_role
            with open(role_file, 'w', encoding='utf-8') as f:
                f.write(self.role)
            self.logger.info(f"[ADAPT ROLE] Using newly adapted role:\n{self.role}")

        except Exception as e:
            self.logger.error(f"[ADAPT ROLE] Role adaptation failed: {e}, keeping default role")

    def _setup_exploration_state(self) -> None:
        """Initialize exploration state"""
        self.graph = nx.DiGraph()
        self.visited_urls = {}
        self.failed_urls = set()
        self.url_stack = [self.starting_url]
        self.current_url = self.starting_url
        self.current_depth = len(self.url_stack)
        self.current_iteration = 0

    def _validate_caesar_config(self) -> None:
        """Validate Caesar-specific configuration"""
        self.logger.assert_true(self.starting_url, "starting_url required")
        self.logger.assert_true(self.allowed_domains, "allowed_domains required")
        self.logger.assert_true(self.max_iterations >= 0, "max_iterations must be >= 0")
        self.logger.assert_true(self.max_depth > 0, "max_depth must be > 0")

    def _validate_starting_url(self):
        """Validate starting URL (http/https/file)"""
        parsed = urlparse(self.starting_url)

        if parsed.scheme == 'file':
            file_path = parsed.path
            if not os.path.isfile(file_path):
                raise ValueError(f"File does not exist or not readable: {file_path}")
            if not os.access(file_path, os.R_OK):
                raise ValueError(f"File is not readable: {file_path}")
        elif parsed.scheme in ['http', 'https']:
            if not parsed.netloc:
                raise ValueError(f"Invalid URL - missing domain: {self.starting_url}")
        else:
            raise ValueError(f"URL must use http/https/file scheme: {self.starting_url}")

    def _log_initialization(self) -> None:
        """Log initialization summary"""
        self.logger.info(f"CaesarAgent '{self.name}' initialized")
        self.logger.info(f"Log dir: {self.get_log_dir()}")
        self.logger.info(f"Starting: {self.starting_url}")
        self.logger.info(f"Domains: {self.allowed_domains}")
        self.logger.info(f"Iterations: {self.max_iterations}, Depth: {self.max_depth}")

    def _get_checkpoint_path(self) -> str:
        """Get checkpoint file path"""
        return os.path.join(self.get_log_dir(), f"{self.get_id()}.checkpoint.json")

    def _save_checkpoint(self, iteration: int) -> None:
        """Save exploration state with optional graph data"""
        try:
            if not self.url_stack:
                error_msg = f"Cannot save checkpoint on iteration {iteration}: empty url_stack"
                raise RuntimeError(error_msg)

            checkpoint_data = {
                'role': self.role,
                'iteration': iteration,
                'current_url': self.current_url,
                'current_depth': self.current_depth,
                'url_stack': self.url_stack,
                'failed_urls': list(self.failed_urls),
                'visited_urls': self.visited_urls,
                'web_searches_used': self.web_searches_used,
                'graph': nx.node_link_data(self.graph, edges="edges"),
                'config': {
                    'starting_url': self.starting_url,
                    'starting_query': self.starting_query,
                    'allowed_domains': self.allowed_domains,
                    'max_iterations': self.max_iterations,
                    'max_depth': self.max_depth,
                },
                'timestamp': datetime.now().isoformat(),
            }

            # Save checkpoint
            with open(self._get_checkpoint_path(), 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=4, ensure_ascii=False)
            self.logger.info(f"Checkpoint saved on iteration {iteration}")

            # Save separate graph file if at interval
            if iteration % self.save_graph_interval == 0 or iteration == self.max_iterations:
                knowledge_graph = checkpoint_data['graph']
                knowledge_graph['iteration'] = iteration
                knowledge_graph['starting_url'] = self.starting_url
                with open(os.path.join(self.get_repo(),
                    f"{self.get_id()}.graph_iter{iteration}.json"), 'w', encoding='utf-8') as f:
                    json.dump(knowledge_graph, f, indent=4, ensure_ascii=False)
                self.logger.info(f"Knowledge graph saved on iteration {iteration}")

        except Exception as e:
            self.logger.error(f"Failed to save checkpoint on iteration {iteration}: {e}")

    def _load_checkpoint(self) -> bool:
        """Load exploration state from checkpoint"""
        checkpoint_path = self._get_checkpoint_path()
        if not os.path.exists(checkpoint_path):
            return False

        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Check config changes
            config = data.get('config', {})
            if config.get('starting_query') != self.starting_query:
                self.logger.error(f"Checkpoint starting_query mismatch: '{config.get('starting_query')}' vs '{self.starting_query}'")
            if config.get('starting_url') != self.starting_url:
                self.logger.error(f"Checkpoint starting_url mismatch")
            if config.get('allowed_domains') != self.allowed_domains:
                self.logger.error(f"Checkpoint allowed_domains mismatch: {config.get('allowed_domains')} vs {self.allowed_domains}")

            # Restore state
            self.current_iteration = data.get('iteration', self.current_iteration)
            self.visited_urls = data.get('visited_urls', self.visited_urls)
            self.url_stack = data.get('url_stack', self.url_stack)
            self.web_searches_used = data.get('web_searches_used', self.web_searches_used)

            # Disabled due to having separating loading mechanism or not necessary
            # self.failed_urls = set(data.get('failed_urls', self.failed_urls))
            # self.role = data.get('role', self.role)

            if not self.url_stack:
                self.logger.error("Invalid checkpoint: empty url_stack")
                return False

            self.current_depth = len(self.url_stack)
            self.current_url = self.url_stack[-1]

            # Restore graph inline
            self.graph = nx.node_link_graph(data.get('graph', self.graph), edges="edges")

            self.logger.info(f"Checkpoint loaded from {data.get('timestamp')}")
            # time.sleep(2)
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
        """Fetch HTML or PDF content from URL or local file"""
        try:
            # Local file handling
            if url.startswith('file://'):
                file_path = url[7:]
                if file_path.lower().endswith('.pdf'):
                    with open(file_path, 'rb') as f:
                        text = '\n\n'.join(page.extract_text() for page in PdfReader(f).pages)
                    return f"<html><body><div>{text}</div></body></html>"
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()

            # Remote URL handling
            response = requests.get(url, timeout=REQUESTS_TIMEOUT,
                headers=REQUESTS_HEADERS, allow_redirects=True)
            response.raise_for_status()

            # PDF detection and extraction
            is_pdf = ('application/pdf' in response.headers.get('content-type', '').lower() or
                      response.content[:4] == b'%PDF')
            if is_pdf:
                text = '\n\n'.join(page.extract_text()
                                  for page in PdfReader(io.BytesIO(response.content)).pages)
                return f"<html><body><div>{text}</div></body></html>"

            return response.text

        except Exception as e:
            self.logger.error(f"Fetch failed for {url}: {e}")
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

            if url not in seen and
                self._is_allowed_url(url) and
                url not in self.failed_urls and
                self.visited_urls.get(url, 0) <= self.max_allowed_revisits and
                url.startswith('http'):
                links.append((url, text))
                seen.add(url)

        return links

    def _extract_text_from_html(self, html: str, max_length: int = MAX_TEXT_LENGTH) -> str:
        """Extract clean text content from HTML"""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            text = soup.get_text(separator=' ', strip=True)
            return ' '.join(text.split())[:max_length]
        except Exception as e:
            self.logger.error(f"HTML text extraction failed: {e}")
            return ""

    def perceive(self) -> Tuple[str, List[Tuple[str, str]]]:
        """Phase 1: Extract content and links from current page"""
        self.logger.info(f"[PERCEIVE] {self.current_url}")

        try:
            html = self._fetch_html(self.current_url)
            if not html:
                self.failed_urls.add(self.current_url)
                return "", []

            text = self._extract_text_from_html(html)
            links = self._extract_links(html, self.current_url)
            return text, links

        except Exception as e:
            self.logger.error(f"Perceive phase failed: {e}")
            self.failed_urls.add(self.current_url)
            return "", []

    def think(self, content: str) -> str:
        """Phase 2: Analyze content and extract insights"""
        self.logger.info("[THINK] Analyzing content")
        if not content: return ""

        prev_insights = ''; related_insights = ''
        if self.current_url in self.graph.nodes:
            prev_insights = self.graph.nodes[self.current_url].get('insights', '')

            # Get neighbor URLs (not node dicts)
            neighbors = (set(self.graph.successors(self.current_url)) |
                         set(self.graph.predecessors(self.current_url))) - {self.current_url}

            # Get insights from neighbor nodes
            related_insights = [
                (n, self.graph.nodes[n].get('insights', ''))
                for n in neighbors if n in self.graph.nodes
                and self.graph.nodes[n].get('insights')
            ]
            related_insights = "\n\n".join(
                f"[{i+1}] Source: {url}\n{insight}"
                for i, (url, insight) in enumerate(related_insights[:MAX_NUM_NEIGHBORS])
            )

        query_task = "- How to answer the query\n" if self.starting_query else ""
        prev_insight_task = "- How this builds upon or challenges previous/related insights" if (prev_insights or related_insights) else ""

        prompt = f"""CONTENT:
{content}

QUERY:
{self.starting_query if self.starting_query else 'No query is available'}

PREVIOUS INSIGHTS:
{prev_insights if prev_insights else 'No previous insights available'}

RELATED INSIGHTS:
{related_insights if related_insights else 'No related insights available'}

YOUR TASK:
Analyze this content and extract key insights focusing on:
- Novel patterns or unexpected connections
- Assumptions being made and alternative perspectives
- Interesting questions raised by the content
{query_task}{prev_insight_task}

IMPORTANT: Use your role as a guide on how to respond!

Depending on the complexity of the content, provide anywhere from 1 to 6 concise but substantive insights, but do not exceed ~800 tokens in total length:"""

        try:
            insights = self.chat_completion(
                prompt,
                override_config=self.exploration_llm_config
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
        self.graph.add_node(self.current_url, insights=insights,
            depth=self.current_depth, iteration=self.current_iteration,
            visit_count=self.visited_urls[self.current_url])
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
        else:
            self.logger.error(
                f"Cannot backtrack, no parent link for {self.current_url} (url_stack size <= 1)")
        return False

    def _get_parent_url(self):
        """Get url to parent page"""
        return self.url_stack[-2] if len(self.url_stack) > 1 else None

    def _determine_exploration_strategy(self, kb_context, memory_context) -> Dict:
        """Chooses best exploration strategy based on current knowledge and memory"""
        if self.web_searches_used < self.max_web_searches:
            web_search_option = f"\n4. **WEB_SEARCH** relevant topics to address current exploration insights (remaining uses: {self.max_web_searches - self.web_searches_used})"
        else:
            web_search_option = f"\n4. **WEB_SEARCH** (do NOT pick, no uses remaining)"

        prompt = f"""Based on accumulated knowledge and navigation patterns, determine the optimal exploration strategy.

CURRENT EXPLORATION CONTEXT:
- Current iteration: {self.current_iteration}/{self.max_iterations}
- Current depth: {self.current_depth}/{self.max_depth}
- Web pages visited: {len(self.visited_urls)}
- Average visits per page: {float(np.mean(list(self.visited_urls.values())))}
- Current URL: {self.current_url}

CURRENT EXPLORATION INSIGHTS:
{kb_context if kb_context else "No exploration insights available."}

HISTORICAL NAVIGATION PATTERNS:
{memory_context if memory_context else "No exploration history available."}

Analyze whether the agent should:
1. **REVISIT** previously visited pages to deepen understanding of relevant known areas
2. **EXPLORE** new un-visited pages to discover novel information or knowledge
3. **BACKTRACK** to the immediate previously visited page to try alternative paths{web_search_option}

Consider:
- Knowledge gaps vs areas of saturation
- Depth of current exploration branch
- Success patterns from previous decisions
- Risk/reward of new exploration vs consolidation

IMPORTANT: Select WEB_SEARCH if exploration has stagnated or to increase exploration diversity, but only if there are still uses remaining
IMPORTANT: Use your role as a guide on how to respond!

Respond with a JSON object in this exact format:
{{
    "action": "choose one action from EXPLORE, REVISIT, BACKTRACK, or WEB_SEARCH",
    "reasoning": "strategic recommendation of exploration strategy in 2-3 sentences",
    "search_query": "short query to find relevant topics or N/A if not WEB_SEARCH"
}}"""

        try:
            response = self.chat_completion(prompt,
                override_config=self.exploration_llm_config,
                response_format={"type": "json_object"})
            data = self.parse_json_response(response)
            if not data or "action" not in data or "reasoning" not in data or \
                "search_query" not in data:
                raise ValueError("Missing required keys in response")
            return data
        except Exception as e:
            self.logger.error(f"Exploration strategy determination failed: {e}")
            return ""

    def _get_web_search_links(self, query: str) -> List[Tuple[str, str]]:
        """Execute web search and return links"""
        if self.web_searches_used >= self.max_web_searches:
            self.logger.error(f"Web search limit reached during exploration: {self.max_web_searches}")
            return []

        try:
            search_engine = BraveSearch(agent=self, config=self.config.get("BraveSearch", {}))
            search_results_url = search_engine.search_and_save([query])
            self.web_searches_used += 1

            html = self._fetch_html(search_results_url)
            if not html: return []

            links = self._extract_links(html, search_results_url)
            self.logger.info(f"Web search '{query}' returned {len(links)} links")
            return links

        except Exception as e:
            self.logger.error(f"Web search during exploration failed for '{query}': {e}")
            return []

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
            if self.starting_query:
                kb_context = self.kb_manager.query(f"In order to answer this query ({self.starting_query}), what should we explore next?")
            else:
                kb_context = self.kb_manager.query(
                    "What patterns, gaps, or questions have emerged from our knowledge? What should we explore next?")

            memory_context = self.recall(
                f"What webpages have I frequently visited and has exploration stagnated? What navigation patterns have emerged in relation to the following insights:\n{kb_context}")

            if self.use_explore_strategy:
                strat = self._determine_exploration_strategy(kb_context, memory_context)
                if strat['action'] == "WEB_SEARCH":
                    explore_strategy = ""
                    web_search_links = self._get_web_search_links(strat['search_query'])
                    if web_search_links: links = web_search_links
                else:
                    explore_strategy = f"{strat['reasoning']}"

        except Exception as e:
            self.logger.error(f"KB/memory/explore for link selection failed: {e}")

        link_options, url_map = self._format_link_options(links)

        prompt = f"""You are selecting the next webpage link to explore

CURRENT EXPLORATION CONTEXT:
- Current iteration: {self.current_iteration}/{self.max_iterations}
- Current depth: {self.current_depth}/{self.max_depth}
- Web pages visited: {len(self.visited_urls)}
- Average visits per page: {float(np.mean(list(self.visited_urls.values())))}
- Current URL: {self.current_url}

CURRENT EXPLORATION INSIGHTS:
{kb_context if kb_context else "No exploration insights available."}

HISTORICAL EXPLORATION PATTERNS:
{memory_context if memory_context else "No exploration history available."}

EXPLORATION STRATEGY:
{explore_strategy if explore_strategy else "No exploration strategy available"}

AVAILABLE PATHS FORWARD:
{'\n'.join(link_options)}

TASK: Based on current exploration context/insights, historical exploration patterns, and exploration strategy, which page link is the most interesting, deepens understanding, and offers the most promising direction to explore?

IMPORTANT: Use your role as a guide on how to respond!

Respond with a JSON object in this exact format:
{{
    "choice": <number from 1 to {len(url_map)}>,
    "reason": "<brief explanation of why this path is promising>"
}}

Your response must be valid JSON only, nothing else."""

        try:
            response = self.chat_completion(
                prompt,
                override_config=self.exploration_llm_config,
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
        if not links and len(self.url_stack) == 1:
            self.logger.error("[ACT] No links at starting_url - exploration exhausted")
            return ""
        if self.current_depth > self.max_depth or not links:
            self._backtrack(); return self.current_url

        next_url, reason = self._select_next_link(links)
        self.logger.info(f"[ACT] Selected link: {next_url}")
        self.logger.info(f"Reason: {reason}")

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
            f"Navigation performed on iteration {self.current_iteration} at depth {self.current_depth}. "
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
        return next_url

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
                               fillcolor='lightblue', fontsize='8')
            viz.edge_attr.update(fontsize='8')

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

    def _restart_exploration(self) -> None:
        """Reset to starting URL while preserving accumulated knowledge"""
        self.logger.info(f"[RESTART] Returning to starting page: {self.starting_url}")
        self.url_stack = [self.starting_url]
        self.current_url = self.starting_url
        self.current_depth = len(self.url_stack)

    def explore(self) -> str:
        """Execute main exploration loop"""
        start_iteration = self.current_iteration + 1
        self.logger.info(f"[EXPLORE] Beginning exploration: iterations {start_iteration} to {self.max_iterations}")

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
                if not next_url:
                    self.logger.error("[EXPLORE] No links to explore, exiting loop early")
                    break

            if iteration % self.save_graph_interval == 0 or iteration == self.max_iterations:
                self._draw_graph_visualization(iteration)

            if iteration % self.checkpoint_interval == 0 or iteration == self.max_iterations:
                self._save_checkpoint(iteration)

        self._draw_graph_visualization(self.current_iteration)
        self._save_checkpoint(self.current_iteration)

        self.logger.info(f"\n[EXPLORE] Exploration complete: visited {len(self.visited_urls)} pages")
        return self._synthesize_artifact()

    def _generate_qa_pairs_classic(self, queries: List[str]) -> List[Tuple[str, str]]:
        """Execute queries against KB"""
        qa_pairs = []
        for query in queries:
            try:
                answer = self.kb_manager.query(query,
                    top_k=self.synthesis_top_k, top_n=self.synthesis_top_n)
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
            response = self.chat_completion(prompt, response_format={"type": "json_object"})
            result = self.parse_json_response(response)
            if result and "query" in result:
                return result["query"]
        except Exception as e:
            self.logger.error(f"Query generation failed: {e}")
        return None

    def _generate_qa_pairs(self, mode: str) -> List[Tuple[str, str]]:
        """Generate Q&A pairs by querying KB"""
        queries = [
            "What are the central themes and patterns discovered?",
            "What unexpected connections or insights emerged?",
            "What contradictions or tensions were revealed?",
            "What questions remain open or were raised?",
            "What novel perspective emerged from this exploration?"
        ]
        if self.starting_query is not None:
            queries = [self.starting_query] + queries
        if mode == "classic":
            return self._generate_qa_pairs_classic(queries)

        # Iterative mode
        queries = [queries[0]]
        answers = []

        for i in range(self.synthesis_iterations):
            answer = self.kb_manager.query(queries[-1],
                top_k=self.synthesis_top_k, top_n=self.synthesis_top_n)
            if not answer: break
            answers.append(answer)

            self.logger.info(f"[SYNTHESIS {i+1}/{self.synthesis_iterations}]\nQ: {queries[-1]}\nA: {answers[-1]}")

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

        qa_pairs = self._generate_qa_pairs(mode)
        if not qa_pairs:
            return {"abstract": "", "artifact": "Unable to generate synthesis questions."}

        perspectives = "\n\n\n".join([f"({i+1}) Question: {q}\n\nAnswer: {a}"
                                     for i, (q, a) in enumerate(qa_pairs)])

        starting_query_task = f" that creatively answers this query - {self.starting_query}" if self.starting_query else ""

        prompt = f"""You explored {len(self.visited_urls)} sources and gathered {self.kb_manager.size()} insights.

Key patterns emerged through querying and analyzing gathered insights
{perspectives}

YOUR TASK:
Drawing heavily upon the key patterns that emerged from the insights, create a novel, exciting, and thought provoking artifact{starting_query_task}:

1. **Artifact Abstract** (100-150 tokens):
    - Summary of the artifact's core discovery and its significance

2. **Artifact Main Text** (up to ~3000 tokens):
    - Emergent patterns not visible in individual sources
    - Novel discoveries and unexpected connections
    - Tensions, contradictions, or open questions
    - New directions or perspectives

IMPORTANT: Avoid excessive jargon while keeping it logical, easy to understand, and convincing to a skeptical reader
IMPORTANT: Use your role as a guide on how to respond!

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
            "synthesis_queries": len(qa_pairs),
            "max_depth": self.current_depth,
            "starting_url": self.starting_url,
            "starting_query": self.starting_query,
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
                f.write(f"METADATA:\n{json.dumps(result['metadata'], indent=4)}")
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