# web_explorer_agent.py
import networkx as nx
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import requests

from .agent import Agent
from .config import DEFAULT_CONFIG, set_attributes_from_config
from .logger import get_logger

# Focused configuration
WEB_EXPLORER_CONFIG = {
    **DEFAULT_CONFIG,

    "CaesarAgent": {
        "max_iterations": 10,
        "starting_url": None,  # Must be provided
        "allowed_domains": [],  # Whitelist of domains
        "max_depth": 3,  # Maximum link depth
        "insight_prompt": "Extract 2-3 key insights from this content focusing on factual information.",
        "navigation_prompt": "Based on the insights gathered, which link is most relevant to continue research on: {topic}",
        "synthesis_prompt": "Synthesize the collected insights into a structured summary.",
    },

    "OpenAIHandler": {
        **DEFAULT_CONFIG["OpenAIHandler"],
        "model": "gpt-4o-mini",  # Cost-effective for exploration
        "temperature": 0.3,  # Moderate creativity
        "max_tokens": 2048,
    }
}


class CaesarAgent(Agent):
    """
    Supervised web research agent with safety constraints.
    Explores web content within defined boundaries to gather insights.
    """

    def __init__(self, name: str = None, role: str = None,
                 repository: str = None, config: Dict = None,
                 starting_url: str = None, allowed_domains: List[str] = None):

        # Merge configs
        self.config = {**WEB_EXPLORER_CONFIG, **(config or {})}

        # Initialize logger first
        self.logger = get_logger()

        # Core setup
        self._setup_config(self.config)
        self._validate_name_role(name, role)
        self._setup_repository_and_logging(repository)

        # Initialize components
        self._setup_openai_handler()
        self._setup_knowledge_base()

        # Web explorer specific setup
        explorer_config = self.config.get('CaesarAgent', {})
        set_attributes_from_config(self, explorer_config,
            ['max_iterations', 'max_depth', 'insight_prompt',
             'navigation_prompt', 'synthesis_prompt'],
            ['starting_url', 'allowed_domains'])

        # Override with constructor params
        if starting_url:
            self.starting_url = starting_url
        if allowed_domains:
            self.allowed_domains = allowed_domains

        # Validate required params
        self.logger.assert_true(self.starting_url,
            "starting_url must be provided")
        self.logger.assert_true(len(self.allowed_domains) > 0,
            "allowed_domains must contain at least one domain")

        # Initialize exploration state
        self.graph = nx.DiGraph()
        self.visited_urls = set()
        self.url_stack = [self.starting_url]
        self.current_url = self.starting_url
        self.current_depth = 0

        self.logger.info(f"CaesarAgent initialized: {self.name}")
        self.logger.info(f"Starting URL: {self.starting_url}")
        self.logger.info(f"Allowed domains: {self.allowed_domains}")

    def _is_allowed_url(self, url: str) -> bool:
        """Check if URL is within allowed domains"""
        parsed = urlparse(url)
        return any(domain in parsed.netloc for domain in self.allowed_domains)

    def _fetch_html(self, url: str) -> Optional[str]:
        """Safely fetch HTML content"""
        try:
            response = requests.get(url, timeout=10,
                                   headers={'User-Agent': 'ResearchBot/1.0'})
            response.raise_for_status()
            return response.text
        except Exception as e:
            self.logger.error(f"Failed to fetch {url}: {e}")
            return None

    def _extract_links(self, html: str, base_url: str) -> List[str]:
        """Extract and filter links from HTML"""
        soup = BeautifulSoup(html, 'html.parser')
        links = []

        for a_tag in soup.find_all('a', href=True):
            url = urljoin(base_url, a_tag['href'])
            if (self._is_allowed_url(url) and
                url not in self.visited_urls and
                url.startswith('http')):
                links.append(url)

        return links[:20]  # Limit to prevent overwhelming the agent

    def perceive(self) -> Tuple[str, List[str]]:
        """Phase 1: Fetch and parse current page"""
        self.logger.info(f"PERCEIVE: Fetching {self.current_url}")

        html = self._fetch_html(self.current_url)
        if not html:
            return "", []

        soup = BeautifulSoup(html, 'html.parser')

        # Extract text content
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text(separator='\n', strip=True)
        text = '\n'.join(line for line in text.split('\n') if line)[:5000]

        # Extract links
        links = self._extract_links(html, self.current_url)

        return text, links

    def think(self, content: str) -> str:
        """Phase 2: Analyze content and extract insights"""
        self.logger.info("THINK: Analyzing content")

        prompt = f"{self.insight_prompt}\n\nContent:\n{content}"
        insights = self.chat_completion(prompt)

        # Store in knowledge base
        self.kb_manager.add_text(
            insights,
            metadata={'url': self.current_url, 'depth': self.current_depth}
        )

        # Add to graph
        self.graph.add_node(self.current_url, insights=insights)

        self.logger.info(f"Extracted insights: {insights[:200]}...")
        return insights

    def act(self, links: List[str]) -> Optional[str]:
        """Phase 3: Choose next URL to explore"""
        if not links:
            self.logger.info("ACT: No links, backtracking")
            if len(self.url_stack) > 1:
                self.url_stack.pop()
                return self.url_stack[-1]
            self.logger.assert_true(False,
                "No links and cannot backtrack from starting page")

        # Query KB for guidance
        kb_advice = self.kb_manager.query(
            f"What topics should we explore next? Current: {self.current_url}",
            top_k=3
        ) if self.kb_manager.size() > 0 else "Explore systematically."

        # Use LLM to choose link
        link_list = '\n'.join(f"{i+1}. {url}" for i, url in enumerate(links))
        prompt = f"""Role: {self.role}

Previous insights: {kb_advice}

Available links:
{link_list}

Choose the most interesting link (respond with just the number 1-{len(links)}):"""

        response = self.chat_completion(prompt)

        try:
            choice = int(response.strip()) - 1
            next_url = links[choice]
            self.graph.add_edge(self.current_url, next_url)
            return next_url
        except (ValueError, IndexError):
            self.logger.warning(f"Invalid choice: {response}, using first link")
            return links[0]

    def explore(self) -> str:
        """Main exploration loop"""
        self.logger.info(f"Starting exploration for {self.max_iterations} iterations")

        for iteration in range(1, self.max_iterations + 1):
            self.logger.info(f"=== Iteration {iteration}/{self.max_iterations} ===")

            # Phase 1: Perceive
            content, links = self.perceive()
            if not content:
                break

            self.visited_urls.add(self.current_url)

            # Phase 2: Think
            insights = self.think(content)

            # Phase 3: Act
            if iteration < self.max_iterations:
                next_url = self.act(links)
                if next_url:
                    self.current_url = next_url
                    self.url_stack.append(next_url)
                    self.current_depth = len(self.url_stack) - 1

        # Generate final artifact
        return self._synthesize_artifact()

    def _synthesize_artifact(self) -> str:
        """Generate final synthesis from collected insights"""
        self.logger.info("Synthesizing final artifact")

        # Query KB multiple times for diverse perspectives
        queries = [
            "What are the main themes discovered?",
            "What surprising connections emerged?",
            "What questions remain unanswered?"
        ]

        perspectives = []
        for q in queries:
            response = self.kb_manager.query(q, top_k=5)
            perspectives.append(f"## {q}\n{response}\n")

        # Generate synthesis
        prompt = f"""{self.synthesis_prompt}

Exploration summary:
- URLs visited: {len(self.visited_urls)}
- Insights gathered: {self.kb_manager.size()}
- Max depth: {self.current_depth}

{' '.join(perspectives)}

Create a structured, insightful summary:"""

        artifact = self.chat_completion(prompt,
            override_config={'max_tokens': 4096})

        self.logger.info("Artifact generated successfully")
        return artifact
