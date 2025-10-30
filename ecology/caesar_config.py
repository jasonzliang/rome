# Maximum number of links to consider when selecting next webpage to go to
MAX_NUM_LINKS = 2000
# Maximum number of visited links to consider when selecting next webpage
MAX_NUM_VISITED_LINKS = 1000
# Maximum of neighboring nodes to look for related insights
MAX_NUM_NEIGHBORS = 10
# Timeout for fetching webpage html using requests
REQUESTS_TIMEOUT = 10
# Headers to use for requests when fetching html
REQUESTS_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br, zstd',
    'Referer': 'https://www.google.com/',
    'DNT': '1',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
    'Sec-Fetch-Dest': 'document',
    'Sec-Fetch-Mode': 'navigate',
    'Sec-Fetch-Site': 'cross-site',
    'Sec-Fetch-User': '?1',
    'Sec-Ch-Ua': '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
    'Sec-Ch-Ua-Mobile': '?0',
    'Sec-Ch-Ua-Platform': '"Windows"',
    'Cache-Control': 'max-age=0',
}

CAESAR_CONFIG = {
    "CaesarAgent": {
        # Total number of pages to explore before stopping
        "max_iterations": 10,
        # Maximum depth of exploration tree before backtracking
        "max_depth": 10000,
        # Initial URL to begin exploration (enabling means start_query must be None)
        "starting_url": "https://en.wikipedia.org/wiki/Main_Page",
        # Initial query to being exploration (enabling means starting_url must be None)
        "starting_query": None,
        # Domains to allow exploration; empty list uses starting_url domain; use ["*"] to allow any
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
        # LLM config for agent's ACT/THINK phases to encourage exploration
        "exploration_llm_config": {
            "model": "gpt-4o",
            "reasoning_effort": "low",
            "temperature": 0.9,
            "max_completion_tokens": 5000,
            "timeout": 120,
        },

        # Set to False for classic mode (ask all queries at once)
        "iterative_synthesis": True,
        # Number of Q/A iterations
        "synthesis_iterations": 10,
        # Top_k for synthesis query and retrieval from DB
        "synthesis_top_k": 50,
        # Top_n for synthesis query and retrieval from DB
        "synthesis_top_n": 10,

        # Overwrites the default role with new role from file, order is [overwrite] -> [adapt]
        "overwrite_role_file": None,
        # Whether to modify agent role based on starting URL and/or insights
        "adapt_role": False,
        # Insights file path used to adapt the role and change it
        "adapt_role_file": None
    },

    "AgentMemory": {
        # Whether to enable agent memory
        "enabled": True,
        # Whether to use vector DB or graph DB
        "use_graph": False,
    },

    "BraveSearch": {
        # Number of search results to fetch
        "num_results": 20,
        # Number of tries to attempt to search
        "max_retries": 3,
        # Delay between search retries
        "retry_delay": 1,
        # Request timeout in seconds
        "timeout": 30,
    },

    # Default config for LLM outside of agent exploration
    "OpenAIHandler": {
        # Model name for LLM
        "model": "gpt-4o",
        # Reasoning effort for GPT-5/O models
        "reasoning_effort": "medium",
        # Base temperature for LLM (overridden by exploration_llm_config for ACT/THINK)
        "temperature": 0.1,
        # Maximum tokens per LLM response
        "max_completion_tokens": 10000,
        # API timeout in seconds
        "timeout": 240,
    }
}