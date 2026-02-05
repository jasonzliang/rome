# Maximum number of links to consider when selecting next webpage to go to
MAX_NUM_LINKS = 2000
# Maximum number of visited links to consider when selecting next webpage
MAX_NUM_VISITED_LINKS = 1000
# Maximum of neighboring nodes to look for related insights
MAX_NUM_NEIGHBORS = 5
# Maximum length (characters) of text to extract from a webpage
MAX_TEXT_LENGTH = 100000
# Timeout for fetching webpage html using requests
REQUESTS_TIMEOUT = 10
# Headers to use for requests when fetching html
REQUESTS_HEADERS = {
    # 1. Standard Chrome Accept Header
    # Must include image formats (avif, webp, apng) because real Chrome always requests them.
    # Missing these is a high-confidence bot signal when impersonating a browser.
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',

    # 2. Accept-Encoding (CRITICAL FIX)
    # You MUST include 'br' (Brotli) and 'zstd'.
    # Because you use impersonate="chrome", your TLS fingerprint claims you support these.
    # If you remove them here, the server sees a "TLS vs Header Mismatch" and blocks you instantly.
    # curl_cffi handles the decompression automatically, so this is safe to include.
    'Accept-Encoding': 'gzip, deflate, br, zstd',

    # 3. Standard Language & Cache
    'Accept-Language': 'en-US,en;q=0.9',
    'Cache-Control': 'max-age=0',

    # 4. Referer Strategy
    # Sets the "Previous Page" to Google. This bypasses simple "direct traffic" blocks.
    'Referer': 'https://www.google.com/',

    # 5. Fetch Metadata (CRITICAL FIX)
    'Sec-Fetch-Dest': 'document',
    'Sec-Fetch-Mode': 'navigate',
    # Changed from 'none' to 'cross-site'.
    # 'none' implies direct entry (typing URL). You cannot use 'none' if you send a Referer.
    # 'cross-site' correctly tells the server "I clicked a link from Google to get here."
    'Sec-Fetch-Site': 'cross-site',
    'Sec-Fetch-User': '?1',
    'Upgrade-Insecure-Requests': '1',
}
# Maximum number of citation sources per query during synthesis
MAX_SYNTHESIS_QUERY_SOURCES = 5
# Max QA pairs to keep in context during iterative synthesis
MAX_SYNTHESIS_QA_CONTEXT = 50
# Number of retries for artifact synthesis before failing
NUM_SYNTHESIS_RETRIES = 5
# Whether to save synthesis artifact as JSON (in addition to text)
SYNTHESIS_SAVE_JSON = False

CAESAR_CONFIG = {
    "CaesarAgent": {
        # Maximum number of words for adapting agent role description
        "role_max_length": 350,
        # Total number of pages to explore before stopping
        "max_iterations": 1000,
        # Maximum depth of exploration tree before backtracking
        "max_depth": 1000,
        # Initial URL to begin exploration (enabling means start_query must be None)
        "starting_url": None,
        # Initial query to being exploration (enabling means starting_url must be None)
        "starting_query": None,
        # Additional queries to generate from initial query to help agent
        "additional_starting_queries": 5,
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
        # Maximum times to revisiting page it has seen before during exploration
        "max_allowed_revisits": 20,
        # Whether to use new link display format or not
        "fancy_link_display": True,
        # Whether to dynamically determine to explore or go back to visited pages
        "use_explore_strategy": True,
        # Maximum web searches allowed during exploration (including starting search)
        "max_web_searches": 0,
        # LLM config for agent's ACT/THINK phases to encourage exploration
        "exploration_llm_config": {
            "model": "gpt-5.2",
            "reasoning_effort": "low",
            "temperature": 0.9,
            "max_completion_tokens": 5000,
            "timeout": 120,
        },

        # Overwrites the default role with new role from file, order is [overwrite] -> [adapt]
        "overwrite_role_file": None,
        # Whether to modify agent role based on starting URL and/or insights
        "adapt_role": False,
        # Insights file path used to adapt the role and change it
        "adapt_role_file": None
    },

    "ArtifactSynthesizer": {
        # Set to True for classic mode (ask all queries at once)
        "synthesis_classic_mode": False,
        # Number of drafts of synthesizing artifact
        "synthesis_drafts": 3,
        # Number of Q/A iterations per draft
        "synthesis_iterations": 20,
        # Top_k for synthesis query and retrieval from DB
        "synthesis_top_k": 50,
        # Top_n for synthesis query and retrieval from DB
        "synthesis_top_n": 10,
        # When synthesizing artifact, use previous artifact if it exists
        "synthesis_prev_artifact": True,
        # Maximum of words for generating final synthesis artifact
        "synthesis_max_length": None,
        # Whether to merge all draft artifacts into final artifact
        "synthesis_merge_artifacts": False,
        # Whether to generate ELI5 explanation for artifact text
        "synthesis_eli5": False,
        # Maximum words for ELI5 explanation (None = unspecified word limit)
        "synthesis_eli5_length": None,
        # Maximum exploration iterations to filter knowledge base for queries
        "synthesis_iteration_filter": None,
    },

    "AgentMemory": {
        # Whether to clear memory or not
        "clear_memory": False,
        # Whether to enable agent memory
        "enabled": True,
        # Whether to use vector DB or graph DB
        "use_graph": False,
    },

    "BraveSearch": {
        # Number of search results to fetch
        "num_results": 20,
        # Number of tries to attempt to search
        "max_retries": 1000,
        # Delay between search retries
        "retry_delay": 1,
        # Request timeout in seconds
        "timeout": 30,
    },

    # Default config for LLM outside of agent exploration
    "OpenAIHandler": {
        # Model name for LLM
        "model": "gpt-5.2",
        # Reasoning effort for GPT-5/O models
        "reasoning_effort": "medium",
        # Base temperature for LLM (overridden by exploration_llm_config for ACT/THINK)
        "temperature": 0.1,
        # Maximum tokens per LLM response
        "max_completion_tokens": 50000,
        # API timeout in seconds
        "timeout": 300,
    },

    # Default config for vector store knowledge base
    "ChromaClientManager": {
        # Model name for LLM
        "llm_model": "gpt-5-mini",
        # Reasoning effort for GPT-5/O models
        "llm_reasoning_effort": "medium",
         # LLM model temperature (for query/reranking)
        "llm_temperature": 0.1,
        # Maximum number of words for query response
        "response_max_length": 350,
    }
}