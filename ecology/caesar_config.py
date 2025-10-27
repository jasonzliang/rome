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

        # False for classic mode
        "iterative_synthesis": True,
        # Number of q/a iterations
        "synthesis_iterations": 10,

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