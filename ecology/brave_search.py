"""BraveSearch with hardcoded constants and automatic file naming"""
import time
import re
import os
import requests
import sys
from typing import Dict, Optional, List, Union, Tuple
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rome.config import (DEFAULT_CONFIG, set_attributes_from_config,
                     SHORT_SUMMARY_LEN, SUMMARY_LENGTH, LONG_SUMMARY_LEN,
                     LONGER_SUMMARY_LEN, LONGEST_SUMMARY_LEN)
from rome.logger import get_logger
from rome.parsing import hash_string
from .caesar_config import CAESAR_CONFIG

# Search endpoint
ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
# Result directory in which to store results
SEARCH_RESULT_DIR = "search_result"
# Max number of search results API supports
MAX_NUM_RESULTS = 20
# Multiplier for backoff during search
BACKOFF_MULTIPLIER = 2
# Delay in sec between each search to avoid hitting rate limit
SEARCH_DELAY = 1


class BraveSearchError(Exception):
    """Base exception for BraveSearch errors"""
    pass
class RateLimitError(BraveSearchError):
    """Rate limit exceeded"""
    pass
class APIKeyError(BraveSearchError):
    """Invalid API key"""
    pass


class BraveSearch:
    """Convert Brave Search API results to local HTML file with retry logic"""

    def __init__(self, agent, config: Dict = None):
        self.agent = agent
        self.logger = get_logger()

        # Read API key from environment variable only
        self.api_key = os.getenv("BRAVE_API_KEY")
        if not self.api_key:
            raise APIKeyError("BRAVE_API_KEY environment variable not set")

        # Load other settings from config
        self.config = config
        set_attributes_from_config(self, self.config, CAESAR_CONFIG['BraveSearch'].keys())

        if self.num_results > MAX_NUM_RESULTS:
            self.logger.error(f"num_results={self.num_results} exceeds API limit, capping to {MAX_NUM_RESULTS}")
            self.num_results = MAX_NUM_RESULTS

        self.output_dir = os.path.join(self.agent.get_log_dir(), SEARCH_RESULT_DIR)
        os.makedirs(self.output_dir, exist_ok=True)

    def _generate_filename(self, query) -> str:
        """Generate filename from query and timestamp"""
        # Sanitize query for filename (remove special chars, limit length)
        if type(query) is list and len(query) > 1:
            query = f"multi-query-{len(query)}_{query[0]}"
        elif type(query) is list and len(query) == 1:
            query = query[0]
        self.logger.assert_true(type(query) is str, "Query must be string or list")

        safe_query = re.sub(r'[^\w\s-]', '', query)
        safe_query = re.sub(r'[-\s]+', '-', safe_query)
        safe_query = safe_query[:SHORT_SUMMARY_LEN]  # Limit length

        # Add timestamp/hash for uniqueness
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        query_hash = hash_string(query)[:8]

        return f"{safe_query}_{query_hash}.html"

    def search_and_save(self, queries: Union[str, List[str]]) -> str:
        """Execute search with retries and return local file URL"""
        # Normalize to list
        query_list = [queries] if isinstance(queries, str) else queries
        self.logger.debug(f"Brave search queries: {query_list}")

        # Setup file name and check cached file
        filename = self._generate_filename(queries)
        html_file = os.path.abspath(os.path.join(self.output_dir, filename))
        if os.path.exists(html_file):
            self.logger.debug(f"Using cached search results html file: {html_file}")
            return f"file://{html_file}"

        # Execute all searches
        all_results = []
        for q in query_list:
            all_results.append((q, self._search_with_retry(q)))
            time.sleep(SEARCH_DELAY)

        # Generate HTML (merged or single)
        html = (self._json_to_html_merged(all_results) if len(query_list) > 1
                else self._json_to_html(all_results[0][1], all_results[0][0]))

        for _, results in all_results:
            self.logger.debug(f"Brave search results: {results}")

        # Write html to file in log dir
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html)

        self.logger.debug(f"Saved search results to html file: {html_file}")
        return f"file://{html_file}"

    def _search_with_retry(self, query: str) -> Dict:
        """Execute search with exponential backoff retry logic"""
        last_exception = None
        delay = self.retry_delay

        for attempt in range(self.max_retries):
            try:
                return self._search(query)

            except requests.exceptions.HTTPError as e:
                last_exception = e

                if e.response.status_code == 401:
                    raise APIKeyError("Invalid or expired API key") from e

                elif e.response.status_code == 429:
                    # Rate limit - check headers for retry time
                    retry_after = e.response.headers.get('Retry-After')
                    if retry_after:
                        wait_time = int(retry_after)
                        self.logger.error(f"Rate limit hit. Waiting {wait_time}s (from Retry-After header)...")
                        time.sleep(wait_time)
                    else:
                        self.logger.error(f"Rate limit hit. Retry {attempt + 1}/{self.max_retries} after {delay}s...")
                        time.sleep(delay)
                        delay *= BACKOFF_MULTIPLIER

                    if attempt == self.max_retries - 1:
                        raise RateLimitError(f"Rate limit exceeded after {self.max_retries} retries") from e

                elif e.response.status_code >= 500:
                    # Server error - retry
                    self.logger.error(f"Server error {e.response.status_code}. Retry {attempt + 1}/{self.max_retries} after {delay}s...")
                    time.sleep(delay)
                    delay *= BACKOFF_MULTIPLIER

                    if attempt == self.max_retries - 1:
                        raise BraveSearchError(f"Server error after {self.max_retries} retries") from e
                else:
                    # Other HTTP error - don't retry
                    raise BraveSearchError(f"HTTP {e.response.status_code}: {e}") from e

            except requests.exceptions.Timeout as e:
                last_exception = e
                self.logger.error(f"Request timeout. Retry {attempt + 1}/{self.max_retries} after {delay}s...")
                time.sleep(delay)
                delay *= BACKOFF_MULTIPLIER

                if attempt == self.max_retries - 1:
                    raise BraveSearchError(f"Request timeout after {self.max_retries} retries") from e

            except requests.exceptions.RequestException as e:
                last_exception = e
                self.logger.error(f"Network error. Retry {attempt + 1}/{self.max_retries} after {delay}s...")
                time.sleep(delay)
                delay *= BACKOFF_MULTIPLIER

                if attempt == self.max_retries - 1:
                    raise BraveSearchError(f"Network error after {self.max_retries} retries") from e

        # Should never reach here, but just in case
        raise BraveSearchError(f"Failed after {self.max_retries} retries") from last_exception

    def _search(self, query: str) -> Dict:
        """Query Brave Search API (single attempt)"""
        headers = {
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip',
            'X-Subscription-Token': self.api_key
        }
        params = {'q': query, 'count': self.num_results}

        # Use hardcoded BRAVE_ENDPOINT constant
        response = requests.get(
            ENDPOINT,
            headers=headers,
            params=params,
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()

    def _json_to_html(self, search_results: Dict, query: str) -> str:
        """Convert JSON to detailed HTML"""
        results = search_results.get('web', {}).get('results', [])
        api_query = search_results.get('query', {}).get('original', query)

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{query}</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 20px auto; padding: 0 20px; }}
        h1 {{ color: #1a0dab; border-bottom: 2px solid #ddd; padding-bottom: 10px; }}
        .query-info {{ background: #f0f0f0; padding: 15px; margin: 20px 0; border-radius: 5px; }}
        .query-info strong {{ color: #333; }}
        .result {{ margin: 25px 0; padding: 15px; border-left: 3px solid #4285f4; background: #f9f9f9; }}
        .result h3 {{ margin: 0 0 8px 0; }}
        .result a {{ color: #1a0dab; text-decoration: none; font-size: 18px; }}
        .result a:hover {{ text-decoration: underline; }}
        .url {{ color: #006621; font-size: 14px; margin: 5px 0; }}
        .description {{ color: #545454; margin: 10px 0; line-height: 1.5; }}
        .meta {{ color: #70757a; font-size: 13px; margin-top: 8px; }}
    </style>
</head>
<body>
    <h1>Search Results: {api_query}</h1>
    <div class="query-info">
        <strong>Query:</strong> {query}<br>
        <strong>Search Results:</strong> {len(results)}
    </div>
"""

        for r in results:
            title = r.get('title', 'No Title')
            url = r.get('url', '')
            description = r.get('description', 'No description available')
            language = r.get('language', 'N/A')
            page_age = r.get('page_age', 'N/A').split('T')[0] if r.get('page_age') else 'N/A'

            html += f"""
    <div class="result">
        <h3><a href="{url}">{title}</a></h3>
        <div class="url">{url}</div>
        <div class="description">{description}</div>
        <div class="meta">Language: {language} | Published: {page_age}</div>
    </div>
"""

        html += """
</body>
</html>
"""
        return html

    def _json_to_html_merged(self, query_results: List[Tuple[str, Dict]]) -> str:
        """Convert multiple query results to merged HTML"""
        total_results = sum(len(results.get('web', {}).get('results', []))
                          for _, results in query_results)

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Merged Search Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 20px auto; padding: 0 20px; }}
        h1 {{ color: #1a0dab; border-bottom: 2px solid #ddd; padding-bottom: 10px; }}
        h2 {{ color: #1a0dab; margin-top: 30px; font-size: 20px; }}
        .summary {{ background: #f0f0f0; padding: 15px; margin: 20px 0; border-radius: 5px; }}
        .result {{ margin: 15px 0; padding: 12px; border-left: 3px solid #4285f4; background: #f9f9f9; }}
        .result h3 {{ margin: 0 0 6px 0; }}
        .result a {{ color: #1a0dab; text-decoration: none; font-size: 16px; }}
        .result a:hover {{ text-decoration: underline; }}
        .url {{ color: #006621; font-size: 13px; margin: 4px 0; }}
        .description {{ color: #545454; margin: 8px 0; line-height: 1.4; }}
    </style>
</head>
<body>
    <h1>Merged Search Results</h1>
    <div class="summary">
        <strong>Queries:</strong> {len(query_results)} | <strong>Total Search Results:</strong> {total_results}
    </div>
"""

        for query, search_results in query_results:
            results = search_results.get('web', {}).get('results', [])
            api_query = search_results.get('query', {}).get('original', query)

            html += f"""
    <h2>Query: {api_query}</h2>
"""

            for r in results:
                title = r.get('title', 'No Title')
                url = r.get('url', '')
                description = r.get('description', 'No description available')

                html += f"""
    <div class="result">
        <h3><a href="{url}">{title}</a></h3>
        <div class="url">{url}</div>
        <div class="description">{description}</div>
    </div>
"""

        html += """
</body>
</html>
"""
        return html