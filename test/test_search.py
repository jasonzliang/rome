#!/usr/bin/env python3
"""
This script tests the Brave Search API by sending a sample search query,
printing the results, and displaying rate limit information.
It retrieves the API key from an environment variable.
"""

# RUN MANUALLY ONLY
import requests
import json
import os
import time

# --- Configuration ---
# The script gets the API key from an environment variable called BRAVE_API_KEY.
SUBSCRIPTION_KEY = os.getenv("BRAVE_API_KEY")
# Brave Search API endpoint for web search.
ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
SEARCH_TERM = "Brave Search API"
# You can change the number of results, up to a maximum of 20.
NUM_RESULTS = 10

def print_rate_limit_info(headers):
    """Parses and prints the rate limit headers in a more readable format."""
    print("\n--- Rate Limit & Usage Details ---")

    limit_header = headers.get('X-RateLimit-Limit', 'N/A,N/A').split(',')
    remaining_header = headers.get('X-RateLimit-Remaining', 'N/A,N/A').split(',')
    reset_header = headers.get('X-RateLimit-Reset', 'N/A,N/A').split(',')

    try:
        # Per-second limit details
        if len(limit_header) > 0 and limit_header[0] != 'N/A':
            print(f"Per-Second Quota: {remaining_header[0]}/{limit_header[0]} requests remaining.")
            print(f"                   (Resets in {reset_header[0]} second(s))")

        # Monthly limit details
        if len(limit_header) > 1 and limit_header[1] != 'N/A':
            monthly_limit = int(limit_header[1].strip())
            monthly_remaining = int(remaining_header[1].strip())
            monthly_used = monthly_limit - monthly_remaining

            # Calculate reset time in days/hours
            reset_seconds = int(reset_header[1].strip())
            days, remainder = divmod(reset_seconds, 86400)
            hours, _ = divmod(remainder, 3600)

            print(f"Monthly Quota:    {monthly_remaining}/{monthly_limit} requests remaining ({monthly_used} used).")
            print(f"                   (Resets in approximately {days} days and {hours} hours)")

    except (IndexError, ValueError) as e:
        # Fallback for unexpected header formats
        print("[!] Could not parse rate limit headers completely. Displaying raw data.")
        print(f"Limit: {headers.get('X-RateLimit-Limit', 'N/A')}")
        print(f"Remaining: {headers.get('X-RateLimit-Remaining', 'N/A')}")
        print(f"Reset (seconds): {headers.get('X-RateLimit-Reset', 'N/A')}")


def test_brave_api(key, endpoint, query):
    """
    Sends a GET request to the Brave Search API and prints the results and rate limits.

    Args:
        key (str): Your Brave Search API subscription key.
        endpoint (str): The API endpoint URL.
        query (str): The search term to look for.
    """
    headers = {'Accept': 'application/json', 'X-Subscription-Token': key}
    # Add the 'count' parameter to control the number of results
    params = {'q': query, 'count': NUM_RESULTS}

    print(f"[*] Testing Brave Search API...")
    print(f"[*] Requesting {NUM_RESULTS} results for query: '{query}'\n")

    try:
        response = requests.get(endpoint, headers=headers, params=params)
        response.raise_for_status()

        print("[+] Successfully received a response from the API.")

        print_rate_limit_info(response.headers)

        search_results = response.json()

        # --- Print Web Results ---
        web_search_data = search_results.get('web')
        if web_search_data and 'results' in web_search_data:
            print(f"\n--- Showing Top {NUM_RESULTS} Web Results ---")
            results = web_search_data.get('results', [])

            if not results:
                print("[-] No web results found for this query.")

            # The loop will now respect the NUM_RESULTS variable
            for i, result in enumerate(results, 1):
                # Extracting more details from the result object
                profile_name = result.get('profile', {}).get('long_name', 'N/A')
                subtype = result.get('subtype', 'generic').capitalize()
                language = result.get('language', 'N/A')
                page_date = result.get('page_age', 'N/A').split('T')[0] # Get just the date part

                print(f"\n{i}. {result.get('title', 'No Title')}")
                print(f"   URL: {result.get('url', 'No URL')}")
                print(f"   Source: {profile_name} ({subtype})")
                print(f"   Snippet: {result.get('description', 'No Snippet')}")
                print(f"   Published: {page_date} | Language: {language}")

        else:
            print("\n[-] Could not find 'web' results in the API response.")

        # --- Print Video Results ---
        video_search_data = search_results.get('videos')
        if video_search_data and 'results' in video_search_data:
            print("\n--- Top Video Results ---")
            video_results = video_search_data.get('results', [])

            if not video_results:
                print("[-] No video results found for this query.")

            for i, result in enumerate(video_results[:3], 1):
                print(f"\n{i}. {result.get('title', 'No Title')}")
                print(f"   URL: {result.get('url', 'No URL')}")
                print(f"   Creator: {result.get('video', {}).get('creator', 'N/A')}")
                print(f"   Duration: {result.get('video', {}).get('duration', 'N/A')}")
                print(f"   Description: {result.get('description', 'No Description')}")
        else:
            print("\n[-] Could not find 'videos' results in the API response.")

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            print(f"\n[!] HTTP 401 Error: Authentication Failed.")
            print("[!] Please check that your BRAVE_API_KEY is valid and has not expired.")
        elif e.response.status_code == 429:
            print(f"\n[!] HTTP 429 Error: Too Many Requests.")
            print("[!] You have exceeded your rate limit. Please wait before trying again.")
        else:
             print(f"\n[!] An HTTP error occurred: {e}")
    except requests.exceptions.RequestException as e:
        print(f"\n[!] A network error occurred during the API request: {e}")
    except Exception as e:
        print(f"\n[!] An unexpected error occurred: {e}")

if __name__ == "__main__":
    if not SUBSCRIPTION_KEY:
        print("[!] Error: The BRAVE_API_KEY environment variable is not set.")
        print("[!] Please set it before running the script.")
        print("[!] For example, on Linux/macOS: export BRAVE_API_KEY='your_key_here'")
        print("[!] On Windows (Command Prompt): set BRAVE_API_KEY=your_key_here")
    else:
        test_brave_api(SUBSCRIPTION_KEY, ENDPOINT, SEARCH_TERM)
