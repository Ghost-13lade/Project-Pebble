"""
Web search tool for Pebble.
Uses DuckDuckGo for privacy-friendly web searches.
"""
from __future__ import annotations

from typing import List, Optional


def search_web(query: str, max_results: int = 3) -> str:
    """
    Search the web using DuckDuckGo and return formatted results.
    
    Args:
        query: The search query
        max_results: Maximum number of results to return (default 3)
    
    Returns:
        Formatted string with search results, or empty string if no results
    """
    if not query.strip():
        return ""
    
    print(f"[Search] Browsing the web for: \"{query}\"...")
    
    try:
        from duckduckgo_search import DDGS
        
        results: List[dict] = []
        
        with DDGS() as ddgs:
            # Search for results
            search_results = ddgs.text(query, max_results=max_results)
            
            if search_results:
                results = list(search_results)
        
        if not results:
            print("[Search] No results found")
            return ""
        
        # Format results
        formatted = []
        for i, result in enumerate(results, 1):
            title = result.get("title", "Untitled")
            snippet = result.get("body", "")
            url = result.get("href", "")
            
            formatted.append(f"{i}. {title}\n   {snippet}\n   Source: {url}")
        
        output = "\n\n".join(formatted)
        print(f"[Search] Found {len(results)} results")
        return output
        
    except ImportError:
        print("[Search] ERROR: duckduckgo-search not installed!")
        print("[Search] Run: pip install duckduckgo-search")
        return ""
    except Exception as e:
        print(f"[Search] Error during search: {e}")
        return ""


def needs_web_search(user_message: str) -> bool:
    """
    Determine if a user message likely needs a web search.
    
    Keywords that trigger web search:
    - "search for", "look up", "google"
    - "what is the current", "what's the latest"
    - "who won", "latest news"
    - "current price", "weather in" (weather is already handled)
    - "when is", "where is" (with current events context)
    - "recent", "latest", "new", "today's"
    
    Args:
        user_message: The user's message
    
    Returns:
        True if web search is likely needed
    """
    message_lower = user_message.lower().strip()
    
    # Direct search triggers
    search_triggers = [
        "search for",
        "search ",
        "look up",
        "google",
        "find information",
        "can you find",
        "check online",
        "on the web",
    ]
    
    # Current/temporal triggers
    temporal_triggers = [
        "current",
        "latest",
        "recent",
        "today's",
        "this week",
        "this month",
        "who won",
        "what happened",
        "news about",
        "price of",
        "stock price",
        "exchange rate",
        "score of",
        "results of",
        "when is the next",
        "upcoming",
        "new release",
        "just came out",
    ]
    
    # Check triggers
    for trigger in search_triggers + temporal_triggers:
        if trigger in message_lower:
            return True
    
    # Question patterns that suggest current info
    question_patterns = [
        "what is the weather",  # Already handled by weather tool
    ]
    
    # Exclude patterns we handle elsewhere
    for pattern in question_patterns:
        if pattern in message_lower:
            return False
    
    return False


def extract_search_query(user_message: str) -> str:
    """
    Extract a clean search query from the user message.
    
    Removes common prefixes and extracts the core search intent.
    
    Args:
        user_message: The user's message
    
    Returns:
        A clean search query
    """
    message = user_message.strip()
    
    # Common prefixes to remove
    prefixes = [
        "search for",
        "search",
        "look up",
        "google",
        "can you find",
        "please find",
        "find information about",
        "check online for",
        "look on the web for",
    ]
    
    message_lower = message.lower()
    for prefix in prefixes:
        if message_lower.startswith(prefix):
            message = message[len(prefix):].strip()
            break
    
    # Remove question marks and clean up
    message = message.rstrip("?.!").strip()
    
    return message