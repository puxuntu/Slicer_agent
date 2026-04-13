"""
GitHubAPIClient - GitHub REST API client for searching Slicer code.
Note: GitHub Code Search API REQUIRES authentication.
"""

import json
import logging
import ssl
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote, urlencode

logger = logging.getLogger(__name__)


class GitHubAPIClient:
    """Client for GitHub REST API. Code search requires token."""
    
    DEFAULT_BASE_URL = "https://api.github.com"
    RAW_BASE_URL = "https://raw.githubusercontent.com"
    DEFAULT_TIMEOUT = 10
    
    def __init__(self, token: Optional[str] = None):
        self.token = token
        self.base_url = self.DEFAULT_BASE_URL
        self.timeout = self.DEFAULT_TIMEOUT
        self._rate_limit_remaining = None
        self._rate_limit_reset = None
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._cache_ttl = 3600
        
    def set_token(self, token: str):
        self.token = token
        
    def has_token(self) -> bool:
        return bool(self.token and self.token.strip())
        
    def _build_headers(self) -> Dict[str, str]:
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "SlicerKimiAgent/1.0",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None, use_cache: bool = True):
        cache_key = f"{endpoint}:{json.dumps(params, sort_keys=True) if params else ''}"
        if use_cache and cache_key in self._cache:
            data, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self._cache_ttl:
                return data, None
        
        url = f"{self.base_url}{endpoint}"
        if params:
            url = f"{url}?{urlencode(params, quote_via=quote)}"
        
        if self._rate_limit_remaining == 0:
            if self._rate_limit_reset and time.time() < self._rate_limit_reset:
                return None, "Rate limit exceeded"
        
        try:
            request = urllib.request.Request(url, headers=self._build_headers(), method='GET')
            ssl_context = ssl.create_default_context()
            
            with urllib.request.urlopen(request, timeout=self.timeout, context=ssl_context) as response:
                self._rate_limit_remaining = int(response.headers.get('X-RateLimit-Remaining', 0))
                self._rate_limit_reset = int(response.headers.get('X-RateLimit-Reset', 0))
                data = json.loads(response.read().decode('utf-8'))
                
                if use_cache:
                    self._cache[cache_key] = (data, time.time())
                return data, None
                
        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8')
            if e.code == 401:
                return None, "Authentication required. Please provide a GitHub token."
            elif e.code == 403:
                return None, "Rate limit exceeded or access forbidden"
            elif e.code == 404:
                return None, "Resource not found"
            else:
                return None, f"HTTP error {e.code}"
        except Exception as e:
            return None, f"Request failed: {str(e)}"
    
    def search_code(self, query: str, repo: Optional[str] = None, language: Optional[str] = "python", per_page: int = 10):
        """Search code - REQUIRES GitHub token."""
        if not self.has_token():
            return [], "GitHub token required for code search. Get one at github.com/settings/tokens (classic token with 'public_repo' scope)"
        
        search_terms = [query]
        if repo:
            search_terms.append(f"repo:{repo}")
        if language:
            search_terms.append(f"language:{language}")
        
        params = {
            "q": " ".join(search_terms),
            "per_page": min(per_page, 100),
        }
        
        data, error = self._make_request("/search/code", params)
        if error:
            return [], error
        
        results = []
        for item in data.get("items", []):
            results.append({
                "name": item.get("name"),
                "path": item.get("path"),
                "repository": item.get("repository", {}).get("full_name"),
                "url": item.get("html_url"),
                "text_matches": item.get("text_matches", []),
            })
        return results, None
    
    def get_file_content(self, owner: str, repo: str, path: str, ref: str = "main"):
        """Get file from raw.githubusercontent.com - no auth needed for public repos."""
        cache_key = f"raw:{owner}/{repo}/{path}@{ref}"
        if cache_key in self._cache:
            data, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self._cache_ttl:
                return data, None
        
        url = f"{self.RAW_BASE_URL}/{owner}/{repo}/{ref}/{path}"
        
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "SlicerKimiAgent/1.0"}, method='GET')
            ssl_context = ssl.create_default_context()
            
            with urllib.request.urlopen(request, timeout=self.timeout, context=ssl_context) as response:
                content = response.read().decode('utf-8')
                self._cache[cache_key] = (content, time.time())
                return content, None
                
        except urllib.error.HTTPError as e:
            if e.code == 404 and ref == "main":
                return self.get_file_content(owner, repo, path, "master")
            return None, f"HTTP error {e.code}"
        except Exception as e:
            return None, str(e)
    
    def get_rate_limit_status(self):
        """Get current rate limit status."""
        data, error = self._make_request("/rate_limit", use_cache=False)
        if error:
            return {"error": error}
        if data and "resources" in data:
            core = data["resources"].get("core", {})
            return {
                "limit": core.get("limit"),
                "remaining": core.get("remaining"),
            }
        return {"error": "Unknown response"}
    
    def clear_cache(self):
        self._cache.clear()
