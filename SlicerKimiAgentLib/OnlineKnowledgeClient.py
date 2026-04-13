"""
OnlineKnowledgeClient - Online knowledge retrieval for Slicer code generation.

Uses GitHub API to retrieve code examples from Slicer repositories.
Note: Code search requires GitHub token. Without token, falls back to
fetching official script repository files from raw.githubusercontent.com
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

from .GitHubAPIClient import GitHubAPIClient

logger = logging.getLogger(__name__)


class OnlineKnowledgeClient:
    """
    Client for retrieving Slicer knowledge from online sources.
    
    Two modes:
    1. With GitHub token: Can search code in Slicer repositories
    2. Without token: Can only fetch raw files (script repository)
    """
    
    SLICER_REPO = "Slicer/Slicer"
    SCRIPT_REPO_PATH = "Docs/developer_guide/script_repository"
    
    def __init__(self, github_token: Optional[str] = None, enabled: bool = True):
        self.github = GitHubAPIClient(token=github_token)
        self.enabled = enabled
        
    def set_enabled(self, enabled: bool):
        self.enabled = enabled
        
    def set_token(self, token: str):
        self.github.set_token(token)
        
    def is_available(self) -> bool:
        """Check if online services are available."""
        if not self.enabled:
            return False
        # Check if we can reach GitHub (raw files are always available for public repos)
        _, error = self.github.get_file_content("Slicer", "Slicer", "README.md", "main")
        return error is None
    
    def has_code_search(self) -> bool:
        """Check if code search is available (requires token)."""
        return self.github.has_token()
    
    def search_code_examples(self, query: str, context: Optional[Dict] = None, max_results: int = 5) -> Dict:
        """
        Search for code examples from online sources.
        
        If GitHub token is available: uses code search API
        If no token: falls back to fetching official script repository files
        """
        if not self.enabled:
            return {"examples": [], "sources": [], "error": "Online search disabled"}
        
        # If we have a token, use code search
        if self.has_code_search():
            return self._search_with_code_api(query, max_results)
        else:
            # Without token, try to fetch relevant script repository files
            return self._fetch_script_repository_examples(query, context, max_results)
    
    def _search_with_code_api(self, query: str, max_results: int) -> Dict:
        """Search using GitHub code search API (requires token)."""
        api_terms = self._extract_api_terms(query)
        search_query = self._build_search_query(query, api_terms)
        
        results, error = self.github.search_code(
            query=search_query,
            repo=self.SLICER_REPO,
            language="python",
            per_page=max_results * 2
        )
        
        if error:
            logger.warning(f"GitHub code search failed: {error}")
            return {"examples": [], "sources": [], "error": error}
        
        examples = []
        sources = []
        for result in results[:max_results]:
            example = self._format_code_result(result)
            if example:
                examples.append(example)
                sources.append(result.get("url"))
        
        # If we found specific API terms, search those too
        if len(examples) < max_results and api_terms:
            for term in api_terms[:2]:
                if len(examples) >= max_results:
                    break
                results, _ = self.github.search_code(
                    query=term, repo=self.SLICER_REPO, language="python", per_page=3
                )
                for result in results:
                    if len(examples) >= max_results:
                        break
                    example = self._format_code_result(result)
                    if example and not self._is_duplicate(example, examples):
                        examples.append(example)
                        sources.append(result.get("url"))
        
        return {"examples": examples, "sources": list(set(sources)), "error": None}
    
    def _fetch_script_repository_examples(
        self, query: str, context: Optional[Dict], max_results: int
    ) -> Dict:
        """
        Fetch from official script repository (no token required).
        This is the fallback when no GitHub token is provided.
        """
        # Identify which script repository files might be relevant
        topics = self._identify_topics(query)
        
        if context and context.get("topics"):
            # Use topics from local search
            topics.extend(context.get("topics", []))
        
        topics = list(set(topics))[:3]  # Limit to 3 unique topics
        
        if not topics:
            topics = ["volumes"]  # Default fallback
        
        all_examples = []
        sources = []
        
        for topic in topics:
            result = self.fetch_official_script_examples(topic)
            if not result.get("error"):
                examples = result.get("examples", [])
                # Limit examples per topic
                for ex in examples[:2]:
                    if len(all_examples) >= max_results:
                        break
                    ex["source_file"] = f"{topic}.md"
                    ex["type"] = "script_repository_online"
                    all_examples.append(ex)
                sources.append(result.get("source"))
        
        if not all_examples:
            return {
                "examples": [],
                "sources": [],
                "error": "No GitHub token provided. Code search requires authentication. Add a token in settings for online code search, or use local script repository only."
            }
        
        return {"examples": all_examples, "sources": sources, "error": None}
    
    def fetch_official_script_examples(self, topic: str) -> Dict:
        """Fetch official script repository examples from Slicer main repo."""
        filename = f"{topic}.md"
        content, error = self.github.get_file_content(
            owner="Slicer", repo="Slicer",
            path=f"{self.SCRIPT_REPO_PATH}/{filename}",
            ref="main"
        )
        
        if error:
            return {"examples": [], "error": error}
        
        examples = self._extract_code_blocks(content)
        return {
            "examples": examples,
            "source": f"https://github.com/Slicer/Slicer/tree/main/{self.SCRIPT_REPO_PATH}/{filename}",
            "error": None
        }
    
    def _identify_topics(self, query: str) -> List[str]:
        """Identify relevant script repository topics from query."""
        query_lower = query.lower()
        topics = []
        
        topic_keywords = {
            "volumes": ["volume", "image", "nrrd", "nifti", "load", "render", "slice", "numpy"],
            "segmentations": ["segmentation", "segment", "labelmap", "mask", "editor", "effect", "threshold"],
            "markups": ["markup", "fiducial", "point", "curve", "line", "plane", "roi"],
            "models": ["model", "mesh", "surface", "stl", "obj", "polydata"],
            "transforms": ["transform", "linear", "grid", "displacement", "rotation", "matrix"],
            "dicom": ["dicom", "pacs", "import", "study", "series", "patient"],
            "gui": ["gui", "ui", "layout", "view", "widget", "window"],
            "plots": ["plot", "chart", "graph", "histogram", "matplotlib"],
        }
        
        for topic, keywords in topic_keywords.items():
            if any(kw in query_lower for kw in keywords):
                topics.append(topic)
        
        return topics
    
    def _extract_api_terms(self, query: str) -> List[str]:
        """Extract potential Slicer API terms from query."""
        terms = []
        patterns = ["slicer.util.", "slicer.mrmlScene", "slicer.modules.", "slicer.app.", "vtkMRML"]
        query_lower = query.lower()
        
        for pattern in patterns:
            if pattern.lower() in query_lower:
                idx = query_lower.find(pattern.lower())
                end_idx = idx + len(pattern)
                while end_idx < len(query) and (query[end_idx].isalnum() or query[end_idx] == '_'):
                    end_idx += 1
                full_term = query[idx:end_idx]
                if full_term and full_term not in terms:
                    terms.append(full_term)
        return terms
    
    def _build_search_query(self, query: str, api_terms: List[str]) -> str:
        """Build an effective GitHub search query."""
        if api_terms:
            return api_terms[0]
        
        stop_words = {'how', 'to', 'do', 'i', 'can', 'you', 'please', 'help', 
                      'create', 'make', 'get', 'set', 'the', 'a', 'an', 'in', 
                      'with', 'from', 'for', 'and', 'or', 'using', 'use'}
        
        words = query.lower().split()
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        tech_terms = ['volume', 'segmentation', 'model', 'transform', 'fiducial',
                      'markup', 'slice', 'render', 'dicom', 'numpy', 'array',
                      'load', 'save', 'export', 'import']
        
        prioritized = [t for t in tech_terms if t in keywords]
        for kw in keywords:
            if kw not in prioritized:
                prioritized.append(kw)
        
        return " ".join(prioritized[:3])
    
    def _format_code_result(self, result: Dict) -> Optional[Dict]:
        """Format a GitHub search result into a code example."""
        path = result.get("path", "")
        
        # Skip test files
        if "/Testing/" in path or "/test" in path.lower():
            return None
        if not path.endswith(".py"):
            return None
        
        text_matches = result.get("text_matches", [])
        code_snippet = ""
        
        for match in text_matches:
            fragment = match.get("fragment", "")
            if fragment:
                code_snippet = fragment
                break
        
        if not code_snippet:
            return None
        
        return {
            "code": self._clean_code_snippet(code_snippet),
            "source": result.get("repository", "Slicer/Slicer"),
            "file": path,
            "url": result.get("url"),
            "type": "github"
        }
    
    def _clean_code_snippet(self, snippet: str) -> str:
        """Clean up a code snippet."""
        lines = snippet.split('\n')
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        
        if lines:
            leading_spaces = [len(line) - len(line.lstrip()) for line in lines if line.strip()]
            if leading_spaces:
                min_indent = min(leading_spaces)
                lines = [line[min_indent:] if line.strip() else line for line in lines]
        
        return '\n'.join(lines)
    
    def _extract_code_blocks(self, markdown_content: str) -> List[Dict]:
        """Extract Python code blocks from markdown content."""
        examples = []
        pattern = r'```python\s*\n(.*?)\n```'
        matches = re.findall(pattern, markdown_content, re.DOTALL)
        
        for i, match in enumerate(matches):
            examples.append({
                "code": match.strip(),
                "index": i,
                "type": "script_repository"
            })
        return examples
    
    def _is_duplicate(self, example: Dict, existing: List[Dict]) -> bool:
        """Check if an example is a duplicate."""
        code = example.get("code", "").strip()
        for existing_ex in existing:
            existing_code = existing_ex.get("code", "").strip()
            if code == existing_code:
                return True
            if len(code) > 50 and len(existing_code) > 50:
                if code in existing_code or existing_code in code:
                    return True
        return False
    
    def get_status(self) -> Dict:
        """Get client status."""
        status = {
            "enabled": self.enabled,
            "has_code_search": self.has_code_search(),
        }
        if self.has_code_search():
            status["rate_limit"] = self.github.get_rate_limit_status()
        return status
    
    def clear_cache(self):
        """Clear all caches."""
        self.github.clear_cache()
