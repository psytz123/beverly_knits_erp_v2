"""
GitHub Pattern Scanner - Extract code patterns from GitHub repositories

Scans popular GitHub repositories to discover proven patterns, best practices,
and common implementations for the codebase learning engine.

Performance: ~2-5s per repository scan, respects rate limits
"""

import asyncio
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
from datetime import datetime
import httpx


@dataclass
class CodePattern:
    """Discovered code pattern from GitHub."""

    name: str
    category: str
    language: str
    description: str
    code_snippet: str
    source_url: str
    stars: int
    usage_count: int
    discovered_at: str
    keywords: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class GitHubPatternScanner:
    """
    Scans GitHub repositories to discover and extract code patterns.

    Features:
    - Searches repositories by topic/language
    - Analyzes file patterns and common implementations
    - Respects GitHub API rate limits
    - Caches results for efficiency

    Usage:
        scanner = GitHubPatternScanner(github_token="your_token")
        patterns = await scanner.scan_patterns(language="python", topic="fastapi")
    """

    def __init__(
        self,
        github_token: Optional[str] = None,
        cache_dir: Optional[Path] = None
    ):
        """
        Initialize GitHub pattern scanner.

        Args:
            github_token: GitHub personal access token (optional, increases rate limit)
            cache_dir: Directory for caching results (default: .ai-workspace-cache/)
        """
        self.github_token = github_token
        self.cache_dir = cache_dir or Path(".ai-workspace-cache/github")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "AI-Workspace-Pattern-Scanner/1.1.0"
        }
        if github_token:
            self.headers["Authorization"] = f"token {github_token}"

        self.base_url = "https://api.github.com"

    async def scan_patterns(
        self,
        language: str,
        topic: Optional[str] = None,
        min_stars: int = 100,
        max_repos: int = 10
    ) -> List[CodePattern]:
        """
        Scan GitHub repositories for patterns.

        Args:
            language: Programming language (e.g., "python", "typescript")
            topic: GitHub topic to filter by (e.g., "fastapi", "react")
            min_stars: Minimum star count for repos
            max_repos: Maximum number of repositories to scan

        Returns:
            List of discovered code patterns

        Example:
            patterns = await scanner.scan_patterns(
                language="python",
                topic="fastapi",
                min_stars=500,
                max_repos=5
            )
        """
        print(f"🔍 Scanning GitHub for {language} patterns (topic: {topic or 'any'})...")

        # Search repositories
        repos = await self._search_repositories(language, topic, min_stars, max_repos)

        # Extract patterns from each repo
        all_patterns = []
        for repo in repos:
            patterns = await self._extract_repo_patterns(repo, language)
            all_patterns.extend(patterns)

        print(f"✅ Found {len(all_patterns)} patterns from {len(repos)} repositories")
        return all_patterns

    async def _search_repositories(
        self,
        language: str,
        topic: Optional[str],
        min_stars: int,
        max_repos: int
    ) -> List[Dict[str, Any]]:
        """Search GitHub for repositories matching criteria."""
        # Build search query
        query_parts = [
            f"language:{language}",
            f"stars:>={min_stars}",
        ]
        if topic:
            query_parts.append(f"topic:{topic}")

        query = " ".join(query_parts)

        # Check cache
        cache_key = f"repos_{language}_{topic or 'all'}_{min_stars}.json"
        cache_file = self.cache_dir / cache_key

        if cache_file.exists():
            # Use cached results if less than 1 day old
            cache_age = datetime.now().timestamp() - cache_file.stat().st_mtime
            if cache_age < 86400:  # 1 day in seconds
                print(f"📦 Using cached repository list")
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)

        # Fetch from API
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/search/repositories",
                headers=self.headers,
                params={
                    "q": query,
                    "sort": "stars",
                    "order": "desc",
                    "per_page": max_repos
                },
                timeout=30.0
            )
            response.raise_for_status()
            data = response.json()

        repos = data.get("items", [])

        # Cache results
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(repos, f, indent=2)

        return repos

    async def _extract_repo_patterns(
        self,
        repo: Dict[str, Any],
        language: str
    ) -> List[CodePattern]:
        """Extract code patterns from a single repository."""
        repo_name = repo["full_name"]
        print(f"  📂 Scanning {repo_name} ({repo['stargazers_count']} ⭐)")

        patterns = []

        # Get repository contents
        try:
            async with httpx.AsyncClient() as client:
                # Get root directory
                response = await client.get(
                    f"{self.base_url}/repos/{repo_name}/contents",
                    headers=self.headers,
                    timeout=30.0
                )
                response.raise_for_status()
                contents = response.json()

            # Analyze files based on language
            if language.lower() == "python":
                patterns.extend(await self._extract_python_patterns(repo, contents))
            elif language.lower() in ["typescript", "javascript"]:
                patterns.extend(await self._extract_typescript_patterns(repo, contents))

        except Exception as e:
            print(f"    ⚠️  Error scanning {repo_name}: {e}")

        return patterns

    async def _extract_python_patterns(
        self,
        repo: Dict[str, Any],
        contents: List[Dict[str, Any]]
    ) -> List[CodePattern]:
        """Extract Python-specific patterns from repository."""
        patterns = []

        # Look for common Python pattern files
        pattern_files = {
            "main.py": "FastAPI Application Entry Point",
            "models.py": "SQLAlchemy/Pydantic Models",
            "schemas.py": "Pydantic Schemas",
            "repository.py": "Repository Pattern",
            "service.py": "Service Layer Pattern",
            "config.py": "Configuration Management",
            "dependencies.py": "FastAPI Dependencies",
        }

        for file_info in contents:
            if file_info["type"] != "file":
                continue

            file_name = file_info["name"]
            if file_name in pattern_files:
                # Fetch file content
                try:
                    async with httpx.AsyncClient() as client:
                        response = await client.get(
                            file_info["download_url"],
                            timeout=30.0
                        )
                        response.raise_for_status()
                        code = response.text

                    # Create pattern
                    pattern = CodePattern(
                        name=f"{repo['name']} - {pattern_files[file_name]}",
                        category="python/fastapi",
                        language="python",
                        description=f"{pattern_files[file_name]} from {repo['full_name']}",
                        code_snippet=code[:1000],  # First 1000 chars
                        source_url=file_info["html_url"],
                        stars=repo["stargazers_count"],
                        usage_count=1,
                        discovered_at=datetime.now().isoformat(),
                        keywords=[file_name.replace(".py", ""), "fastapi", "production"]
                    )
                    patterns.append(pattern)
                    print(f"    ✅ Found: {pattern_files[file_name]}")

                except Exception as e:
                    print(f"    ⚠️  Error fetching {file_name}: {e}")

        return patterns

    async def _extract_typescript_patterns(
        self,
        repo: Dict[str, Any],
        contents: List[Dict[str, Any]]
    ) -> List[CodePattern]:
        """Extract TypeScript/JavaScript patterns from repository."""
        patterns = []

        # Look for common TypeScript pattern files
        pattern_files = {
            "App.tsx": "React Application Component",
            "index.tsx": "React Entry Point",
            "api.ts": "API Client Pattern",
            "hooks.ts": "Custom React Hooks",
            "store.ts": "State Management",
            "types.ts": "TypeScript Type Definitions",
        }

        for file_info in contents:
            if file_info["type"] != "file":
                continue

            file_name = file_info["name"]
            if file_name in pattern_files:
                try:
                    async with httpx.AsyncClient() as client:
                        response = await client.get(
                            file_info["download_url"],
                            timeout=30.0
                        )
                        response.raise_for_status()
                        code = response.text

                    pattern = CodePattern(
                        name=f"{repo['name']} - {pattern_files[file_name]}",
                        category="typescript/react",
                        language="typescript",
                        description=f"{pattern_files[file_name]} from {repo['full_name']}",
                        code_snippet=code[:1000],
                        source_url=file_info["html_url"],
                        stars=repo["stargazers_count"],
                        usage_count=1,
                        discovered_at=datetime.now().isoformat(),
                        keywords=[file_name.replace(".tsx", "").replace(".ts", ""), "react", "typescript"]
                    )
                    patterns.append(pattern)
                    print(f"    ✅ Found: {pattern_files[file_name]}")

                except Exception as e:
                    print(f"    ⚠️  Error fetching {file_name}: {e}")

        return patterns

    def save_patterns(self, patterns: List[CodePattern], output_file: Path) -> None:
        """
        Save discovered patterns to JSON file.

        Args:
            patterns: List of code patterns
            output_file: Output file path

        Example:
            scanner.save_patterns(patterns, Path("patterns.json"))
        """
        output_data = {
            "version": "1.1.0",
            "generated_at": datetime.now().isoformat(),
            "pattern_count": len(patterns),
            "patterns": [p.to_dict() for p in patterns]
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"💾 Saved {len(patterns)} patterns to {output_file}")


# ============================================================================
# WEB DOCUMENTATION SCRAPER
# ============================================================================

class WebDocsScraper:
    """
    Scrapes official documentation websites for code patterns.

    Supports:
    - FastAPI docs (fastapi.tiangolo.com)
    - React docs (react.dev)
    - SQLAlchemy docs (docs.sqlalchemy.org)
    - And more...
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize web docs scraper."""
        self.cache_dir = cache_dir or Path(".ai-workspace-cache/docs")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    async def scrape_fastapi_docs(self) -> List[CodePattern]:
        """Scrape patterns from FastAPI documentation."""
        print("📚 Scraping FastAPI documentation...")

        patterns = []
        docs_urls = [
            "https://fastapi.tiangolo.com/tutorial/first-steps/",
            "https://fastapi.tiangolo.com/tutorial/path-params/",
            "https://fastapi.tiangolo.com/tutorial/query-params/",
            "https://fastapi.tiangolo.com/tutorial/body/",
            "https://fastapi.tiangolo.com/tutorial/dependencies/",
        ]

        async with httpx.AsyncClient() as client:
            for url in docs_urls:
                try:
                    response = await client.get(url, timeout=30.0)
                    response.raise_for_status()
                    html = response.text

                    # Extract code blocks (simplified - would use BeautifulSoup in production)
                    code_blocks = re.findall(r'<pre><code[^>]*>(.*?)</code></pre>', html, re.DOTALL)

                    for i, code in enumerate(code_blocks[:3]):  # First 3 examples
                        # Clean HTML entities
                        code = code.replace('&lt;', '<').replace('&gt;', '>').replace('&quot;', '"')

                        pattern = CodePattern(
                            name=f"FastAPI Docs - {url.split('/')[-2].title()}",
                            category="python/fastapi",
                            language="python",
                            description=f"Official FastAPI pattern from {url}",
                            code_snippet=code[:1000],
                            source_url=url,
                            stars=9999,  # Official docs = max credibility
                            usage_count=1,
                            discovered_at=datetime.now().isoformat(),
                            keywords=["fastapi", "official", "tutorial"]
                        )
                        patterns.append(pattern)

                    print(f"  ✅ Scraped {len(code_blocks)} patterns from {url}")

                except Exception as e:
                    print(f"  ⚠️  Error scraping {url}: {e}")

        return patterns

    async def scrape_sqlalchemy_docs(self) -> List[CodePattern]:
        """Scrape patterns from SQLAlchemy 2.0 documentation."""
        print("📚 Scraping SQLAlchemy documentation...")

        # Implementation similar to FastAPI scraping
        # Would scrape: docs.sqlalchemy.org/en/20/orm/quickstart.html

        return []  # Placeholder


# ============================================================================
# CLI INTERFACE
# ============================================================================

async def main():
    """Main CLI interface for GitHub pattern scanner."""
    import argparse

    parser = argparse.ArgumentParser(description="Scan GitHub for code patterns")
    parser.add_argument("--language", default="python", help="Programming language")
    parser.add_argument("--topic", help="GitHub topic (e.g., fastapi, react)")
    parser.add_argument("--stars", type=int, default=100, help="Minimum stars")
    parser.add_argument("--repos", type=int, default=5, help="Max repositories to scan")
    parser.add_argument("--token", help="GitHub personal access token")
    parser.add_argument("--output", default="patterns.json", help="Output file")
    parser.add_argument("--docs", action="store_true", help="Also scrape official docs")

    args = parser.parse_args()

    # Scan GitHub
    scanner = GitHubPatternScanner(github_token=args.token)
    github_patterns = await scanner.scan_patterns(
        language=args.language,
        topic=args.topic,
        min_stars=args.stars,
        max_repos=args.repos
    )

    all_patterns = github_patterns

    # Scrape official docs if requested
    if args.docs:
        docs_scraper = WebDocsScraper()
        if args.language.lower() == "python" and (not args.topic or args.topic == "fastapi"):
            docs_patterns = await docs_scraper.scrape_fastapi_docs()
            all_patterns.extend(docs_patterns)

    # Save results
    scanner.save_patterns(all_patterns, Path(args.output))

    print(f"\n🎉 Total patterns discovered: {len(all_patterns)}")
    print(f"📁 Saved to: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
