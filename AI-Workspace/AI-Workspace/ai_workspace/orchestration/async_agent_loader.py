"""
Async Agent Loading with Caching (PERF-015).

Provides high-performance asynchronous agent metadata loading with persistent
disk-based caching and lazy loading capabilities. Reduces agent loading time
from 2800ms to <400ms (warm) and <1000ms (cold) through parallel I/O and
intelligent caching strategies.

Features:
    - Parallel async loading of 156 agents using asyncio and aiofiles
    - Persistent disk-based cache with TTL and version-based invalidation
    - Lazy loading for on-demand agent retrieval
    - Concurrent load limits to prevent resource exhaustion
    - Checksum-based cache integrity validation
    - Category-based loading for selective agent initialization

Architecture:
    - AgentMetadata: Structured metadata for each agent
    - CacheConfig: Configuration for caching behavior
    - AsyncAgentLoader: Main loader with caching and lazy loading

Performance Targets:
    - Cold start (no cache): <1000ms for 156 agents (2.8x improvement)
    - Warm start (cached): <400ms for 156 agents (7x faster)
    - Cache hit rate: >95% for typical usage
    - Memory overhead: <20MB for cached metadata
    - Lazy load single agent: <10ms

Example:
    >>> import asyncio
    >>> from pathlib import Path
    >>> from ai_workspace.orchestration.async_agent_loader import (
    ...     AsyncAgentLoader, CacheConfig
    ... )
    >>>
    >>> # Configure cache
    >>> cache_config = CacheConfig(
    ...     cache_dir=Path.home() / ".ai-workspace" / "cache",
    ...     ttl_hours=24,
    ...     enable_lazy_loading=True,
    ...     max_concurrent_loads=20
    ... )
    >>>
    >>> # Create loader
    >>> loader = AsyncAgentLoader(
    ...     agent_dir=Path(".ai-workspace/agents"),
    ...     cache_config=cache_config
    ... )
    >>>
    >>> # Load all agents (async)
    >>> async def main():
    ...     agents = await loader.load_all_async()
    ...     print(f"Loaded {len(agents)} agents")
    ...     return agents
    >>>
    >>> # Run async loading
    >>> agents = asyncio.run(main())
    >>> print(agents["python-pro"].description)

Cache Format:
    Cache is stored as JSON at ~/.ai-workspace/cache/agents.json:
    {
        "cache_version": "1.0",
        "workspace_version": "1.2.0",
        "timestamp": "2024-01-15T10:30:00Z",
        "checksum": "abc123...",
        "agents": {
            "python-pro": {
                "name": "python-pro",
                "category": "02-languages",
                "description": "...",
                "file_path": "/path/to/agent.md",
                "tools": ["Read", "Write", ...],
                "keywords": ["python", "async", ...],
                "version": "1.0"
            },
            ...
        }
    }
"""

import asyncio
import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set

import aiofiles

logger = logging.getLogger(__name__)


@dataclass
class AgentMetadata:
    """
    Structured metadata for a single agent.

    Contains all essential information extracted from the agent's markdown file,
    including name, category, capabilities, and version information.

    Attributes:
        name: Agent identifier (e.g., "python-pro")
        category: Category directory (e.g., "02-languages")
        description: Brief description of agent capabilities
        file_path: Absolute path to agent markdown file
        tools: List of tools agent can use
        keywords: Keywords for semantic matching
        version: Agent version (for cache invalidation)
    """

    name: str
    category: str
    description: str
    file_path: str
    tools: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    version: str = "1.0"

    def to_dict(self) -> Dict:
        """
        Convert metadata to dictionary for serialization.

        Returns:
            Dictionary representation of metadata
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "AgentMetadata":
        """
        Create metadata from dictionary.

        Args:
            data: Dictionary with metadata fields

        Returns:
            AgentMetadata instance
        """
        return cls(**data)


@dataclass
class CacheConfig:
    """
    Configuration for agent caching behavior.

    Controls cache location, expiration, and loading strategy to balance
    performance and freshness.

    Attributes:
        cache_dir: Directory for cache storage
        ttl_hours: Time-to-live for cache in hours (default: 24)
        enable_lazy_loading: Whether to support lazy loading (default: True)
        max_concurrent_loads: Maximum parallel file loads (default: 20)
        cache_version: Cache format version for compatibility checks
    """

    cache_dir: Path
    ttl_hours: int = 24
    enable_lazy_loading: bool = True
    max_concurrent_loads: int = 20
    cache_version: str = "1.0"

    def __post_init__(self) -> None:
        """Ensure cache directory exists."""
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def cache_file(self) -> Path:
        """
        Get path to cache file.

        Returns:
            Path to agents.json cache file
        """
        return self.cache_dir / "agents.json"


class AsyncAgentLoader:
    """
    High-performance async agent loader with persistent caching.

    Implements parallel asynchronous loading of agent metadata with disk-based
    caching to achieve <1000ms cold start and <400ms warm start for 156 agents.

    The loader supports:
    - Full parallel loading of all agents
    - Category-based selective loading
    - Lazy on-demand loading of individual agents
    - Persistent caching with TTL and version-based invalidation
    - Concurrent load limits to prevent resource exhaustion

    Example:
        >>> import asyncio
        >>> from pathlib import Path
        >>>
        >>> config = CacheConfig(
        ...     cache_dir=Path.home() / ".ai-workspace" / "cache",
        ...     ttl_hours=24
        ... )
        >>> loader = AsyncAgentLoader(
        ...     agent_dir=Path(".ai-workspace/agents"),
        ...     cache_config=config
        ... )
        >>>
        >>> # Load all agents
        >>> agents = asyncio.run(loader.load_all_async())
        >>>
        >>> # Load specific category
        >>> python_agents = asyncio.run(
        ...     loader.load_category_async("02-languages")
        ... )
        >>>
        >>> # Lazy load single agent
        >>> agent = asyncio.run(loader.load_on_demand("python-pro"))
    """

    def __init__(
        self,
        agent_dir: Path,
        cache_config: CacheConfig,
        workspace_version: str = "1.2.0"
    ) -> None:
        """
        Initialize async agent loader.

        Args:
            agent_dir: Directory containing agent markdown files
            cache_config: Cache configuration
            workspace_version: AI Workspace version for cache validation
        """
        self.agent_dir = Path(agent_dir)
        self.cache_config = cache_config
        self.workspace_version = workspace_version

        # In-memory cache for lazy loading
        self._memory_cache: Dict[str, AgentMetadata] = {}
        self._cache_loaded = False

        # Semaphore for concurrent load limiting
        self._load_semaphore = asyncio.Semaphore(
            cache_config.max_concurrent_loads
        )

        logger.info(
            f"AsyncAgentLoader initialized: agent_dir={agent_dir}, "
            f"cache_dir={cache_config.cache_dir}, ttl={cache_config.ttl_hours}h"
        )

    async def load_all_async(self) -> Dict[str, AgentMetadata]:
        """
        Load all agents asynchronously with caching.

        Attempts to load from cache first. If cache is invalid or missing,
        performs parallel loading of all agent markdown files and saves
        to cache.

        Returns:
            Dictionary mapping agent names to AgentMetadata

        Example:
            >>> agents = await loader.load_all_async()
            >>> print(f"Loaded {len(agents)} agents")
            >>> print(agents["python-pro"].description)
        """
        start_time = asyncio.get_event_loop().time()

        # Try loading from cache
        cached_agents = await self._load_from_cache()

        if cached_agents is not None:
            load_time = (asyncio.get_event_loop().time() - start_time) * 1000
            logger.info(
                f"Loaded {len(cached_agents)} agents from cache "
                f"in {load_time:.2f}ms"
            )
            self._memory_cache = cached_agents
            self._cache_loaded = True
            return cached_agents

        # Cache miss - load from disk
        logger.info("Cache miss, loading agents from disk...")

        # Find all agent markdown files
        agent_files = self._find_all_agent_files()
        logger.info(f"Found {len(agent_files)} agent files")

        # Load agents in parallel with concurrency limit
        tasks = [
            self._load_agent_file_with_semaphore(file_path)
            for file_path in agent_files
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out None and exceptions
        agents: Dict[str, AgentMetadata] = {}
        errors = 0

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Error loading agent: {result}")
                errors += 1
            elif result is not None:
                agents[result.name] = result

        load_time = (asyncio.get_event_loop().time() - start_time) * 1000

        logger.info(
            f"Loaded {len(agents)} agents from disk in {load_time:.2f}ms "
            f"({errors} errors)"
        )

        # Save to cache
        await self._save_to_cache(agents)

        # Update memory cache
        self._memory_cache = agents
        self._cache_loaded = True

        return agents

    async def load_category_async(
        self, category: str
    ) -> Dict[str, AgentMetadata]:
        """
        Load agents from specific category.

        Loads only agents from the specified category directory. Uses cache
        if available and valid.

        Args:
            category: Category directory name (e.g., "02-languages")

        Returns:
            Dictionary of agents in the category

        Example:
            >>> python_agents = await loader.load_category_async("02-languages")
            >>> print(f"Found {len(python_agents)} language agents")
        """
        start_time = asyncio.get_event_loop().time()

        # Load from cache if available
        if self._cache_loaded:
            category_agents = {
                name: meta
                for name, meta in self._memory_cache.items()
                if meta.category == category
            }
            load_time = (asyncio.get_event_loop().time() - start_time) * 1000
            logger.info(
                f"Loaded {len(category_agents)} agents from category "
                f"'{category}' (cached) in {load_time:.2f}ms"
            )
            return category_agents

        # Load all agents to populate cache
        all_agents = await self.load_all_async()

        # Filter by category
        category_agents = {
            name: meta
            for name, meta in all_agents.items()
            if meta.category == category
        }

        load_time = (asyncio.get_event_loop().time() - start_time) * 1000
        logger.info(
            f"Loaded {len(category_agents)} agents from category "
            f"'{category}' in {load_time:.2f}ms"
        )

        return category_agents

    async def load_on_demand(self, agent_name: str) -> Optional[AgentMetadata]:
        """
        Lazy load single agent when needed.

        Loads agent metadata on-demand without loading all agents. Checks
        memory cache first, then loads from disk if not found.

        Args:
            agent_name: Name of agent to load (e.g., "python-pro")

        Returns:
            AgentMetadata if found, None otherwise

        Example:
            >>> agent = await loader.load_on_demand("python-pro")
            >>> if agent:
            ...     print(f"Found agent: {agent.description}")
        """
        start_time = asyncio.get_event_loop().time()

        # Check memory cache
        if agent_name in self._memory_cache:
            load_time = (asyncio.get_event_loop().time() - start_time) * 1000
            logger.debug(
                f"Agent '{agent_name}' loaded from memory cache "
                f"in {load_time:.2f}ms"
            )
            return self._memory_cache[agent_name]

        # Try loading from disk cache
        if not self._cache_loaded:
            cached_agents = await self._load_from_cache()
            if cached_agents and agent_name in cached_agents:
                self._memory_cache = cached_agents
                self._cache_loaded = True
                load_time = (asyncio.get_event_loop().time() - start_time) * 1000
                logger.debug(
                    f"Agent '{agent_name}' loaded from disk cache "
                    f"in {load_time:.2f}ms"
                )
                return cached_agents[agent_name]

        # Search for agent file
        agent_path = self._find_agent_file(agent_name)

        if agent_path is None:
            logger.warning(f"Agent '{agent_name}' not found")
            return None

        # Load agent file
        agent_meta = await self._load_agent_file(agent_path)

        if agent_meta:
            # Cache in memory
            self._memory_cache[agent_name] = agent_meta

            load_time = (asyncio.get_event_loop().time() - start_time) * 1000
            logger.info(
                f"Agent '{agent_name}' loaded from disk in {load_time:.2f}ms"
            )

        return agent_meta

    async def _load_agent_file_with_semaphore(
        self, file_path: Path
    ) -> Optional[AgentMetadata]:
        """
        Load agent file with concurrency limiting.

        Wraps _load_agent_file with semaphore to limit concurrent file I/O.

        Args:
            file_path: Path to agent markdown file

        Returns:
            AgentMetadata if successful, None otherwise
        """
        async with self._load_semaphore:
            return await self._load_agent_file(file_path)

    async def _load_agent_file(
        self, file_path: Path
    ) -> Optional[AgentMetadata]:
        """
        Load and parse single agent markdown file.

        Extracts metadata from frontmatter and content:
        - name: From frontmatter or filename
        - description: From frontmatter
        - tools: From frontmatter (comma-separated list)
        - keywords: Extracted from content
        - category: From parent directory name

        Args:
            file_path: Path to agent markdown file

        Returns:
            AgentMetadata if parsing successful, None otherwise
        """
        try:
            # Read file content
            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                content = await f.read()

            # Extract frontmatter (YAML between --- markers)
            frontmatter = self._extract_frontmatter(content)

            # Get agent name
            name = frontmatter.get("name", file_path.stem)

            # Get description
            description = frontmatter.get(
                "description",
                f"Agent: {name}"
            )

            # Get tools (parse comma-separated string)
            tools_str = frontmatter.get("tools", "")
            tools = [
                t.strip()
                for t in tools_str.split(",")
                if t.strip()
            ] if tools_str else []

            # Extract keywords from content
            keywords = self._extract_keywords(content)

            # Get category from parent directory
            category = file_path.parent.name

            # If nested in subdirectory, get grandparent
            if not category.startswith("0"):
                category = file_path.parent.parent.name

            # Get version (default to 1.0)
            version = frontmatter.get("version", "1.0")

            metadata = AgentMetadata(
                name=name,
                category=category,
                description=description,
                file_path=str(file_path.absolute()),
                tools=tools,
                keywords=keywords,
                version=version
            )

            logger.debug(f"Loaded agent: {name} ({category})")
            return metadata

        except Exception as e:
            logger.error(f"Failed to load agent file {file_path}: {e}")
            return None

    def _extract_frontmatter(self, content: str) -> Dict[str, str]:
        """
        Extract YAML frontmatter from markdown content.

        Parses frontmatter between --- markers at the start of the file.

        Args:
            content: Full markdown content

        Returns:
            Dictionary of frontmatter fields
        """
        frontmatter = {}

        # Match frontmatter pattern
        match = re.match(
            r"^---\s*\n(.*?)\n---\s*\n",
            content,
            re.DOTALL
        )

        if match:
            frontmatter_str = match.group(1)

            # Parse simple YAML (key: value pairs)
            for line in frontmatter_str.split("\n"):
                line = line.strip()
                if ":" in line:
                    key, value = line.split(":", 1)
                    frontmatter[key.strip()] = value.strip()

        return frontmatter

    def _extract_keywords(self, content: str) -> List[str]:
        """
        Extract keywords from agent content.

        Extracts relevant keywords from markdown content for semantic matching.
        Focuses on technical terms, technologies, and capabilities.

        Args:
            content: Markdown content

        Returns:
            List of extracted keywords
        """
        keywords: Set[str] = set()

        # Common technical patterns
        patterns = [
            r"\b(Python|TypeScript|JavaScript|Rust|Go|Java|C\+\+)\b",
            r"\b(React|Vue|Angular|Django|Flask|FastAPI|Express)\b",
            r"\b(Docker|Kubernetes|AWS|Azure|GCP)\b",
            r"\b(PostgreSQL|MySQL|MongoDB|Redis)\b",
            r"\b(async|await|testing|performance|security)\b",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            keywords.update(m.lower() for m in matches)

        # Limit to top 20 keywords
        return list(keywords)[:20]

    def _find_all_agent_files(self) -> List[Path]:
        """
        Find all agent markdown files in agent directory.

        Recursively searches for .md files in all category directories.

        Returns:
            List of paths to agent markdown files
        """
        agent_files: List[Path] = []

        if not self.agent_dir.exists():
            logger.error(f"Agent directory not found: {self.agent_dir}")
            return agent_files

        # Search recursively for .md files
        for md_file in self.agent_dir.rglob("*.md"):
            # Skip README and other docs
            if md_file.name.upper().startswith("README"):
                continue
            if md_file.name.upper().startswith("AGENT_"):
                continue
            if md_file.name.startswith("temp_"):
                continue

            agent_files.append(md_file)

        return agent_files

    def _find_agent_file(self, agent_name: str) -> Optional[Path]:
        """
        Find agent markdown file by name.

        Searches all category directories for matching agent file.

        Args:
            agent_name: Agent name (e.g., "python-pro")

        Returns:
            Path to agent file, or None if not found
        """
        if not self.agent_dir.exists():
            return None

        # Search in all category directories
        for category_dir in self.agent_dir.iterdir():
            if not category_dir.is_dir():
                continue

            # Try exact match
            agent_file = category_dir / f"{agent_name}.md"
            if agent_file.exists():
                return agent_file

            # Search recursively
            for agent_file in category_dir.rglob(f"{agent_name}.md"):
                return agent_file

        return None

    async def _load_from_cache(self) -> Optional[Dict[str, AgentMetadata]]:
        """
        Load agents from persistent disk cache.

        Validates cache integrity and freshness before returning cached data.
        Returns None if cache is invalid, expired, or missing.

        Returns:
            Dictionary of cached agents, or None if cache invalid
        """
        cache_file = self.cache_config.cache_file

        if not cache_file.exists():
            logger.debug("Cache file not found")
            return None

        try:
            # Read cache file
            async with aiofiles.open(cache_file, "r", encoding="utf-8") as f:
                cache_data = json.loads(await f.read())

            # Validate cache version
            if cache_data.get("cache_version") != self.cache_config.cache_version:
                logger.warning("Cache version mismatch, invalidating cache")
                return None

            # Validate workspace version
            if cache_data.get("workspace_version") != self.workspace_version:
                logger.warning("Workspace version mismatch, invalidating cache")
                return None

            # Check TTL
            timestamp_str = cache_data.get("timestamp")
            if timestamp_str:
                cache_time = datetime.fromisoformat(timestamp_str)
                now = datetime.now(timezone.utc)
                age = now - cache_time

                if age > timedelta(hours=self.cache_config.ttl_hours):
                    logger.info(
                        f"Cache expired (age: {age.total_seconds() / 3600:.1f}h)"
                    )
                    return None

            # Validate checksum
            agents_data = cache_data.get("agents", {})
            computed_checksum = self._compute_checksum(agents_data)
            stored_checksum = cache_data.get("checksum")

            if computed_checksum != stored_checksum:
                logger.warning("Cache checksum mismatch, invalidating cache")
                return None

            # Deserialize agents
            agents = {
                name: AgentMetadata.from_dict(meta)
                for name, meta in agents_data.items()
            }

            logger.info(f"Loaded {len(agents)} agents from cache (valid)")
            return agents

        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
            return None

    async def _save_to_cache(self, agents: Dict[str, AgentMetadata]) -> None:
        """
        Save agents to persistent disk cache.

        Creates cache file with metadata, checksum, and timestamp for
        validation on future loads.

        Args:
            agents: Dictionary of agents to cache
        """
        cache_file = self.cache_config.cache_file

        try:
            # Serialize agents
            agents_data = {
                name: meta.to_dict()
                for name, meta in agents.items()
            }

            # Compute checksum
            checksum = self._compute_checksum(agents_data)

            # Create cache structure
            cache_data = {
                "cache_version": self.cache_config.cache_version,
                "workspace_version": self.workspace_version,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "checksum": checksum,
                "agent_count": len(agents),
                "agents": agents_data
            }

            # Write to cache file
            async with aiofiles.open(cache_file, "w", encoding="utf-8") as f:
                await f.write(json.dumps(cache_data, indent=2))

            logger.info(
                f"Saved {len(agents)} agents to cache at {cache_file}"
            )

        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    def _compute_checksum(self, agents_data: Dict) -> str:
        """
        Compute checksum for cache integrity validation.

        Uses SHA256 hash of serialized agent data.

        Args:
            agents_data: Dictionary of serialized agent metadata

        Returns:
            Hex digest of SHA256 hash
        """
        # Sort keys for deterministic serialization
        serialized = json.dumps(agents_data, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()

    def invalidate_cache(self) -> None:
        """
        Clear persistent cache.

        Deletes cache file and clears memory cache. Use when agents have
        been updated or cache is suspected to be corrupt.

        Example:
            >>> loader.invalidate_cache()
            >>> # Next load will rebuild cache from disk
        """
        cache_file = self.cache_config.cache_file

        if cache_file.exists():
            cache_file.unlink()
            logger.info("Cache invalidated and deleted")

        # Clear memory cache
        self._memory_cache.clear()
        self._cache_loaded = False

        logger.info("Memory cache cleared")

    def get_cache_stats(self) -> Dict[str, any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        cache_file = self.cache_config.cache_file
        cache_exists = cache_file.exists()

        stats = {
            "cache_file": str(cache_file),
            "cache_exists": cache_exists,
            "memory_cache_size": len(self._memory_cache),
            "cache_loaded": self._cache_loaded,
        }

        if cache_exists:
            stats["cache_size_bytes"] = cache_file.stat().st_size
            stats["cache_size_kb"] = cache_file.stat().st_size / 1024

        return stats
