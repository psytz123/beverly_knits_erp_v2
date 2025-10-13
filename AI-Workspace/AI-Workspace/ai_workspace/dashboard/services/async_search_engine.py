"""Async wrapper for search engine with non-blocking operations.

Provides async interface to search engine with parallel index building
and concurrent search capabilities.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .search_engine import SearchEngine, SearchResult

logger = logging.getLogger(__name__)


class AsyncSearchEngine:
    """Async wrapper for search engine with non-blocking I/O.

    Wraps synchronous search engine operations in async interface
    to prevent blocking the event loop during index building and searching.
    """

    def __init__(
        self, workspace_root: Optional[Path] = None, max_workers: int = 4
    ) -> None:
        """Initialize async search engine.

        Args:
            workspace_root: Root directory of workspace
            max_workers: Maximum number of thread pool workers
        """
        self.engine = SearchEngine(workspace_root)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._indexes_built = False

    async def rebuild_indexes_async(self) -> None:
        """Rebuild search indexes asynchronously.

        Runs index building in thread pool to avoid blocking event loop.
        """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self.executor, self.engine.rebuild_indexes)
        self._indexes_built = True
        logger.info("Search indexes rebuilt asynchronously")

    async def search_async(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 20,
    ) -> Tuple[List[SearchResult], int, float]:
        """Execute search query asynchronously.

        Args:
            query: Search query string
            filters: Optional filters
            limit: Maximum results

        Returns:
            Tuple of (results, total_count, query_time_ms)
        """
        # Ensure indexes are built
        if not self._indexes_built:
            await self.rebuild_indexes_async()

        # Run search in thread pool
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            self.executor, self.engine.search, query, filters, limit
        )

        return results

    async def get_stats_async(self) -> Dict[str, Any]:
        """Get search index statistics asynchronously.

        Returns:
            Dictionary with index statistics
        """
        if not self._indexes_built:
            await self.rebuild_indexes_async()

        loop = asyncio.get_event_loop()

        # Collect statistics
        stats = await loop.run_in_executor(self.executor, self._collect_stats)

        return stats

    def _collect_stats(self) -> Dict[str, Any]:
        """Collect statistics (runs in thread pool)."""
        indexes_stats = {}

        if self.engine._reuse_index:
            indexes_stats["reuse_checks"] = {
                "total_documents": self.engine._reuse_index.total_docs,
                "total_terms": len(self.engine._reuse_index.document_frequencies),
                "fields_indexed": self.engine._reuse_index.field_names,
            }

        if self.engine._gates_index:
            indexes_stats["phase_gates"] = {
                "total_documents": self.engine._gates_index.total_docs,
                "total_terms": len(self.engine._gates_index.document_frequencies),
                "fields_indexed": self.engine._gates_index.field_names,
            }

        if self.engine._agents_index:
            indexes_stats["agents"] = {
                "total_documents": self.engine._agents_index.total_docs,
                "total_terms": len(self.engine._agents_index.document_frequencies),
                "fields_indexed": self.engine._agents_index.field_names,
            }

        total_docs = sum(idx["total_documents"] for idx in indexes_stats.values())
        total_terms = sum(idx["total_terms"] for idx in indexes_stats.values())

        return {
            "indexes": indexes_stats,
            "total_documents": total_docs,
            "total_unique_terms": total_terms,
        }

    async def get_suggestions_async(
        self, prefix: str = "", limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get search suggestions asynchronously.

        Args:
            prefix: Partial query string
            limit: Maximum suggestions

        Returns:
            List of suggestion dictionaries
        """
        if not self._indexes_built:
            await self.rebuild_indexes_async()

        loop = asyncio.get_event_loop()
        suggestions = await loop.run_in_executor(
            self.executor, self._get_suggestions, prefix, limit
        )

        return suggestions

    def _get_suggestions(self, prefix: str, limit: int) -> List[Dict[str, Any]]:
        """Get suggestions (runs in thread pool)."""
        all_terms: Dict[str, int] = {}

        for index in [
            self.engine._reuse_index,
            self.engine._gates_index,
            self.engine._agents_index,
        ]:
            if index:
                for term, freq in index.document_frequencies.items():
                    if not prefix or term.startswith(prefix.lower()):
                        all_terms[term] = all_terms.get(term, 0) + freq

        # Sort by frequency
        sorted_terms = sorted(all_terms.items(), key=lambda x: x[1], reverse=True)
        return [{"term": term, "frequency": freq} for term, freq in sorted_terms[:limit]]

    async def close(self) -> None:
        """Clean up resources."""
        self.executor.shutdown(wait=False)
