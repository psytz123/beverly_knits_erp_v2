"""Whoosh-based full-text search index for AI Workspace Dashboard.

Provides high-performance full-text search indexing with <100ms query times
for large codebases. Replaces O(n) file scanning with persistent index.

Performance Targets:
- Index build: <5s for 156 agents
- Search time: <100ms for 1000 files, <1s for 10000 files
- Index size: <10MB for typical workspace
- Incremental updates: <50ms per file
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from whoosh import index
from whoosh.fields import DATETIME, ID, KEYWORD, TEXT, Schema
from whoosh.qparser import MultifieldParser, QueryParser
from whoosh.query import And, Or, Term
from whoosh.writing import AsyncWriter, CLEAR

logger = logging.getLogger(__name__)

# Constants
DEFAULT_INDEX_DIR = ".ai-workspace/cache/search_index"
DEFAULT_RESULT_LIMIT = 50
INDEX_SCHEMA_VERSION = "1.0.0"


class SearchIndex:
    """Full-text search index using Whoosh for fast agent and code search.

    Features:
    - Multi-field indexing (name, description, content, tags, category)
    - Full-text search with relevance scoring
    - Filter support (category, tags, date range)
    - Incremental updates (add/update/delete)
    - Async writes for performance
    - Index optimization and maintenance

    Attributes:
        index_dir: Directory for persistent index storage
        schema: Whoosh schema definition
        idx: Whoosh index instance
        parser: Multi-field query parser
    """

    def __init__(self, index_dir: Optional[Path] = None) -> None:
        """Initialize search index.

        Args:
            index_dir: Directory for index storage (default: .ai-workspace/cache/search_index)
        """
        if index_dir is None:
            index_dir = Path.cwd() / DEFAULT_INDEX_DIR

        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        # Define search schema
        self.schema = Schema(
            path=ID(stored=True, unique=True),  # Unique file path
            name=TEXT(stored=True, field_boost=2.0),  # Agent name (boosted)
            category=KEYWORD(stored=True, lowercase=True),  # Agent category
            description=TEXT(stored=True, field_boost=1.5),  # Description (boosted)
            content=TEXT(stored=False),  # Full content (not stored, only indexed)
            tags=KEYWORD(stored=True, lowercase=True, commas=True),  # Tags
            modified=DATETIME(stored=True),  # Last modified timestamp
            file_size=ID(stored=True),  # File size in bytes
        )

        # Create or open index
        if index.exists_in(str(self.index_dir)):
            self.idx = index.open_dir(str(self.index_dir))
            logger.info(f"Opened existing search index at {self.index_dir}")
        else:
            self.idx = index.create_in(str(self.index_dir), self.schema)
            logger.info(f"Created new search index at {self.index_dir}")

        # Multi-field query parser with field weights
        self.parser = MultifieldParser(
            ["name", "description", "content", "tags"],
            schema=self.schema,
            fieldboosts={
                "name": 2.0,
                "description": 1.5,
                "tags": 1.8,
                "content": 1.0,
            }
        )

    def index_file(
        self,
        file_path: Path,
        metadata: Dict[str, Any],
        async_write: bool = True
    ) -> None:
        """Add or update a single file in the index.

        Args:
            file_path: Path to the file
            metadata: File metadata including name, description, content, etc.
            async_write: Use async writer for better performance (default: True)

        Raises:
            Exception: If indexing fails
        """
        try:
            # Prepare document data
            doc_data = {
                "path": str(file_path),
                "name": metadata.get("name", file_path.stem),
                "category": metadata.get("category", ""),
                "description": metadata.get("description", ""),
                "content": metadata.get("content", ""),
                "tags": metadata.get("tags", ""),
                "modified": metadata.get("modified", datetime.now()),
                "file_size": str(metadata.get("file_size", 0)),
            }

            # Use async writer for performance
            if async_write:
                writer = AsyncWriter(self.idx)
            else:
                writer = self.idx.writer()

            try:
                # Update or add document (upsert operation)
                writer.update_document(**doc_data)
                writer.commit()
                logger.debug(f"Indexed: {file_path}")
            except Exception as e:
                writer.cancel()
                raise Exception(f"Failed to index {file_path}: {e}") from e

        except Exception as e:
            logger.error(f"Error indexing file {file_path}: {e}")
            raise

    def search(
        self,
        query: str,
        limit: int = DEFAULT_RESULT_LIMIT,
        filters: Optional[Dict[str, Any]] = None,
        highlight: bool = False
    ) -> List[Dict[str, Any]]:
        """Execute full-text search query.

        Args:
            query: Search query string
            limit: Maximum number of results (default: 50)
            filters: Optional filters (category, tags, date_range)
            highlight: Return highlighted snippets (default: False)

        Returns:
            List of search results with scores and metadata
        """
        if not query.strip():
            return []

        with self.idx.searcher() as searcher:
            # Parse main query
            try:
                q = self.parser.parse(query)
            except Exception as e:
                logger.warning(f"Query parse error: {e}, using simple query")
                q = QueryParser("content", self.schema).parse(query)

            # Apply filters
            if filters:
                filter_queries = []

                # Category filter
                if "category" in filters and filters["category"]:
                    categories = filters["category"] if isinstance(filters["category"], list) else [filters["category"]]
                    for cat in categories:
                        filter_queries.append(Term("category", cat.lower()))

                # Tags filter
                if "tags" in filters and filters["tags"]:
                    tags = filters["tags"] if isinstance(filters["tags"], list) else [filters["tags"]]
                    for tag in tags:
                        filter_queries.append(Term("tags", tag.lower()))

                # Combine filters with OR, then AND with main query
                if filter_queries:
                    if len(filter_queries) == 1:
                        q = And([q, filter_queries[0]])
                    else:
                        q = And([q, Or(filter_queries)])

            # Execute search
            try:
                results = searcher.search(q, limit=limit)

                # Convert results to dictionaries
                output = []
                for hit in results:
                    result_dict = {
                        "path": hit["path"],
                        "name": hit["name"],
                        "category": hit.get("category", ""),
                        "description": hit.get("description", ""),
                        "score": hit.score,
                        "rank": hit.rank + 1,
                    }

                    # Add highlights if requested
                    if highlight and "content" in hit:
                        result_dict["highlights"] = hit.highlights("content", top=3)

                    output.append(result_dict)

                return output

            except Exception as e:
                logger.error(f"Search execution error: {e}")
                return []

    def rebuild_index(
        self,
        files: List[Tuple[Path, Dict[str, Any]]],
        show_progress: bool = False
    ) -> None:
        """Rebuild entire index from scratch.

        Args:
            files: List of (file_path, metadata) tuples
            show_progress: Print progress messages (default: False)

        Raises:
            Exception: If rebuild fails
        """
        writer = self.idx.writer()

        try:
            # Clear existing index
            writer.mergetype = CLEAR

            total = len(files)
            for idx_num, (file_path, metadata) in enumerate(files, 1):
                try:
                    # Prepare document
                    doc_data = {
                        "path": str(file_path),
                        "name": metadata.get("name", file_path.stem),
                        "category": metadata.get("category", ""),
                        "description": metadata.get("description", ""),
                        "content": metadata.get("content", ""),
                        "tags": metadata.get("tags", ""),
                        "modified": metadata.get("modified", datetime.now()),
                        "file_size": str(metadata.get("file_size", 0)),
                    }

                    # Add to index
                    writer.add_document(**doc_data)

                    if show_progress and idx_num % 10 == 0:
                        logger.info(f"Indexed {idx_num}/{total} files")

                except Exception as e:
                    logger.warning(f"Skipping {file_path}: {e}")
                    continue

            # Commit changes
            writer.commit(optimize=True)
            logger.info(f"Index rebuilt: {total} files indexed")

        except Exception as e:
            writer.cancel()
            logger.error(f"Index rebuild failed: {e}")
            raise

    def delete_document(self, file_path: Path) -> None:
        """Remove a document from the index.

        Args:
            file_path: Path to the file to remove
        """
        writer = self.idx.writer()
        try:
            writer.delete_by_term("path", str(file_path))
            writer.commit()
            logger.debug(f"Deleted from index: {file_path}")
        except Exception as e:
            writer.cancel()
            logger.error(f"Failed to delete {file_path}: {e}")
            raise

    def delete_by_category(self, category: str) -> int:
        """Remove all documents in a category.

        Args:
            category: Category name

        Returns:
            Number of documents deleted
        """
        writer = self.idx.writer()
        try:
            deleted = writer.delete_by_term("category", category.lower())
            writer.commit()
            logger.info(f"Deleted {deleted} documents from category '{category}'")
            return deleted
        except Exception as e:
            writer.cancel()
            logger.error(f"Failed to delete category {category}: {e}")
            raise

    def optimize(self) -> None:
        """Optimize index for better search performance.

        Merges segments and removes deleted documents.
        Should be called periodically for maintenance.
        """
        try:
            writer = self.idx.writer()
            writer.commit(optimize=True)
            logger.info("Index optimized successfully")
        except Exception as e:
            logger.error(f"Index optimization failed: {e}")
            raise

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics.

        Returns:
            Dictionary with index statistics
        """
        with self.idx.searcher() as searcher:
            stats = {
                "total_docs": searcher.doc_count_all(),
                "unique_terms": sum(1 for _ in searcher.lexicon("content")),
                "index_version": INDEX_SCHEMA_VERSION,
                "index_path": str(self.index_dir),
            }

            # Get category breakdown
            category_counts: Dict[str, int] = {}
            for doc_num in range(searcher.doc_count_all()):
                try:
                    doc = searcher.stored_fields(doc_num)
                    cat = doc.get("category", "unknown")
                    category_counts[cat] = category_counts.get(cat, 0) + 1
                except Exception:
                    continue

            stats["categories"] = category_counts

            # Get index size
            try:
                index_size = sum(
                    f.stat().st_size for f in self.index_dir.rglob("*") if f.is_file()
                )
                stats["index_size_mb"] = round(index_size / (1024 * 1024), 2)
            except Exception as e:
                logger.warning(f"Could not calculate index size: {e}")
                stats["index_size_mb"] = 0

            return stats

    def incremental_update(
        self,
        workspace_dir: Path,
        file_extensions: Optional[Set[str]] = None
    ) -> Tuple[int, int, int]:
        """Incrementally update index based on file modification times.

        Args:
            workspace_dir: Workspace directory to scan
            file_extensions: Set of file extensions to index (default: {".md"})

        Returns:
            Tuple of (added, updated, deleted) counts
        """
        if file_extensions is None:
            file_extensions = {".md"}

        added = 0
        updated = 0
        deleted = 0

        # Get current indexed files
        indexed_files: Dict[str, datetime] = {}
        with self.idx.searcher() as searcher:
            for doc_num in range(searcher.doc_count_all()):
                try:
                    doc = searcher.stored_fields(doc_num)
                    indexed_files[doc["path"]] = doc.get("modified", datetime.min)
                except Exception:
                    continue

        # Scan workspace for files
        current_files: Set[str] = set()
        for ext in file_extensions:
            for file_path in workspace_dir.rglob(f"*{ext}"):
                if file_path.is_file():
                    current_files.add(str(file_path))

                    # Check if file needs indexing
                    file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)

                    if str(file_path) not in indexed_files:
                        # New file
                        self._index_single_file(file_path)
                        added += 1
                    elif file_mtime > indexed_files[str(file_path)]:
                        # Modified file
                        self._index_single_file(file_path)
                        updated += 1

        # Remove deleted files from index
        for indexed_path in indexed_files:
            if indexed_path not in current_files:
                self.delete_document(Path(indexed_path))
                deleted += 1

        logger.info(f"Incremental update: +{added} ~{updated} -{deleted}")
        return added, updated, deleted

    def _index_single_file(self, file_path: Path) -> None:
        """Index a single file (internal helper).

        Args:
            file_path: Path to file
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Extract metadata from markdown
            metadata = self._extract_metadata(content, file_path)

            # Index the file
            self.index_file(file_path, metadata, async_write=True)

        except Exception as e:
            logger.warning(f"Could not index {file_path}: {e}")

    def _extract_metadata(self, content: str, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from markdown content.

        Args:
            content: File content
            file_path: Path to file

        Returns:
            Metadata dictionary
        """
        import re

        metadata: Dict[str, Any] = {
            "name": file_path.stem,
            "category": file_path.parent.name,
            "content": content,
            "modified": datetime.fromtimestamp(file_path.stat().st_mtime),
            "file_size": file_path.stat().st_size,
        }

        # Extract YAML frontmatter
        yaml_match = re.match(r'^---\s*\n(.*?)\n---', content, re.DOTALL)
        if yaml_match:
            yaml_content = yaml_match.group(1)

            # Parse key-value pairs
            for line in yaml_content.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip().strip('"\'')

                    if key in ["name", "description"]:
                        metadata[key] = value
                    elif key == "tags":
                        metadata["tags"] = value

        # Fallback: extract from first heading
        if not metadata.get("name"):
            heading_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
            if heading_match:
                metadata["name"] = heading_match.group(1).strip()

        # Fallback: extract description from first paragraph
        if not metadata.get("description"):
            desc_match = re.search(r'\n\n(.+?)\n', content)
            if desc_match:
                metadata["description"] = desc_match.group(1).strip()[:200]

        return metadata

    def close(self) -> None:
        """Close the index."""
        if hasattr(self, 'idx'):
            self.idx.close()
            logger.debug("Search index closed")
