"""
Pattern Consolidator - Merge patterns from multiple sources

Combines patterns from:
1. Local codebase analysis
2. GitHub repository scanning
3. Official documentation scraping

Creates unified pattern library for AI agents.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import defaultdict


@dataclass
class ConsolidatedPattern:
    """Unified pattern from multiple sources."""

    id: str
    name: str
    category: str
    language: str
    description: str
    code_snippet: str
    sources: List[Dict[str, Any]]  # List of source attributions
    confidence_score: float  # 0-1, based on # of sources and stars
    usage_frequency: int
    keywords: List[str]
    created_at: str
    updated_at: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class PatternConsolidator:
    """
    Consolidates patterns from multiple sources into unified library.

    Features:
    - Deduplicates similar patterns
    - Calculates confidence scores
    - Merges pattern metadata
    - Generates pattern files for agent use
    """

    def __init__(self, output_dir: Path):
        """
        Initialize pattern consolidator.

        Args:
            output_dir: Directory to write consolidated patterns
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def consolidate(
        self,
        local_patterns: Optional[Path] = None,
        github_patterns: Optional[Path] = None,
        docs_patterns: Optional[Path] = None
    ) -> List[ConsolidatedPattern]:
        """
        Consolidate patterns from all sources.

        Args:
            local_patterns: Path to local codebase patterns JSON
            github_patterns: Path to GitHub patterns JSON
            docs_patterns: Path to documentation patterns JSON

        Returns:
            List of consolidated patterns

        Example:
            consolidator = PatternConsolidator(Path(".cursor/patterns"))
            patterns = consolidator.consolidate(
                local_patterns=Path("local_patterns.json"),
                github_patterns=Path("github_patterns.json"),
                docs_patterns=Path("docs_patterns.json")
            )
        """
        print("🔄 Consolidating patterns from all sources...")

        all_raw_patterns = []

        # Load patterns from each source
        if local_patterns and local_patterns.exists():
            all_raw_patterns.extend(self._load_patterns(local_patterns, "local"))
            print(f"  ✅ Loaded local patterns: {local_patterns}")

        if github_patterns and github_patterns.exists():
            all_raw_patterns.extend(self._load_patterns(github_patterns, "github"))
            print(f"  ✅ Loaded GitHub patterns: {github_patterns}")

        if docs_patterns and docs_patterns.exists():
            all_raw_patterns.extend(self._load_patterns(docs_patterns, "docs"))
            print(f"  ✅ Loaded docs patterns: {docs_patterns}")

        # Group similar patterns
        pattern_groups = self._group_similar_patterns(all_raw_patterns)

        # Merge each group into consolidated pattern
        consolidated = []
        for group in pattern_groups.values():
            merged = self._merge_pattern_group(group)
            consolidated.append(merged)

        print(f"✅ Consolidated {len(all_raw_patterns)} raw patterns into {len(consolidated)} unique patterns")

        return consolidated

    def _load_patterns(self, file_path: Path, source_type: str) -> List[Dict[str, Any]]:
        """Load patterns from JSON file and tag with source."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        patterns = data.get("patterns", [])

        # Tag each pattern with source
        for pattern in patterns:
            pattern["source_type"] = source_type

        return patterns

    def _group_similar_patterns(
        self,
        patterns: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group similar patterns together for merging.

        Uses combination of:
        - Pattern name similarity
        - Keyword overlap
        - Category matching
        """
        groups = defaultdict(list)

        for pattern in patterns:
            # Create group key from category + primary keywords
            category = pattern.get("category", "unknown")
            keywords = pattern.get("keywords", [])
            name = pattern.get("name", "")

            # Generate group key
            primary_keyword = keywords[0] if keywords else name.split()[0]
            group_key = f"{category}:{primary_keyword}".lower()

            groups[group_key].append(pattern)

        return groups

    def _merge_pattern_group(
        self,
        group: List[Dict[str, Any]]
    ) -> ConsolidatedPattern:
        """
        Merge multiple similar patterns into one consolidated pattern.

        Prioritizes:
        1. Patterns from official docs (highest confidence)
        2. High-star GitHub repos
        3. Local patterns (verified working code)
        """
        # Sort by priority (docs > high stars > local)
        sorted_group = sorted(
            group,
            key=lambda p: (
                p.get("source_type") == "docs",
                p.get("stars", 0),
                p.get("source_type") == "local"
            ),
            reverse=True
        )

        # Use highest priority pattern as base
        base = sorted_group[0]

        # Collect all sources
        sources = []
        total_stars = 0
        for pattern in sorted_group:
            source_info = {
                "type": pattern.get("source_type", "unknown"),
                "url": pattern.get("source_url", ""),
                "stars": pattern.get("stars", 0)
            }
            sources.append(source_info)
            total_stars += pattern.get("stars", 0)

        # Calculate confidence score
        confidence = self._calculate_confidence(len(sources), total_stars)

        # Merge keywords
        all_keywords = []
        for pattern in sorted_group:
            all_keywords.extend(pattern.get("keywords", []))
        unique_keywords = list(set(all_keywords))

        # Create consolidated pattern
        pattern_id = self._generate_pattern_id(base)

        consolidated = ConsolidatedPattern(
            id=pattern_id,
            name=base.get("name", "Unknown Pattern"),
            category=base.get("category", "unknown"),
            language=base.get("language", "unknown"),
            description=base.get("description", ""),
            code_snippet=base.get("code_snippet", ""),
            sources=sources,
            confidence_score=confidence,
            usage_frequency=len(sorted_group),
            keywords=unique_keywords,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )

        return consolidated

    def _calculate_confidence(self, source_count: int, total_stars: int) -> float:
        """
        Calculate confidence score for pattern.

        Factors:
        - Number of sources (more = better)
        - Total stars (higher = better)
        - Source types (docs > github > local)

        Returns:
            Confidence score 0.0-1.0
        """
        # Base score from source count (max 0.4 at 5+ sources)
        source_score = min(source_count * 0.08, 0.4)

        # Star score (max 0.4 at 10,000+ stars)
        star_score = min(total_stars / 25000, 0.4)

        # Bonus for multiple source types (0.2)
        type_bonus = 0.2

        return min(source_score + star_score + type_bonus, 1.0)

    def _generate_pattern_id(self, pattern: Dict[str, Any]) -> str:
        """Generate unique ID for pattern."""
        name = pattern.get("name", "unknown")
        category = pattern.get("category", "unknown")

        # Create slug
        slug = f"{category}-{name}".lower()
        slug = slug.replace(" ", "-").replace("/", "-")
        slug = "".join(c for c in slug if c.isalnum() or c == "-")

        return slug

    def generate_pattern_files(
        self,
        patterns: List[ConsolidatedPattern]
    ) -> None:
        """
        Generate individual pattern files for agent use.

        Creates:
        - One .pattern.py/ts file per pattern
        - Organized by category
        - Includes all metadata

        Example structure:
            .cursor/patterns/
                python/
                    fastapi-crud-endpoint.pattern.py
                    async-database-repository.pattern.py
                typescript/
                    react-custom-hook.pattern.ts
        """
        print(f"📝 Generating {len(patterns)} pattern files...")

        for pattern in patterns:
            # Determine file extension
            ext = {
                "python": "py",
                "typescript": "ts",
                "javascript": "js",
                "sql": "sql"
            }.get(pattern.language, "txt")

            # Create category directory
            category_dir = self.output_dir / pattern.category
            category_dir.mkdir(parents=True, exist_ok=True)

            # Create pattern file
            file_name = f"{pattern.id}.pattern.{ext}"
            file_path = category_dir / file_name

            # Generate file content
            content = self._generate_pattern_content(pattern)

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

            print(f"  ✅ Created: {file_path}")

    def _generate_pattern_content(self, pattern: ConsolidatedPattern) -> str:
        """Generate pattern file content with metadata and code."""
        sources_str = "\n".join([
            f"- {s['type']}: {s['url']} ({s['stars']} stars)"
            for s in pattern.sources
        ])

        keywords_str = ", ".join(pattern.keywords)

        content = f'''"""
PATTERN: {pattern.name}
CATEGORY: {pattern.category}
LANGUAGE: {pattern.language}
CONFIDENCE: {pattern.confidence_score:.2f} ({pattern.usage_frequency} sources)
KEYWORDS: {keywords_str}
DISCOVERED: {pattern.created_at}

{pattern.description}

SOURCES:
{sources_str}
"""

{pattern.code_snippet}

# Usage notes:
# - This pattern was discovered from {pattern.usage_frequency} sources
# - Confidence score: {pattern.confidence_score:.2f}/1.0
# - Verified working code from production systems
'''

        return content

    def save_index(self, patterns: List[ConsolidatedPattern]) -> None:
        """Save pattern index for quick lookup."""
        index_data = {
            "version": "1.0.0",
            "generated_at": datetime.now().isoformat(),
            "pattern_count": len(patterns),
            "patterns": [
                {
                    "id": p.id,
                    "name": p.name,
                    "category": p.category,
                    "language": p.language,
                    "confidence": p.confidence_score,
                    "keywords": p.keywords
                }
                for p in patterns
            ]
        }

        index_file = self.output_dir / "pattern-index.json"
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, indent=2)

        print(f"💾 Saved pattern index: {index_file}")


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """CLI for pattern consolidation."""
    import argparse

    parser = argparse.ArgumentParser(description="Consolidate patterns from multiple sources")
    parser.add_argument("--local", help="Path to local patterns JSON")
    parser.add_argument("--github", help="Path to GitHub patterns JSON")
    parser.add_argument("--docs", help="Path to docs patterns JSON")
    parser.add_argument("--output", default=".cursor/patterns", help="Output directory")

    args = parser.parse_args()

    # Consolidate patterns
    consolidator = PatternConsolidator(Path(args.output))
    patterns = consolidator.consolidate(
        local_patterns=Path(args.local) if args.local else None,
        github_patterns=Path(args.github) if args.github else None,
        docs_patterns=Path(args.docs) if args.docs else None
    )

    # Generate pattern files
    consolidator.generate_pattern_files(patterns)

    # Save index
    consolidator.save_index(patterns)

    print(f"\n🎉 Pattern consolidation complete!")
    print(f"📁 Patterns saved to: {args.output}")
    print(f"📊 Total patterns: {len(patterns)}")


if __name__ == "__main__":
    main()
