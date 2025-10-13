#!/usr/bin/env python3
"""
Lazy Agent Loader
Loads only relevant agents based on detected stack for improved performance.

Usage:
    python agent_loader.py --stack-config .ai-workspace-config.yml
    python agent_loader.py --languages python typescript
"""

from typing import List, Set, Dict
from pathlib import Path
import yaml
import sys
import argparse


class AgentLoader:
    """Lazy loading system for agents."""

    # Map languages to agent categories
    LANGUAGE_CATEGORIES = {
        'python': ['02-languages/scripting', '03-frameworks/backend'],
        'typescript': ['02-languages/web', '03-frameworks/frontend', '03-frameworks/backend'],
        'javascript': ['02-languages/web', '03-frameworks/frontend'],
        'rust': ['02-languages/systems'],
        'go': ['02-languages/systems'],
        'java': ['02-languages/systems', '03-frameworks/backend'],
        'kotlin': ['03-frameworks/mobile'],
        'swift': ['03-frameworks/mobile'],
    }

    # Map frameworks to agent categories
    FRAMEWORK_CATEGORIES = {
        'react': ['03-frameworks/frontend'],
        'vue': ['03-frameworks/frontend'],
        'angular': ['03-frameworks/frontend'],
        'nextjs': ['03-frameworks/frontend'],
        'fastapi': ['03-frameworks/backend'],
        'django': ['03-frameworks/backend'],
        'flask': ['03-frameworks/backend'],
        'spring': ['03-frameworks/backend'],
        'laravel': ['03-frameworks/backend'],
        'rails': ['03-frameworks/backend'],
    }

    # Always load these categories (core functionality)
    CORE_CATEGORIES = [
        '00-orchestration',
        '05-quality/review',
        '05-quality/performance',
    ]

    def __init__(self, workspace_path: Path = Path(".ai-workspace")):
        """Initialize loader."""
        self.workspace_path = workspace_path
        self.agents_dir = workspace_path / "agents"
        self.cache_dir = workspace_path / "cache"
        self.cache_dir.mkdir(exist_ok=True)

    def load_relevant_agents(
        self,
        languages: List[str] = None,
        frameworks: List[str] = None,
        load_all: bool = False
    ) -> Dict[str, List[Path]]:
        """
        Load relevant agents based on stack.

        Args:
            languages: List of programming languages
            frameworks: List of frameworks
            load_all: Load all agents (disable lazy loading)

        Returns:
            Dictionary mapping categories to agent files
        """
        if load_all:
            return self._load_all_agents()

        # Determine which categories to load
        categories_to_load = set(self.CORE_CATEGORIES)

        # Add language-specific categories
        if languages:
            for lang in languages:
                lang_lower = lang.lower()
                if lang_lower in self.LANGUAGE_CATEGORIES:
                    categories_to_load.update(self.LANGUAGE_CATEGORIES[lang_lower])

        # Add framework-specific categories
        if frameworks:
            for fw in frameworks:
                fw_lower = fw.lower().replace('.', '').replace('-', '')
                if fw_lower in self.FRAMEWORK_CATEGORIES:
                    categories_to_load.update(self.FRAMEWORK_CATEGORIES[fw_lower])

        # Load agents from selected categories
        loaded_agents = {}

        for category in categories_to_load:
            category_path = self.agents_dir / category
            if category_path.exists():
                agents = self._load_agents_from_category(category_path)
                if agents:
                    loaded_agents[category] = agents

        # Always include some utility categories
        utility_categories = ['08-support/documentation', '09-utilities']
        for category in utility_categories:
            category_path = self.agents_dir / category
            if category_path.exists() and category not in loaded_agents:
                agents = self._load_agents_from_category(category_path)
                if agents:
                    loaded_agents[category] = agents

        return loaded_agents

    def _load_all_agents(self) -> Dict[str, List[Path]]:
        """Load all available agents."""
        all_agents = {}

        for category_dir in self.agents_dir.iterdir():
            if category_dir.is_dir():
                category_name = category_dir.name
                agents = self._load_agents_from_category(category_dir)
                if agents:
                    all_agents[category_name] = agents

        return all_agents

    def _load_agents_from_category(self, category_path: Path) -> List[Path]:
        """Load all agent files from a category directory."""
        agents = []

        for agent_file in category_path.rglob("*.md"):
            # Skip README and guide files
            if agent_file.name in ["README.md", "AGENT_SELECTION_GUIDE.md", "temp_AGENT_SELECTION_GUIDE.md"]:
                continue

            agents.append(agent_file)

        return agents

    def load_from_config(self, config_file: Path) -> Dict[str, List[Path]]:
        """Load agents based on stack configuration file."""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            stack = config.get('stack', {})
            languages = stack.get('languages', [])
            frameworks = stack.get('frameworks', [])

            return self.load_relevant_agents(languages, frameworks)

        except Exception as e:
            print(f"Error loading config: {e}")
            print("Falling back to loading all agents")
            return self._load_all_agents()

    def print_loaded_agents(self, loaded_agents: Dict[str, List[Path]]) -> None:
        """Print summary of loaded agents."""
        total_agents = sum(len(agents) for agents in loaded_agents.values())

        print(f"\n📦 Lazy Agent Loading Summary")
        print("=" * 70)
        print(f"Total agents loaded: {total_agents}")
        print(f"Categories loaded: {len(loaded_agents)}\n")

        for category, agents in sorted(loaded_agents.items()):
            print(f"📁 {category}: {len(agents)} agents")
            for agent in sorted(agents[:5]):  # Show first 5
                print(f"   • {agent.stem}")
            if len(agents) > 5:
                print(f"   ... and {len(agents) - 5} more")
            print()

        print("=" * 70)

        # Calculate percentage saved
        all_agents_count = len(list(self.agents_dir.rglob("*.md")))
        saved = ((all_agents_count - total_agents) / all_agents_count) * 100

        print(f"\n⚡ Performance:")
        print(f"   Loaded: {total_agents}/{all_agents_count} agents")
        print(f"   Saved: {saved:.1f}% loading time")
        print(f"   Memory: ~{(all_agents_count - total_agents) * 5}KB saved (estimated)")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Lazy load agents based on stack configuration"
    )
    parser.add_argument(
        "--stack-config",
        type=Path,
        help="Path to stack configuration file (.ai-workspace-config.yml)"
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        help="Languages to load agents for"
    )
    parser.add_argument(
        "--frameworks",
        nargs="+",
        help="Frameworks to load agents for"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Load all agents (disable lazy loading)"
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path(".ai-workspace"),
        help="Path to workspace (default: .ai-workspace)"
    )

    args = parser.parse_args()

    try:
        loader = AgentLoader(args.workspace)

        if args.stack_config:
            loaded_agents = loader.load_from_config(args.stack_config)
        else:
            loaded_agents = loader.load_relevant_agents(
                languages=args.languages,
                frameworks=args.frameworks,
                load_all=args.all
            )

        loader.print_loaded_agents(loaded_agents)
        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️  Cancelled")
        return 1

    except Exception as e:
        print(f"\n❌ Failed to load agents: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
