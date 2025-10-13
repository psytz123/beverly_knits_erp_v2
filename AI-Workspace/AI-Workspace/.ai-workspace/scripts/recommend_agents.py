#!/usr/bin/env python3
"""
Agent Recommendation Engine
Suggests best agents for specific tasks based on semantic matching.

Usage:
    python recommend_agents.py "build REST API"
    python recommend_agents.py "optimize database queries" --top 5
    python recommend_agents.py "setup CI/CD" --category infrastructure
"""

from typing import List, Dict, Tuple
from dataclasses import dataclass
from pathlib import Path
import re
import sys
import argparse


@dataclass
class AgentRecommendation:
    """Represents a recommended agent."""
    name: str
    category: str
    file_path: str
    confidence: float
    matching_keywords: List[str]
    description: str = ""


class AgentRecommender:
    """Recommend agents based on task descriptions."""

    # Keyword mappings for different domains
    KEYWORD_MAPPINGS = {
        # Backend development
        'api': ['backend-developer', 'api-designer', 'graphql-architect', 'websocket-engineer'],
        'rest': ['backend-developer', 'api-designer'],
        'graphql': ['graphql-architect', 'api-designer'],
        'backend': ['backend-developer', 'api-designer', 'microservices-architect'],

        # Frontend development
        'frontend': ['frontend-developer', 'react-specialist', 'vue-expert', 'angular-architect'],
        'ui': ['frontend-developer', 'ui-designer'],
        'react': ['react-specialist', 'nextjs-developer', 'frontend-developer'],
        'vue': ['vue-expert', 'frontend-developer'],
        'angular': ['angular-architect', 'frontend-developer'],
        'nextjs': ['nextjs-developer', 'react-specialist'],

        # Languages
        'python': ['python-pro', 'backend-developer', 'django-developer'],
        'typescript': ['typescript-pro', 'frontend-developer', 'backend-developer'],
        'javascript': ['javascript-pro', 'frontend-developer'],
        'rust': ['rust-engineer', 'backend-developer'],
        'go': ['golang-pro', 'backend-developer', 'microservices-architect'],
        'java': ['java-architect', 'spring-boot-engineer'],

        # Databases
        'database': ['database-optimizer', 'postgres-pro', 'database-administrator'],
        'postgres': ['postgres-pro', 'database-optimizer'],
        'postgresql': ['postgres-pro', 'database-optimizer'],
        'sql': ['database-optimizer', 'sql-pro', 'postgres-pro'],
        'mongodb': ['database-administrator', 'data-engineer'],

        # Infrastructure
        'kubernetes': ['kubernetes-specialist', 'devops-engineer', 'platform-engineer'],
        'docker': ['devops-engineer', 'deployment-engineer', 'platform-engineer'],
        'ci/cd': ['devops-engineer', 'deployment-engineer'],
        'terraform': ['terraform-engineer', 'cloud-architect'],
        'cloud': ['cloud-architect', 'devops-engineer', 'platform-engineer'],

        # Testing & Quality
        'test': ['test-automator', 'qa-expert'],
        'testing': ['test-automator', 'qa-expert'],
        'qa': ['qa-expert', 'test-automator'],
        'security': ['security-engineer', 'security-auditor', 'penetration-tester'],
        'performance': ['performance-engineer', 'debugger'],

        # Architecture
        'architecture': ['cloud-architect', 'microservices-architect', 'architect-reviewer'],
        'microservices': ['microservices-architect', 'backend-developer', 'golang-pro'],
        'design': ['architect-reviewer', 'api-designer', 'ui-designer'],

        # Data & AI
        'data': ['data-engineer', 'data-analyst', 'data-scientist'],
        'machine learning': ['ml-engineer', 'data-scientist', 'mlops-engineer'],
        'ai': ['ai-engineer', 'ml-engineer', 'llm-architect'],

        # Mobile
        'mobile': ['mobile-app-developer', 'mobile-developer', 'flutter-expert'],
        'ios': ['swift-expert', 'mobile-app-developer'],
        'android': ['kotlin-specialist', 'mobile-app-developer'],
        'flutter': ['flutter-expert', 'mobile-developer'],
    }

    def __init__(self, workspace_path: Path = Path(".ai-workspace")):
        """Initialize recommender."""
        self.workspace_path = workspace_path
        self.agents_dir = workspace_path / "agents"
        self.agents_cache = {}

    def recommend(self, task_description: str, top_n: int = 10, category: str = None) -> List[AgentRecommendation]:
        """
        Recommend agents for a given task.

        Args:
            task_description: Description of the task
            top_n: Number of recommendations to return
            category: Filter by agent category (optional)

        Returns:
            List of agent recommendations sorted by confidence
        """
        recommendations = []

        # Scan all agent files
        for agent_file in self.agents_dir.rglob("*.md"):
            # Skip README files
            if agent_file.name in ["README.md", "AGENT_SELECTION_GUIDE.md", "temp_AGENT_SELECTION_GUIDE.md"]:
                continue

            # Get agent name and category
            agent_name = agent_file.stem
            agent_category = self._get_category(agent_file)

            # Skip if category filter doesn't match
            if category and category.lower() not in agent_category.lower():
                continue

            # Calculate confidence score
            confidence, matching_keywords = self._calculate_confidence(
                task_description, agent_name, agent_file
            )

            if confidence > 0:
                recommendations.append(AgentRecommendation(
                    name=agent_name,
                    category=agent_category,
                    file_path=str(agent_file.relative_to(self.workspace_path)),
                    confidence=confidence,
                    matching_keywords=matching_keywords
                ))

        # Sort by confidence (descending) and return top N
        recommendations.sort(key=lambda x: x.confidence, reverse=True)
        return recommendations[:top_n]

    def _get_category(self, agent_file: Path) -> str:
        """Get category from file path."""
        try:
            # Agent files are in category directories
            # e.g., .ai-workspace/agents/01-development/backend/api-designer.md
            parts = agent_file.relative_to(self.agents_dir).parts
            if len(parts) >= 2:
                category = parts[0]
                subcategory = parts[1] if len(parts) > 2 else ""

                # Clean up category names
                category_map = {
                    "00-orchestration": "Orchestration",
                    "01-development": "Development",
                    "02-languages": "Languages",
                    "03-frameworks": "Frameworks",
                    "04-infrastructure": "Infrastructure",
                    "05-quality": "Quality",
                    "06-data-ai": "Data & AI",
                    "07-specialized": "Specialized",
                    "08-support": "Support",
                    "09-utilities": "Utilities",
                }

                return f"{category_map.get(category, category)} / {subcategory}"
            return "General"
        except:
            return "Unknown"

    def _calculate_confidence(
        self,
        task_description: str,
        agent_name: str,
        agent_file: Path
    ) -> Tuple[float, List[str]]:
        """Calculate confidence score for agent match."""
        confidence = 0.0
        matching_keywords = []

        # Normalize task description
        task_lower = task_description.lower()
        task_words = set(re.findall(r'\w+', task_lower))

        # 1. Direct keyword matching (40%)
        for keyword, agents in self.KEYWORD_MAPPINGS.items():
            if keyword in task_lower:
                if agent_name in agents:
                    confidence += 0.4
                    matching_keywords.append(keyword)
                    break

        # 2. Agent name similarity (30%)
        agent_words = set(re.findall(r'\w+', agent_name.lower().replace('-', ' ')))
        name_overlap = len(task_words & agent_words)
        if name_overlap > 0:
            confidence += 0.3 * (name_overlap / len(agent_words))
            matching_keywords.extend(task_words & agent_words)

        # 3. File content matching (30%)
        try:
            content = agent_file.read_text(encoding='utf-8').lower()

            # Extract description (first paragraph or title)
            lines = content.split('\n')
            description = ""
            for line in lines[:10]:  # Check first 10 lines
                if line.strip() and not line.startswith('#'):
                    description = line.strip()
                    break

            content_words = set(re.findall(r'\w+', content))
            content_overlap = len(task_words & content_words)

            if content_overlap > 0:
                confidence += 0.3 * min(content_overlap / 10, 1.0)  # Cap at 10 matches

        except Exception:
            pass

        return confidence, list(set(matching_keywords))


def print_recommendations(
    recommendations: List[AgentRecommendation],
    task_description: str
) -> None:
    """Print formatted recommendations."""
    print(f"\n🤖 Agent Recommendations for: '{task_description}'")
    print("=" * 80)

    if not recommendations:
        print("\n❌ No matching agents found.")
        print("\n💡 Try:")
        print("   • Using different keywords")
        print("   • Being more specific about the task")
        print("   • Browsing agents: ls .ai-workspace/agents/")
        return

    print(f"\n✅ Found {len(recommendations)} matching agents\n")

    for i, rec in enumerate(recommendations, 1):
        confidence_pct = rec.confidence * 100
        confidence_bar = "█" * int(confidence_pct / 10) + "▒" * (10 - int(confidence_pct / 10))

        color = "🟢" if confidence_pct >= 70 else "🟡" if confidence_pct >= 40 else "⚪"

        print(f"{color} {i}. @{rec.name}")
        print(f"   Category: {rec.category}")
        print(f"   Confidence: [{confidence_bar}] {confidence_pct:.1f}%")

        if rec.matching_keywords:
            keywords_str = ", ".join(rec.matching_keywords[:5])
            print(f"   Matches: {keywords_str}")

        print(f"   File: {rec.file_path}")
        print()

    print("=" * 80)
    print("\n📝 How to use:")
    print(f"   • Reference in CLAUDE.md: @{recommendations[0].name}")
    print(f"   • View details: cat .ai-workspace/{recommendations[0].file_path}")
    print(f"   • Multiple agents: @{recommendations[0].name} @{recommendations[1].name}")
    print("\n💡 Tip: Use multiple specialized agents for complex tasks")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Recommend agents for specific tasks",
        epilog="Example: python recommend_agents.py 'build REST API'"
    )
    parser.add_argument(
        "task",
        help="Task description (what you want to accomplish)"
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of recommendations (default: 10)"
    )
    parser.add_argument(
        "--category",
        help="Filter by category (e.g., 'development', 'infrastructure')"
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path(".ai-workspace"),
        help="Path to workspace (default: .ai-workspace)"
    )

    args = parser.parse_args()

    try:
        recommender = AgentRecommender(args.workspace)
        recommendations = recommender.recommend(
            args.task,
            top_n=args.top,
            category=args.category
        )

        print_recommendations(recommendations, args.task)
        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️  Cancelled")
        return 1

    except Exception as e:
        print(f"\n❌ Recommendation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
