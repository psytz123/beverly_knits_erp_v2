"""
Agent Factory - Dynamic Agent Instantiation.

This module provides functionality to parse agent markdown files and create
executable agent instances with proper metadata extraction, validation, and
error handling.

Classes:
    AgentParsingError: Raised when agent markdown parsing fails
    AgentNotFoundError: Raised when agent file doesn't exist
    ExecutableAgent: Executable agent instance created from markdown
    AgentFactory: Factory for creating executable agents from markdown files

Example:
    >>> factory = AgentFactory()
    >>> agent = factory.create_agent("python-pro")
    >>> print(agent.metadata.name)
    python-pro
    >>> print(agent.metadata.tools)
    ['Read', 'Write', 'MultiEdit', 'Bash', ...]
"""

import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List

from .interfaces import AgentMetadata, IAgentFactory

logger = logging.getLogger(__name__)


class AgentParsingError(Exception):
    """
    Raised when agent markdown parsing fails.

    This exception is raised when:
    - YAML frontmatter is missing or malformed
    - Required fields (name, description, tools) are missing
    - YAML parsing fails
    - Frontmatter format is invalid
    """
    pass


class AgentNotFoundError(Exception):
    """
    Raised when agent file doesn't exist.

    This exception is raised when:
    - Agent name doesn't match any markdown file
    - Agent file path is invalid
    - Agent directory doesn't exist
    """
    pass


class ExecutableAgent:
    """
    Executable agent instance created from markdown.

    This class represents a fully initialized agent ready for execution,
    with parsed metadata, content, and tool permissions.

    Attributes:
        metadata: Parsed agent metadata from frontmatter
        content: Agent instructions/prompt (markdown body)
        tools: Dictionary of tool names to permission flags

    Example:
        >>> metadata = AgentMetadata(
        ...     name="python-pro",
        ...     description="Expert Python developer",
        ...     tools=["Read", "Write", "Bash"]
        ... )
        >>> agent = ExecutableAgent(metadata, "Agent instructions here...")
        >>> print(agent.get_prompt()[:50])
        Agent instructions here...
    """

    def __init__(self, metadata: AgentMetadata, content: str):
        """
        Initialize executable agent.

        Args:
            metadata: Parsed agent metadata from frontmatter
            content: Agent instructions/prompt (markdown body)

        Raises:
            ValueError: If metadata or content is invalid
        """
        if not metadata:
            raise ValueError("Agent metadata cannot be None")
        if not content:
            logger.warning(f"Agent {metadata.name} has empty content")

        self.metadata = metadata
        self.content = content
        self.tools = self._initialize_tools(metadata.tools)

    def _initialize_tools(self, tool_names: List[str]) -> Dict[str, bool]:
        """
        Initialize tool access based on permissions.

        Args:
            tool_names: List of tool names agent has permission to use

        Returns:
            Dictionary mapping tool names to permission flags (True = allowed)
        """
        # For now, just return tool names with permission flag
        # Future: Could integrate with actual tool registry/permissions system
        return {tool.strip(): True for tool in tool_names}

    def get_prompt(self) -> str:
        """
        Get agent's execution prompt.

        Returns:
            Full agent instructions including metadata and content
        """
        return self.content

    def has_tool(self, tool_name: str) -> bool:
        """
        Check if agent has permission to use a tool.

        Args:
            tool_name: Name of tool to check

        Returns:
            True if agent has permission, False otherwise
        """
        return self.tools.get(tool_name, False)

    def __repr__(self) -> str:
        """String representation of executable agent."""
        return f"ExecutableAgent(name={self.metadata.name}, tools={len(self.metadata.tools)})"

    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"{self.metadata.name} ({self.metadata.category})"


class AgentFactory(IAgentFactory):
    """
    Create executable agents from markdown specifications.

    This factory class handles:
    - Loading agent markdown files from the workspace
    - Parsing YAML frontmatter and extracting metadata
    - Validating agent specifications
    - Creating executable agent instances
    - Caching agents for performance

    Attributes:
        workspace: Path to .ai-workspace directory
        agents_dir: Path to agents directory
        cache: Dictionary of cached agent instances

    Example:
        >>> factory = AgentFactory()
        >>> agent = factory.create_agent("python-pro")
        >>> print(agent.metadata.description)
        Expert Python developer specializing in...
    """

    def __init__(self, workspace_path: Path = Path(".ai-workspace")):
        """
        Initialize agent factory.

        Args:
            workspace_path: Path to .ai-workspace directory (default: ".ai-workspace")

        Raises:
            ValueError: If workspace_path doesn't exist
        """
        self.workspace = workspace_path
        self.agents_dir = workspace_path / "agents"
        self.cache: Dict[str, ExecutableAgent] = {}

        if not self.workspace.exists():
            logger.warning(f"Workspace path does not exist: {workspace_path}")

        if not self.agents_dir.exists():
            logger.warning(f"Agents directory does not exist: {self.agents_dir}")

        logger.info(f"AgentFactory initialized with workspace: {workspace_path}")

    def create_agent(self, agent_name: str) -> ExecutableAgent:
        """
        Create executable agent from markdown file.

        This method:
        1. Checks cache for existing instance
        2. Finds agent markdown file by name
        3. Parses metadata from YAML frontmatter
        4. Extracts markdown content
        5. Creates and caches ExecutableAgent instance

        Args:
            agent_name: Name of agent (e.g., "python-pro")

        Returns:
            ExecutableAgent instance ready for execution

        Raises:
            AgentNotFoundError: If agent file doesn't exist
            AgentParsingError: If agent markdown is malformed

        Example:
            >>> factory = AgentFactory()
            >>> agent = factory.create_agent("python-pro")
            >>> print(agent.metadata.description)
            Expert Python developer specializing in...
        """
        # Normalize agent name (remove .md extension if present)
        agent_name = agent_name.replace(".md", "").strip()

        # Check cache first
        if agent_name in self.cache:
            logger.debug(f"Returning cached agent: {agent_name}")
            return self.cache[agent_name]

        # Find agent file
        agent_path = self._find_agent_file(agent_name)
        if not agent_path:
            raise AgentNotFoundError(
                f"Agent '{agent_name}' not found. "
                f"Searched in: {self.agents_dir}\n"
                f"Ensure the agent markdown file exists in the agents directory."
            )

        # Parse metadata and create agent
        try:
            metadata = self.parse_agent_metadata(agent_path)
            content = self._extract_content(agent_path)

            # Create executable agent
            agent = ExecutableAgent(metadata, content)

            # Cache for future use
            self.cache[agent_name] = agent

            logger.info(
                f"Created agent: {agent_name} "
                f"(category: {metadata.category}, tools: {len(metadata.tools)})"
            )
            return agent

        except AgentParsingError:
            # Re-raise parsing errors with context
            raise
        except Exception as e:
            logger.error(f"Failed to create agent '{agent_name}': {e}")
            raise AgentParsingError(
                f"Unexpected error creating agent '{agent_name}': {e}"
            )

    def parse_agent_metadata(self, md_path: Path) -> AgentMetadata:
        """
        Extract metadata from agent markdown file.

        Parses YAML frontmatter and validates required fields:
        - name: Agent identifier
        - description: Agent capabilities description
        - tools: Comma-separated string or list of tool names

        Args:
            md_path: Path to agent markdown file

        Returns:
            AgentMetadata with parsed frontmatter

        Raises:
            AgentParsingError: If markdown is malformed or missing required fields

        Example:
            Given markdown file:
            ```
            ---
            name: python-pro
            description: Expert Python developer
            tools: Read, Write, Bash
            ---
            Agent content here...
            ```

            Returns:
            ```
            AgentMetadata(
                name="python-pro",
                description="Expert Python developer",
                tools=["Read", "Write", "Bash"]
            )
            ```
        """
        try:
            # Read file content
            content = md_path.read_text(encoding="utf-8")

            # Validate frontmatter presence
            if not content.startswith("---"):
                raise AgentParsingError(
                    f"Missing YAML frontmatter in {md_path.name}. "
                    f"Agent files must start with '---'"
                )

            # Split frontmatter and content
            parts = content.split("---", 2)
            if len(parts) < 3:
                raise AgentParsingError(
                    f"Malformed YAML frontmatter in {md_path.name}. "
                    f"Must be enclosed in '---' delimiters"
                )

            frontmatter = parts[1].strip()

            # Parse YAML
            try:
                metadata_dict = yaml.safe_load(frontmatter)
            except yaml.YAMLError as e:
                raise AgentParsingError(
                    f"Invalid YAML in {md_path.name}: {e}"
                )

            # Validate YAML parsed to dictionary
            if not isinstance(metadata_dict, dict):
                raise AgentParsingError(
                    f"YAML frontmatter in {md_path.name} must be a dictionary, "
                    f"got: {type(metadata_dict)}"
                )

            # Validate required fields
            required_fields = ["name", "description", "tools"]
            missing = [f for f in required_fields if f not in metadata_dict]

            if missing:
                raise AgentParsingError(
                    f"Agent {md_path.stem} has invalid format. "
                    f"Ensure YAML frontmatter includes: {', '.join(required_fields)}. "
                    f"Missing: {', '.join(missing)}"
                )

            # Parse tools (can be comma-separated string or list)
            tools = metadata_dict["tools"]
            if isinstance(tools, str):
                # Convert "Read, Write, Bash" → ["Read", "Write", "Bash"]
                tools = [t.strip() for t in tools.split(",") if t.strip()]
            elif isinstance(tools, list):
                # Ensure all items are strings and strip whitespace
                tools = [str(t).strip() for t in tools if str(t).strip()]
            else:
                raise AgentParsingError(
                    f"Tools must be comma-separated string or list in {md_path.name}, "
                    f"got: {type(tools)}"
                )

            # Validate tools list is not empty after parsing
            if not tools:
                raise AgentParsingError(
                    f"Agent {md_path.stem} must have at least one tool"
                )

            # Get category from file path
            category = self._get_category(md_path)

            # Create and validate AgentMetadata
            return AgentMetadata(
                name=metadata_dict["name"],
                description=metadata_dict["description"],
                tools=tools,
                model=metadata_dict.get("model", "claude-sonnet-4"),
                category=category,
                file_path=str(md_path.resolve())
            )

        except AgentParsingError:
            # Re-raise parsing errors
            raise
        except Exception as e:
            raise AgentParsingError(
                f"Unexpected error parsing {md_path.name}: {e}"
            )

    def _find_agent_file(self, agent_name: str) -> Optional[Path]:
        """
        Find agent markdown file by name.

        Searches all category directories recursively for agent markdown files.
        Skips README and guide files.

        Args:
            agent_name: Agent name (e.g., "python-pro")

        Returns:
            Path to agent file or None if not found

        Example:
            >>> factory = AgentFactory()
            >>> path = factory._find_agent_file("python-pro")
            >>> print(path)
            .ai-workspace/agents/02-languages/scripting/python-pro.md
        """
        if not self.agents_dir.exists():
            return None

        # Try direct lookup in each category
        for category_dir in self.agents_dir.iterdir():
            if not category_dir.is_dir():
                continue

            # Check direct file in category root
            agent_file = category_dir / f"{agent_name}.md"
            if agent_file.exists() and agent_file.is_file():
                return agent_file

            # Search recursively (for nested categories)
            for agent_file in category_dir.rglob(f"{agent_name}.md"):
                # Skip README and guide files
                if agent_file.name in [
                    "README.md",
                    "AGENT_SELECTION_GUIDE.md",
                    "temp_AGENT_SELECTION_GUIDE.md"
                ]:
                    continue
                return agent_file

        return None

    def _extract_content(self, md_path: Path) -> str:
        """
        Extract markdown content (everything after frontmatter).

        Args:
            md_path: Path to agent markdown file

        Returns:
            Markdown content without frontmatter

        Example:
            Given file:
            ```
            ---
            name: test
            ---
            Content here
            ```

            Returns: "Content here"
        """
        content = md_path.read_text(encoding="utf-8")
        parts = content.split("---", 2)

        if len(parts) >= 3:
            return parts[2].strip()
        else:
            return ""

    def _get_category(self, md_path: Path) -> str:
        """
        Get agent category from file path.

        Extracts category and subcategory from agent file path using
        the workspace directory structure.

        Args:
            md_path: Path to agent markdown file

        Returns:
            Category string (e.g., "Languages / scripting")

        Example:
            >>> path = Path(".ai-workspace/agents/02-languages/scripting/python-pro.md")
            >>> category = factory._get_category(path)
            >>> print(category)
            Languages / scripting
        """
        try:
            # e.g., .ai-workspace/agents/02-languages/scripting/python-pro.md
            # → "Languages / scripting"
            relative = md_path.relative_to(self.agents_dir)
            parts = relative.parts

            if len(parts) >= 2:
                category = parts[0]  # "02-languages"
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

                category_name = category_map.get(category, category)
                return f"{category_name} / {subcategory}" if subcategory else category_name

            return "General"

        except Exception as e:
            logger.warning(f"Failed to extract category from {md_path}: {e}")
            return "Unknown"

    def list_all_agents(self) -> List[AgentMetadata]:
        """
        List all available agents.

        Recursively scans the agents directory and parses metadata from
        all valid agent markdown files. Skips README and guide files.

        Returns:
            List of AgentMetadata for all discovered agents

        Example:
            >>> factory = AgentFactory()
            >>> agents = factory.list_all_agents()
            >>> print(f"Found {len(agents)} agents")
            Found 156 agents
            >>> print(agents[0].name)
            python-pro
        """
        if not self.agents_dir.exists():
            logger.warning(f"Agents directory does not exist: {self.agents_dir}")
            return []

        agents = []

        for agent_file in self.agents_dir.rglob("*.md"):
            # Skip README and guide files
            if agent_file.name in [
                "README.md",
                "AGENT_SELECTION_GUIDE.md",
                "temp_AGENT_SELECTION_GUIDE.md"
            ]:
                continue

            try:
                metadata = self.parse_agent_metadata(agent_file)
                agents.append(metadata)
            except AgentParsingError as e:
                logger.warning(f"Skipping malformed agent {agent_file.name}: {e}")
                continue
            except Exception as e:
                logger.error(f"Unexpected error parsing {agent_file.name}: {e}")
                continue

        logger.info(f"Discovered {len(agents)} agents across {self.agents_dir}")
        return agents

    def clear_cache(self) -> None:
        """
        Clear the agent cache.

        Useful for forcing re-parsing of agents after modifications.

        Example:
            >>> factory = AgentFactory()
            >>> factory.create_agent("python-pro")  # Cached
            >>> factory.clear_cache()
            >>> factory.create_agent("python-pro")  # Re-parsed
        """
        self.cache.clear()
        logger.info("Agent cache cleared")

    def get_cached_agents(self) -> List[str]:
        """
        Get list of cached agent names.

        Returns:
            List of agent names currently in cache

        Example:
            >>> factory = AgentFactory()
            >>> factory.create_agent("python-pro")
            >>> factory.get_cached_agents()
            ['python-pro']
        """
        return list(self.cache.keys())
