#!/usr/bin/env python3
"""
Tech Stack Auto-Detection System
Detects languages, frameworks, databases, and tools to configure AI Workspace.
"""
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Set
import sys


class StackDetector:
    """Automatically detect project technology stack."""

    def __init__(self, project_root: Path):
        """Initialize detector with project root path."""
        self.root = project_root
        self.stack: Dict[str, List[str]] = {
            "languages": [],
            "frameworks": [],
            "databases": [],
            "tools": [],
            "cloud": [],
            "specialized": [],
        }
        self.confidence: Dict[str, int] = {}

    def detect(self) -> Dict[str, Any]:
        """Run all detection methods and return comprehensive stack info."""
        self._detect_languages()
        self._detect_frameworks()
        self._detect_databases()
        self._detect_tools()
        self._detect_cloud()
        self._detect_specialized()

        return {
            "stack": self.stack,
            "confidence": self.confidence,
            "primary_language": self._get_primary_language(),
            "project_type": self._infer_project_type(),
        }

    def _detect_languages(self) -> None:
        """Detect programming languages from project files."""
        indicators = {
            "Python": {
                "files": ["pyproject.toml", "requirements.txt", "setup.py", "Pipfile"],
                "extensions": [".py"],
                "confidence_boost": 10,
            },
            "TypeScript": {
                "files": ["tsconfig.json"],
                "extensions": [".ts", ".tsx"],
                "confidence_boost": 10,
            },
            "JavaScript": {
                "files": ["package.json"],
                "extensions": [".js", ".jsx"],
                "confidence_boost": 5,
            },
            "Rust": {
                "files": ["Cargo.toml", "Cargo.lock"],
                "extensions": [".rs"],
                "confidence_boost": 10,
            },
            "Go": {
                "files": ["go.mod", "go.sum"],
                "extensions": [".go"],
                "confidence_boost": 10,
            },
            "Java": {
                "files": ["pom.xml", "build.gradle", "build.gradle.kts"],
                "extensions": [".java"],
                "confidence_boost": 10,
            },
            "C++": {
                "files": ["CMakeLists.txt", "Makefile"],
                "extensions": [".cpp", ".cc", ".cxx", ".hpp"],
                "confidence_boost": 8,
            },
            "C#": {
                "files": [".csproj", ".sln"],
                "extensions": [".cs"],
                "confidence_boost": 10,
            },
            "PHP": {
                "files": ["composer.json"],
                "extensions": [".php"],
                "confidence_boost": 10,
            },
            "Ruby": {
                "files": ["Gemfile", "Gemfile.lock"],
                "extensions": [".rb"],
                "confidence_boost": 10,
            },
            "Swift": {
                "files": ["Package.swift"],
                "extensions": [".swift"],
                "confidence_boost": 10,
            },
            "Kotlin": {
                "files": ["build.gradle.kts"],
                "extensions": [".kt", ".kts"],
                "confidence_boost": 10,
            },
        }

        for lang, indicators_data in indicators.items():
            confidence = 0

            # Check for indicator files
            for file in indicators_data["files"]:
                if (self.root / file).exists():
                    confidence += indicators_data["confidence_boost"]

            # Count files with language extensions
            if "extensions" in indicators_data:
                for ext in indicators_data["extensions"]:
                    count = len(list(self.root.rglob(f"*{ext}")))
                    if count > 0:
                        confidence += min(count, 20)  # Cap at 20

            if confidence > 0:
                self.stack["languages"].append(lang)
                self.confidence[lang] = confidence

    def _detect_frameworks(self) -> None:
        """Detect frameworks from dependency files and project structure."""
        # Python frameworks
        if (self.root / "pyproject.toml").exists():
            content = (self.root / "pyproject.toml").read_text().lower()
            python_frameworks = {
                "FastAPI": ["fastapi"],
                "Django": ["django"],
                "Flask": ["flask"],
                "SQLAlchemy": ["sqlalchemy"],
                "Pydantic": ["pydantic"],
                "pytest": ["pytest"],
            }
            for fw, patterns in python_frameworks.items():
                if any(p in content for p in patterns):
                    self.stack["frameworks"].append(fw)

        if (self.root / "requirements.txt").exists():
            content = (self.root / "requirements.txt").read_text().lower()
            if "fastapi" in content:
                self.stack["frameworks"].append("FastAPI")
            if "django" in content:
                self.stack["frameworks"].append("Django")
            if "flask" in content:
                self.stack["frameworks"].append("Flask")

        # TypeScript/JavaScript frameworks
        if (self.root / "package.json").exists():
            try:
                pkg = json.loads((self.root / "package.json").read_text())
                deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}

                js_frameworks = {
                    "Next.js": ["next"],
                    "React": ["react"],
                    "Vue": ["vue"],
                    "Angular": ["@angular/core"],
                    "NestJS": ["@nestjs/core"],
                    "Express": ["express"],
                    "Svelte": ["svelte"],
                }

                for fw, pkg_names in js_frameworks.items():
                    if any(p in deps for p in pkg_names):
                        self.stack["frameworks"].append(fw)
            except json.JSONDecodeError:
                pass

        # Java frameworks
        if (self.root / "pom.xml").exists():
            content = (self.root / "pom.xml").read_text().lower()
            if "spring-boot" in content:
                self.stack["frameworks"].append("Spring Boot")

        # Ruby frameworks
        if (self.root / "Gemfile").exists():
            content = (self.root / "Gemfile").read_text().lower()
            if "rails" in content:
                self.stack["frameworks"].append("Rails")

        # PHP frameworks
        if (self.root / "composer.json").exists():
            try:
                composer = json.loads((self.root / "composer.json").read_text())
                deps = composer.get("require", {})
                if "laravel/framework" in deps:
                    self.stack["frameworks"].append("Laravel")
                if "symfony/symfony" in deps:
                    self.stack["frameworks"].append("Symfony")
            except json.JSONDecodeError:
                pass

        # .NET frameworks
        for csproj in self.root.rglob("*.csproj"):
            content = csproj.read_text().lower()
            if "microsoft.aspnetcore" in content:
                self.stack["frameworks"].append(".NET Core")
            if "targetframeworkversion>v4." in content:
                self.stack["frameworks"].append(".NET Framework 4.x")

        # Mobile frameworks
        if (self.root / "pubspec.yaml").exists():
            self.stack["frameworks"].append("Flutter")
        if (self.root / "android").exists() and (self.root / "ios").exists():
            if (self.root / "package.json").exists():
                self.stack["frameworks"].append("React Native")

    def _detect_databases(self) -> None:
        """Detect databases from docker-compose, env files, and configs."""
        sources = [
            self.root / "docker-compose.yml",
            self.root / "docker-compose.yaml",
            self.root / ".env",
            self.root / ".env.example",
        ]

        content = ""
        for source in sources:
            if source.exists():
                content += source.read_text().lower()

        databases = {
            "PostgreSQL": ["postgres:", "postgresql:", "psql", "pg_"],
            "MySQL": ["mysql:", "mariadb:"],
            "MongoDB": ["mongo:", "mongodb:"],
            "Redis": ["redis:"],
            "SQLite": ["sqlite"],
            "Elasticsearch": ["elasticsearch:"],
            "Cassandra": ["cassandra:"],
            "DynamoDB": ["dynamodb"],
        }

        for db, patterns in databases.items():
            if any(p in content for p in patterns):
                self.stack["databases"].append(db)

    def _detect_tools(self) -> None:
        """Detect development tools and infrastructure."""
        tools = {
            "Docker": ["Dockerfile", "docker-compose.yml", "docker-compose.yaml"],
            "Kubernetes": ["k8s/", "kubernetes/", "deployment.yaml"],
            "Terraform": [".tf"],
            "Ansible": ["ansible/", "playbook.yml"],
            "GitHub Actions": [".github/workflows/"],
            "GitLab CI": [".gitlab-ci.yml"],
            "Jenkins": ["Jenkinsfile"],
            "CircleCI": [".circleci/"],
            "Kafka": ["kafka"],
            "RabbitMQ": ["rabbitmq"],
            "Temporal": ["temporal"],
        }

        for tool, indicators in tools.items():
            for indicator in indicators:
                paths = list(self.root.rglob(indicator))
                if paths or (self.root / indicator).exists():
                    self.stack["tools"].append(tool)
                    break

    def _detect_cloud(self) -> None:
        """Detect cloud providers and services."""
        # Check for cloud-specific config files
        cloud_indicators = {
            "AWS": [".aws/", "cloudformation/", "serverless.yml"],
            "Azure": ["azure-pipelines.yml", ".azure/"],
            "GCP": ["gcp/", "app.yaml", "cloudbuild.yaml"],
            "Vercel": ["vercel.json"],
            "Netlify": ["netlify.toml"],
            "Heroku": ["Procfile", "app.json"],
        }

        for cloud, indicators in cloud_indicators.items():
            if any((self.root / ind).exists() for ind in indicators):
                self.stack["cloud"].append(cloud)

    def _detect_specialized(self) -> None:
        """Detect specialized domains (blockchain, IoT, ML, etc.)."""
        # Machine Learning
        if (self.root / "requirements.txt").exists():
            content = (self.root / "requirements.txt").read_text().lower()
            ml_libs = ["tensorflow", "pytorch", "sklearn", "keras", "prophet"]
            if any(lib in content for lib in ml_libs):
                self.stack["specialized"].append("Machine Learning")

        # Blockchain
        blockchain_indicators = ["truffle-config.js", "hardhat.config.js", "foundry.toml"]
        if any((self.root / ind).exists() for ind in blockchain_indicators):
            self.stack["specialized"].append("Blockchain")

        # IoT
        iot_indicators = ["platformio.ini", "arduino/"]
        if any((self.root / ind).exists() for ind in iot_indicators):
            self.stack["specialized"].append("IoT")

        # Game Development
        game_indicators = ["unity/", "unreal/", "godot/"]
        if any((self.root / ind).exists() for ind in game_indicators):
            self.stack["specialized"].append("Game Development")

    def _get_primary_language(self) -> str:
        """Determine primary language by confidence score."""
        if not self.confidence:
            return "Unknown"

        return max(self.confidence, key=self.confidence.get)

    def _infer_project_type(self) -> str:
        """Infer project type from detected stack."""
        # Microservices
        if "Docker" in self.stack["tools"] and "Kubernetes" in self.stack["tools"]:
            if len(self.stack["databases"]) > 1:
                return "microservices"

        # Monolith
        if any(fw in self.stack["frameworks"] for fw in ["Django", "Rails", "Laravel"]):
            return "monolith"

        # SPA
        if any(fw in self.stack["frameworks"] for fw in ["React", "Vue", "Angular"]):
            return "spa"

        # API
        if any(fw in self.stack["frameworks"] for fw in ["FastAPI", "Express", "NestJS"]):
            return "api"

        # Mobile
        if any(fw in self.stack["frameworks"] for fw in ["Flutter", "React Native"]):
            return "mobile"

        # Data Science
        if "Machine Learning" in self.stack["specialized"]:
            return "data-science"

        return "general"


def save_stack_config(detection_result: Dict[str, Any], output_path: Path) -> None:
    """Save detected stack configuration to YAML file."""
    import yaml

    config = {
        "# AI Workspace Configuration": None,
        "# Auto-generated by stack detection v1.1.0": None,
        "project": {
            "detected_stack": detection_result["stack"],
            "primary_language": detection_result["primary_language"],
            "project_type": detection_result["project_type"],
            "detection_date": "2025-10-07",
            "confidence_scores": detection_result["confidence"],
        },
        "enabled_rules": _generate_rule_list(detection_result),
        "recommended_agents": _recommend_agents(detection_result),
    }

    # Remove None keys (comments)
    clean_config = {k: v for k, v in config.items() if v is not None}

    output_path.write_text(yaml.dump(clean_config, sort_keys=False, allow_unicode=True))
    print(f"✅ Stack configuration saved to {output_path}")


def _generate_rule_list(detection: Dict[str, Any]) -> List[str]:
    """Generate list of rules to enable based on detected stack."""
    rules = ["00-core/operating-charter.md", "00-core/reuse-first.md", "00-core/phase-gates.md"]

    # Language-specific rules
    lang_mapping = {
        "Python": "01-language/python-rules.md",
        "TypeScript": "01-language/typescript-rules.md",
        "JavaScript": "01-language/javascript-rules.md",
        "Rust": "01-language/rust-rules.md",
        "Go": "01-language/go-rules.md",
        "Java": "01-language/java-rules.md",
    }

    for lang in detection["stack"]["languages"]:
        if lang in lang_mapping:
            rules.append(lang_mapping[lang])

    # Framework-specific rules
    fw_mapping = {
        "FastAPI": "02-framework/fastapi-rules.md",
        "Django": "02-framework/django-rules.md",
        "Next.js": "02-framework/nextjs-rules.md",
        "React": "02-framework/react-rules.md",
    }

    for fw in detection["stack"]["frameworks"]:
        if fw in fw_mapping:
            rules.append(fw_mapping[fw])

    return rules


def _recommend_agents(detection: Dict[str, Any]) -> Dict[str, List[str]]:
    """Recommend agents based on detected stack."""
    recommendations = {"primary": [], "secondary": [], "optional": []}

    stack = detection["stack"]

    # Language agents
    if "Python" in stack["languages"]:
        recommendations["primary"].append("@python-pro")
    if "TypeScript" in stack["languages"]:
        recommendations["primary"].append("@typescript-pro")

    # Framework agents
    if "FastAPI" in stack["frameworks"]:
        recommendations["primary"].extend(["@api-designer", "@microservices-architect"])
    if "React" in stack["frameworks"]:
        recommendations["primary"].append("@react-specialist")

    # Database agents
    if "PostgreSQL" in stack["databases"]:
        recommendations["primary"].append("@postgres-pro")
        recommendations["secondary"].append("@database-optimizer")

    # Infrastructure agents
    if "Docker" in stack["tools"]:
        recommendations["primary"].append("@devops-engineer")
    if "Kubernetes" in stack["tools"]:
        recommendations["secondary"].append("@kubernetes-specialist")

    # Quality agents (always recommended)
    recommendations["secondary"].extend(["@code-reviewer", "@test-automator", "@qa-expert"])

    return recommendations


def main() -> None:
    """Main entry point."""
    # Set UTF-8 encoding for Windows
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    project_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()

    print("AI Workspace - Technology Stack Detection v1.1.0")
    print("=" * 60)
    print(f"Analyzing: {project_root.absolute()}")
    print(f"Multi-language support: Python, TypeScript, JavaScript, Rust, Go, Java\n")

    detector = StackDetector(project_root)
    result = detector.detect()

    print("Detection Results:")
    print("-" * 60)
    print(f"Project Type: {result['project_type']}")
    print(f"Primary Language: {result['primary_language']}\n")

    for category, items in result["stack"].items():
        if items:
            print(f"  {category.title()}: {', '.join(items)}")

    # Save configuration
    config_path = project_root / ".ai-workspace-config.yml"
    save_stack_config(result, config_path)

    print(f"\nNext steps:")
    print(f"  1. Review: {config_path}")
    print(f"  2. Run: python .ai-workspace/scripts/generate_claude_md.py")
    print(f"  3. Run: python .ai-workspace/scripts/setup.py")
    print(f"\n✨ Multi-language reuse analysis ready for 6 languages!")


if __name__ == "__main__":
    main()
