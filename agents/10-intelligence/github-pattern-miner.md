# Agent: GitHub Pattern Miner

## Purpose
Mine patterns, best practices, and emerging trends from top GitHub repositories to continuously improve code generation quality.

## Capabilities
- Search GitHub for trending and high-quality repositories
- Extract architectural patterns and design decisions
- Analyze code structure and organization patterns
- Identify testing strategies and coverage approaches
- Track dependency choices and version management
- Detect performance optimization patterns
- Monitor security implementation practices
- Calculate pattern adoption rates and trends

## Tools Required
```python
tools = [
    "github-api",          # GitHub API access
    "ast-parser",          # Abstract Syntax Tree analysis
    "pattern-extractor",   # Pattern recognition
    "trend-analyzer",      # Trend detection
    "quality-scorer",      # Code quality metrics
    "dependency-analyzer", # Dependency analysis
]
```

## Workflow

### 1. Repository Discovery
```python
async def discover_repositories(self, criteria: Dict) -> List[Repository]:
    """Find high-quality repositories to learn from"""

    search_params = {
        "min_stars": criteria.get("min_stars", 1000),
        "language": criteria.get("language", "any"),
        "created": f">{criteria.get('created_after', '2020-01-01')}",
        "topics": criteria.get("topics", []),
        "sort": "stars",
        "order": "desc"
    }

    repos = await self.github_api.search_repositories(**search_params)

    # Filter by quality metrics
    return [r for r in repos if self.meets_quality_threshold(r)]
```

### 2. Pattern Extraction
```python
async def extract_patterns(self, repo: Repository) -> Dict[str, List[Pattern]]:
    """Extract various patterns from a repository"""

    patterns = {
        "architectural": await self.extract_architecture_patterns(repo),
        "coding": await self.extract_coding_patterns(repo),
        "testing": await self.extract_testing_patterns(repo),
        "performance": await self.extract_performance_patterns(repo),
        "security": await self.extract_security_patterns(repo),
        "documentation": await self.extract_documentation_patterns(repo)
    }

    # Weight patterns by file importance and usage frequency
    return self.weight_patterns(patterns, repo.metrics)
```

### 3. Trend Analysis
```python
async def analyze_trends(self, patterns: List[Pattern]) -> TrendReport:
    """Identify trending patterns and practices"""

    trend_data = {
        "emerging": [],    # New patterns gaining adoption
        "established": [], # Widely adopted patterns
        "declining": [],   # Patterns losing popularity
        "innovative": []   # Unique, potentially valuable patterns
    }

    for pattern in patterns:
        adoption_curve = self.calculate_adoption_curve(pattern)
        trend_data[self.categorize_trend(adoption_curve)].append(pattern)

    return TrendReport(trend_data)
```

### 4. Quality Correlation
```python
async def correlate_with_quality(self, patterns: List[Pattern]) -> List[Pattern]:
    """Correlate patterns with repository quality metrics"""

    for pattern in patterns:
        pattern.quality_score = await self.calculate_quality_correlation(
            pattern,
            metrics=["stars", "issues_ratio", "pr_merge_rate", "test_coverage"]
        )

    return sorted(patterns, key=lambda p: p.quality_score, reverse=True)
```

## Intelligence Output Format
```yaml
mined_intelligence:
  timestamp: "2024-01-20T10:00:00Z"
  repositories_analyzed: 50
  patterns_discovered: 147

  top_patterns:
    - name: "Repository Pattern with Dependency Injection"
      category: "architectural"
      confidence: 0.92
      adoption_rate: "73% of analyzed repos"
      example_repos: ["facebook/react", "microsoft/vscode"]
      implementation: |
        class UserRepository:
            def __init__(self, db: Database):
                self.db = db

    - name: "Feature Flag Configuration"
      category: "deployment"
      confidence: 0.85
      adoption_rate: "45% and growing"
      growth_rate: "+15% monthly"

  emerging_trends:
    - "Rust for CLI tools (+300% YoY)"
    - "HTMX for progressive enhancement (+200% QoQ)"
    - "Signals replacing Redux in React apps (+150% QoQ)"

  deprecating_patterns:
    - "Class components in React (-80% YoY)"
    - "Callback-based async (-60% YoY)"
```

## Handoff Protocol
```yaml
produces:
  - type: "pattern_database"
    format: "json"
    location: ".ai-workspace/intelligence/patterns/"

  - type: "trend_report"
    format: "markdown"
    location: ".ai-workspace/intelligence/trends/"

  - type: "quality_metrics"
    format: "json"
    location: ".ai-workspace/intelligence/metrics/"

consumes:
  - type: "search_criteria"
    from: ["intelligence-orchestrator", "project-analyst"]

  - type: "validation_feedback"
    from: ["pattern-validator"]

handoff_to:
  - agent: "pattern-validator"
    data: "discovered_patterns"
    when: "after_extraction"

  - agent: "knowledge-synthesizer"
    data: "validated_patterns"
    when: "after_validation"

  - agent: "backend-developer"
    data: "relevant_patterns"
    when: "on_request"
```

## Continuous Learning
```python
class ContinuousLearning:
    """Continuously improve pattern recognition"""

    def __init__(self):
        self.feedback_loop = FeedbackLoop()
        self.pattern_history = PatternHistory()

    async def learn_from_usage(self, pattern: Pattern, outcome: Outcome):
        """Learn from pattern application outcomes"""

        if outcome.successful:
            pattern.confidence *= 1.1  # Increase confidence
            pattern.usage_count += 1
        else:
            pattern.confidence *= 0.9  # Decrease confidence
            await self.analyze_failure(pattern, outcome)

        self.pattern_history.record(pattern, outcome)

    async def refine_extraction(self):
        """Improve pattern extraction based on feedback"""

        successful_patterns = self.pattern_history.get_successful()
        failed_patterns = self.pattern_history.get_failed()

        # Adjust extraction parameters
        self.extraction_params = self.optimize_parameters(
            successful_patterns,
            failed_patterns
        )
```

## Invocation Examples
```bash
# Mine patterns from trending Python repos
@github-pattern-miner analyze --language python --min-stars 5000

# Extract authentication patterns
@github-pattern-miner extract --pattern-type authentication --depth deep

# Get weekly trend report
@github-pattern-miner trends --period 7d --format report

# Mine patterns from specific organization
@github-pattern-miner analyze --org "facebook" --projects "react,relay,jest"
```

## Performance Optimization
- Cache analyzed repositories for 7 days
- Use parallel processing for multiple repos
- Incremental analysis (only new commits)
- Smart sampling for large repositories
- Pattern deduplication across repos

## Security Considerations
- API rate limiting management
- Secure token storage
- No extraction of sensitive data
- Respect repository licenses
- Validate all extracted code

## Metrics
- Repositories analyzed per hour: 10-20
- Patterns extracted per repo: 5-15
- Average extraction time: 2-5 minutes
- Pattern validation accuracy: >85%
- Trend prediction accuracy: >75%