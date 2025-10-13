# Agent: Stack Overflow Harvester

## Purpose
Learn from millions of community-validated solutions, common problems, and best practices by analyzing Stack Overflow questions, answers, and discussions.

## Capabilities
- Search Stack Overflow for relevant questions and solutions
- Extract high-quality, validated answers
- Identify common problems and pitfalls
- Learn debugging strategies and error solutions
- Track trending technologies and questions
- Extract code snippets and patterns
- Analyze voting patterns for quality signals
- Monitor technology migration patterns

## Tools Required
```python
tools = [
    "stackexchange-api",   # Stack Overflow API access
    "answer-analyzer",     # Answer quality analysis
    "code-extractor",      # Extract code from answers
    "sentiment-analyzer",  # Community sentiment
    "trend-detector",      # Trend analysis
    "solution-validator",  # Validate solutions
]
```

## Workflow

### 1. Question Discovery
```python
async def discover_questions(self, topics: List[str]) -> List[Question]:
    """Find relevant high-quality questions"""

    search_criteria = {
        "tags": topics,
        "min_score": 10,
        "has_accepted_answer": True,
        "answer_count": ">=2",
        "sort": "votes",
        "created": "last_6_months"
    }

    questions = await self.stackoverflow_api.search_questions(**search_criteria)

    # Filter for quality and relevance
    return [q for q in questions if self.is_high_quality(q)]
```

### 2. Solution Extraction
```python
async def extract_solutions(self, question: Question) -> Dict[str, Solution]:
    """Extract solutions and patterns from answers"""

    solutions = {
        "accepted": None,
        "highest_voted": None,
        "alternative_approaches": [],
        "common_mistakes": [],
        "performance_optimizations": []
    }

    # Get accepted answer
    if question.accepted_answer_id:
        solutions["accepted"] = await self.analyze_answer(
            question.accepted_answer_id
        )

    # Get all high-quality answers
    answers = await self.get_answers(question.id, min_score=5)

    for answer in answers:
        solution = await self.extract_solution(answer)

        # Categorize solution
        if self.is_optimization(solution):
            solutions["performance_optimizations"].append(solution)
        elif self.is_warning(solution):
            solutions["common_mistakes"].append(solution)
        else:
            solutions["alternative_approaches"].append(solution)

    return solutions
```

### 3. Pattern Recognition
```python
async def recognize_patterns(self, solutions: List[Solution]) -> PatternReport:
    """Identify recurring patterns across solutions"""

    patterns = PatternReport()

    # Group similar solutions
    solution_clusters = self.cluster_solutions(solutions)

    for cluster in solution_clusters:
        pattern = Pattern()
        pattern.name = self.identify_pattern_name(cluster)
        pattern.frequency = len(cluster.solutions)
        pattern.confidence = self.calculate_confidence(cluster)

        # Extract common code structure
        pattern.template = self.extract_common_template(cluster)

        # Identify when to use/avoid
        pattern.use_cases = self.extract_use_cases(cluster)
        pattern.anti_patterns = self.extract_antipatterns(cluster)

        patterns.add(pattern)

    return patterns
```

### 4. Problem Analysis
```python
async def analyze_common_problems(self, tag: str) -> ProblemAnalysis:
    """Analyze common problems and their solutions"""

    problems = ProblemAnalysis()

    # Get frequently asked questions
    frequent_questions = await self.get_frequent_questions(tag, limit=100)

    # Categorize problems
    categories = {
        "errors": [],           # Common errors and fixes
        "performance": [],      # Performance issues
        "compatibility": [],    # Version/platform issues
        "conceptual": [],       # Misunderstandings
        "edge_cases": []        # Edge cases and gotchas
    }

    for question in frequent_questions:
        category = self.categorize_problem(question)
        problem = {
            "description": question.title,
            "frequency": question.view_count,
            "solutions": await self.get_solutions(question),
            "root_cause": self.analyze_root_cause(question),
            "prevention": self.suggest_prevention(question)
        }
        categories[category].append(problem)

    return categories
```

## Intelligence Output Format
```yaml
stackoverflow_intelligence:
  timestamp: "2024-01-20T10:00:00Z"
  questions_analyzed: 500
  solutions_extracted: 1250
  patterns_identified: 67

  top_solutions:
    - problem: "React hooks dependency array infinite loop"
      solution_confidence: 0.95
      votes: 1523
      approach: "Use useCallback and useMemo correctly"
      code: |
        const memoizedCallback = useCallback(
          () => doSomething(a, b),
          [a, b],  // Dependencies
        );
      common_mistakes:
        - "Forgetting dependencies"
        - "Including objects without memoization"

    - problem: "Python async/await with database connections"
      solution_confidence: 0.89
      implementation: "Use connection pooling with async context manager"

  common_problems:
    authentication:
      - issue: "JWT token expiration handling"
        frequency: "asked 230 times"
        best_solution: "Implement refresh token pattern"
        gotchas: ["Silent token expiry", "Race conditions"]

    performance:
      - issue: "N+1 query problem"
        frequency: "asked 450 times"
        solutions: ["Eager loading", "DataLoader pattern", "Query optimization"]

  emerging_topics:
    - "AI-assisted coding workflows" (+200% QoQ)
    - "Edge computing patterns" (+150% QoQ)
    - "WebAssembly integration" (+100% QoQ)

  deprecated_solutions:
    - "componentWillMount in React" (use: "useEffect")
    - "Python 2 syntax" (migrate: "Python 3.10+")
```

## Handoff Protocol
```yaml
produces:
  - type: "solution_database"
    format: "json"
    location: ".ai-workspace/intelligence/solutions/"

  - type: "problem_catalog"
    format: "markdown"
    location: ".ai-workspace/intelligence/problems/"

  - type: "gotcha_warnings"
    format: "yaml"
    location: ".ai-workspace/intelligence/gotchas/"

consumes:
  - type: "technology_stack"
    from: ["project-analyst", "intelligence-orchestrator"]

  - type: "problem_reports"
    from: ["error-detective", "debugger"]

handoff_to:
  - agent: "knowledge-synthesizer"
    data: "validated_solutions"
    when: "after_extraction"

  - agent: "error-detective"
    data: "common_errors"
    when: "on_demand"

  - agent: "backend-developer"
    data: "relevant_solutions"
    when: "encountering_problem"
```

## Continuous Learning
```python
class CommunityLearning:
    """Learn from community voting and feedback"""

    def __init__(self):
        self.solution_tracker = SolutionTracker()
        self.quality_metrics = QualityMetrics()

    async def track_solution_effectiveness(self, solution: Solution, outcome: Outcome):
        """Track which solutions work in practice"""

        self.solution_tracker.record(
            solution_id=solution.id,
            applied_to=outcome.context,
            success=outcome.successful,
            modifications=outcome.modifications_needed
        )

        # Update solution confidence
        if outcome.successful:
            solution.practical_confidence *= 1.05
        else:
            solution.practical_confidence *= 0.95
            await self.analyze_failure_reason(solution, outcome)

    async def identify_quality_signals(self):
        """Learn what makes a good Stack Overflow answer"""

        successful_solutions = self.solution_tracker.get_successful()

        # Identify patterns in good answers
        quality_patterns = {
            "has_explanation": 0.89,  # Good answers explain why
            "includes_example": 0.92,  # Examples improve success
            "mentions_gotchas": 0.85,  # Warnings prevent issues
            "updated_recently": 0.78   # Recent answers more reliable
        }

        self.quality_metrics.update(quality_patterns)
```

## Invocation Examples
```bash
# Harvest Python async solutions
@stackoverflow-harvester search --tags "python,asyncio" --min-score 50

# Find common React errors and solutions
@stackoverflow-harvester problems --tag "reactjs" --category errors

# Get trending topics
@stackoverflow-harvester trends --period 30d --growth-rate ">50%"

# Extract solutions for specific error
@stackoverflow-harvester solve --error "TypeError: Cannot read property" --context react
```

## Performance Optimization
- Cache analyzed questions for 14 days
- Batch API requests (100 items per request)
- Incremental updates for trending topics
- Smart filtering by view count and score
- Deduplicate similar solutions

## Quality Assurance
- Verify answer acceptance rate
- Check answer update recency
- Validate code syntax
- Cross-reference with documentation
- Filter outdated solutions

## Metrics
- Questions analyzed per hour: 100-200
- Solutions extracted per question: 2-5
- Pattern identification accuracy: >80%
- Solution applicability rate: >70%
- False positive rate: <15%