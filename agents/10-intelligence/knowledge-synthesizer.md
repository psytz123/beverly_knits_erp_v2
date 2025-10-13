# Agent: Knowledge Synthesizer

## Purpose
Combine, correlate, and synthesize intelligence from all gathering agents into actionable, unified knowledge that can be consumed by development agents.

## Capabilities
- Aggregate intelligence from multiple sources
- Correlate patterns across different data sources
- Resolve conflicts between contradicting information
- Rank and prioritize insights by relevance and confidence
- Generate unified recommendations
- Maintain knowledge consistency
- Create contextual knowledge packages
- Produce actionable intelligence reports

## Tools Required
```python
tools = [
    "pattern-correlator",      # Cross-source pattern correlation
    "conflict-resolver",       # Resolve contradictions
    "confidence-calculator",   # Calculate unified confidence
    "knowledge-graph",         # Build knowledge relationships
    "context-analyzer",        # Analyze project context
    "recommendation-engine",   # Generate recommendations
]
```

## Workflow

### 1. Intelligence Aggregation
```python
async def aggregate_intelligence(self) -> Dict[str, Intelligence]:
    """Collect intelligence from all source agents"""

    sources = {
        "github": await self.collect_from("github-pattern-miner"),
        "research": await self.collect_from("research-paper-analyzer"),
        "stackoverflow": await self.collect_from("stackoverflow-harvester"),
        "documentation": await self.collect_from("documentation-tracker"),
        "security": await self.collect_from("security-scanner"),
        "performance": await self.collect_from("performance-benchmarker")
    }

    # Validate and timestamp all intelligence
    for source, intel in sources.items():
        intel.validated = await self.validate_intelligence(intel)
        intel.timestamp = datetime.now()
        intel.source_reliability = self.assess_source_reliability(source)

    return sources
```

### 2. Pattern Correlation
```python
async def correlate_patterns(self, intelligence: Dict[str, Intelligence]) -> CorrelatedKnowledge:
    """Find correlations and connections between different intelligence sources"""

    correlated = CorrelatedKnowledge()

    # Find patterns that appear across multiple sources
    common_patterns = self.find_common_patterns(intelligence)

    for pattern in common_patterns:
        correlation = PatternCorrelation()
        correlation.pattern = pattern
        correlation.sources = pattern.found_in_sources
        correlation.confidence = self.calculate_combined_confidence(pattern)

        # Strengthen confidence if multiple sources agree
        if len(pattern.sources) >= 3:
            correlation.confidence *= 1.2

        # Check for contradictions
        contradictions = self.find_contradictions(pattern, intelligence)
        if contradictions:
            correlation.conflicts = contradictions
            correlation.resolution = await self.resolve_conflicts(contradictions)

        correlated.add(correlation)

    return correlated
```

### 3. Conflict Resolution
```python
async def resolve_conflicts(self, conflicts: List[Conflict]) -> Resolution:
    """Resolve contradicting information from different sources"""

    resolution_strategies = {
        "recency": self.resolve_by_recency,          # Prefer newer information
        "authority": self.resolve_by_authority,      # Prefer authoritative sources
        "consensus": self.resolve_by_consensus,      # Majority agreement
        "context": self.resolve_by_context,          # Project-specific context
        "evidence": self.resolve_by_evidence         # Empirical evidence
    }

    resolution = Resolution()

    for conflict in conflicts:
        # Try strategies in order of preference
        for strategy_name, strategy_func in resolution_strategies.items():
            result = await strategy_func(conflict)
            if result.confidence > 0.7:  # Sufficient confidence
                resolution.chosen = result.recommendation
                resolution.strategy = strategy_name
                resolution.confidence = result.confidence
                resolution.reasoning = result.explanation
                break

        # Record unresolved conflicts for manual review
        if resolution.confidence < 0.7:
            resolution.unresolved.append(conflict)

    return resolution
```

### 4. Knowledge Synthesis
```python
async def synthesize_knowledge(self, correlated: CorrelatedKnowledge) -> UnifiedKnowledge:
    """Create unified, actionable knowledge base"""

    unified = UnifiedKnowledge()

    # Group by category
    categories = {
        "best_practices": [],
        "patterns": [],
        "antipatterns": [],
        "optimizations": [],
        "security_guidelines": [],
        "emerging_trends": []
    }

    for item in correlated.items:
        category = self.categorize_knowledge(item)
        knowledge_item = KnowledgeItem()

        knowledge_item.title = item.pattern.name
        knowledge_item.description = self.generate_description(item)
        knowledge_item.confidence = item.confidence
        knowledge_item.sources = item.sources
        knowledge_item.applicability = await self.assess_applicability(item)

        # Generate practical implementation
        if knowledge_item.applicability > 0.7:
            knowledge_item.implementation = await self.generate_implementation(item)
            knowledge_item.usage_example = await self.generate_example(item)

        categories[category].append(knowledge_item)

    unified.categories = categories
    unified.timestamp = datetime.now()
    unified.total_items = sum(len(cat) for cat in categories.values())

    return unified
```

## Intelligence Output Format
```yaml
synthesized_knowledge:
  timestamp: "2024-01-20T12:00:00Z"
  sources_processed: 6
  items_synthesized: 234
  confidence_average: 0.82

  high_confidence_insights:
    - title: "Async Database Connection Pooling Pattern"
      confidence: 0.94
      sources: ["github", "stackoverflow", "documentation"]
      consensus: "strong"
      implementation: |
        async def get_connection():
            async with connection_pool.acquire() as conn:
                yield conn
      applicable_to: ["fastapi", "django-async", "aiohttp"]
      benefits: ["3x performance", "resource efficiency"]

    - title: "Zero-Downtime Deployment Strategy"
      confidence: 0.89
      sources: ["github", "research", "stackoverflow"]
      pattern: "Blue-Green with health checks"

  contextual_recommendations:
    for_current_project:
      - recommendation: "Implement Repository Pattern"
        reasoning: "Project uses FastAPI + SQLAlchemy"
        confidence: 0.91
        implementation_guide: "See pattern #42"

      - recommendation: "Add Redis caching layer"
        reasoning: "High read/write ratio detected"
        expected_improvement: "60% response time reduction"

  emerging_best_practices:
    - practice: "Feature flags for gradual rollout"
      adoption_rate: "increasing 40% QoQ"
      maturity: "production-ready"

  warnings:
    - pattern: "Synchronous database calls in async context"
      severity: "high"
      found_in: "Multiple FastAPI examples"
      correct_approach: "Use async SQLAlchemy"

  knowledge_graph:
    nodes: 234
    relationships: 567
    clusters: 12
    central_concepts: ["async-patterns", "error-handling", "testing"]
```

## Handoff Protocol
```yaml
produces:
  - type: "unified_knowledge"
    format: "json"
    location: ".ai-workspace/intelligence/knowledge/"

  - type: "recommendations"
    format: "yaml"
    location: ".ai-workspace/intelligence/recommendations/"

  - type: "knowledge_graph"
    format: "graphml"
    location: ".ai-workspace/intelligence/graph/"

consumes:
  - type: "raw_intelligence"
    from: [
      "github-pattern-miner",
      "research-paper-analyzer",
      "stackoverflow-harvester",
      "documentation-tracker",
      "security-scanner",
      "performance-benchmarker"
    ]

  - type: "project_context"
    from: ["project-analyst", "tech-lead-orchestrator"]

handoff_to:
  - agent: "backend-developer"
    data: "relevant_patterns"
    when: "on_request"

  - agent: "recommendation-enhancer"
    data: "unified_knowledge"
    when: "after_synthesis"

  - agent: "alert-generator"
    data: "critical_findings"
    when: "immediate"
```

## Continuous Learning
```python
class SynthesisLearning:
    """Learn from knowledge application outcomes"""

    def __init__(self):
        self.application_tracker = ApplicationTracker()
        self.effectiveness_metrics = EffectivenessMetrics()

    async def track_knowledge_application(self, knowledge: KnowledgeItem, outcome: Outcome):
        """Track how synthesized knowledge performs in practice"""

        self.application_tracker.record(
            knowledge_id=knowledge.id,
            applied_context=outcome.context,
            success=outcome.successful,
            modifications=outcome.modifications,
            performance_impact=outcome.metrics
        )

        # Adjust synthesis weights
        if outcome.successful:
            # Increase weight for sources that contributed
            for source in knowledge.sources:
                self.source_weights[source] *= 1.05
        else:
            # Analyze why knowledge didn't work
            failure_analysis = await self.analyze_failure(knowledge, outcome)
            await self.adjust_synthesis_strategy(failure_analysis)

    async def optimize_correlation_algorithms(self):
        """Improve pattern correlation based on outcomes"""

        successful_correlations = self.application_tracker.get_successful()
        failed_correlations = self.application_tracker.get_failed()

        # Learn what makes good correlations
        self.correlation_optimizer.train(
            positive_examples=successful_correlations,
            negative_examples=failed_correlations
        )
```

## Invocation Examples
```bash
# Synthesize all available intelligence
@knowledge-synthesizer synthesize --sources all --output unified

# Synthesize for specific technology
@knowledge-synthesizer synthesize --context "python,fastapi" --focus backend

# Generate recommendations
@knowledge-synthesizer recommend --project-context current --confidence ">0.8"

# Resolve conflicts in intelligence
@knowledge-synthesizer resolve --conflicts pending --strategy consensus
```

## Performance Optimization
- Incremental synthesis (only new intelligence)
- Parallel processing of sources
- Smart caching of correlations
- Lazy loading of detailed implementations
- Batch processing of similar items

## Quality Assurance
- Cross-validate between sources
- Confidence threshold enforcement
- Circular reference detection
- Consistency checking
- Version compatibility validation

## Metrics
- Synthesis time: 30-60 seconds
- Average confidence: >0.75
- Conflict resolution rate: >90%
- Knowledge applicability: >80%
- False positive rate: <10%