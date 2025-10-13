# Intelligence Gathering Agents

This category contains specialized agents for continuous learning and intelligence gathering from external sources, with integrated Model Context Protocol (MCP) support for federated knowledge management.

## Purpose

These agents work together to:
- Mine patterns from GitHub repositories
- Analyze research papers for innovations
- Harvest solutions from Stack Overflow
- Track breakthrough techniques and trends
- Validate code quality and security
- Organize and curate knowledge base
- Coordinate complex multi-agent workflows
- Provide real-time intelligence updates

## Agent Categories

### Local Intelligence Agents
- **github-pattern-miner** - Mines patterns from top GitHub repositories
- **research-paper-analyzer** - Extracts innovations from academic papers
- **stackoverflow-harvester** - Learns from community solutions
- **knowledge-synthesizer** - Combines intelligence from all sources

### MCP Intelligence Agents (Federated & Real-Time)

#### Pattern Discovery
- **mcp-pattern-hunter** - Multi-source pattern discovery with confidence scoring
  - Discovers patterns from GitHub, Stack Overflow, arXiv
  - Real-time pattern discovery with automatic deduplication
  - Confidence scoring (0.0-1.0) and quality validation
  - Integration with Federated Store and WebSocket sync

#### Innovation Tracking
- **mcp-innovation-tracker** - Research innovation and trend monitoring
  - Tracks arXiv papers, GitHub trending, ML conferences
  - Impact assessment and citation velocity analysis
  - Technology trend trajectory prediction
  - Breakthrough detection and early alerts

#### Quality Validation
- **mcp-quality-guardian** - Code quality and security validation
  - OWASP Top 10 vulnerability scanning
  - Complexity analysis and code smell detection
  - Pre-commit validation and CI/CD integration
  - Quality scoring (0-100) with actionable recommendations

#### Knowledge Management
- **mcp-knowledge-curator** - Knowledge base organization and maintenance
  - Automatic categorization and tagging
  - Duplicate detection and intelligent merging
  - Quality scoring and improvement suggestions
  - Gap analysis and coverage recommendations

#### Multi-Agent Coordination
- **mcp-orchestrator** - Complex workflow coordination
  - Multi-agent task distribution and dependency resolution
  - Parallel execution and result aggregation
  - Failure handling and retry mechanisms
  - End-to-end workflow reporting

## MCP Integration

### Model Context Protocol (MCP)
The MCP intelligence agents connect to federated knowledge services for:
- **Distributed Knowledge Base**: Shared pattern catalog across projects
- **Real-Time Sync**: WebSocket-based live updates
- **Confidence Scoring**: Multi-factor pattern quality assessment
- **Deduplication**: Cross-source duplicate detection
- **Impact Tracking**: Citation velocity and adoption metrics

### Architecture
```
Local Project
    ↓
MCP Intelligence Agents
    ↓
MCP Knowledge Base Server
    ↓
Federated Store (Distributed)
    ↓
External Sources (GitHub, arXiv, Stack Overflow)
```

## Workflow Patterns

### Pattern Discovery Workflow
```
mcp-pattern-hunter (discover)
  → mcp-quality-guardian (validate)
  → mcp-knowledge-curator (categorize & store)
```

### Innovation Tracking Workflow
```
mcp-innovation-tracker (monitor)
  → mcp-knowledge-curator (analyze gaps)
  → mcp-pattern-hunter (fill gaps)
```

### Full Intelligence Pipeline
```
Stage 1 (Parallel):
  - mcp-pattern-hunter (GitHub)
  - mcp-pattern-hunter (Stack Overflow)
  - mcp-innovation-tracker (arXiv)

Stage 2 (Parallel):
  - mcp-quality-guardian (security scan)
  - mcp-quality-guardian (complexity check)

Stage 3 (Sequential):
  - mcp-knowledge-curator (categorize)
  - mcp-knowledge-curator (deduplicate)
  - mcp-knowledge-curator (score quality)
```

## Configuration

### Basic MCP Setup
```yaml
# .agent-workspace/config/intelligence.yml
mcp:
  enabled: true
  knowledge_base: https://mcp-kb.example.com
  authentication:
    token_env: MCP_AUTH_TOKEN

agents:
  pattern_hunter:
    enabled: true
    min_confidence: 0.7
    sources: [github, stackoverflow, arxiv]

  innovation_tracker:
    enabled: true
    impact_threshold: 0.6
    update_frequency: daily

  quality_guardian:
    enabled: true
    min_quality_score: 70
    pre_commit_validation: true

  knowledge_curator:
    enabled: true
    auto_categorize: true
    deduplication_threshold: 0.90
```

### Advanced Configuration
```yaml
# Advanced intelligence configuration
mcp:
  websocket:
    enabled: true
    reconnect: true
    heartbeat_interval: 30s

  federated_store:
    enabled: true
    regions: [us-west, eu-central]
    replication: synchronous

orchestration:
  parallel_execution: true
  max_concurrent_agents: 10
  retry_on_failure: true
  max_retries: 3

quality_thresholds:
  pattern_confidence: 0.7
  innovation_impact: 0.6
  code_quality: 70
  security_critical_block: true
```

## Usage Examples

### Discover Patterns
```bash
# Using MCP Pattern Hunter
mcp-pattern-hunter discover \
  --keywords "async,python,performance" \
  --sources github,stackoverflow \
  --min-confidence 0.7
```

### Track Innovations
```bash
# Using MCP Innovation Tracker
mcp-innovation-tracker monitor \
  --keywords "transformer,efficiency" \
  --impact-threshold 0.8 \
  --alert-on-breakthrough
```

### Validate Code Quality
```bash
# Using MCP Quality Guardian
mcp-quality-guardian validate \
  --staged-files \
  --min-quality-score 70 \
  --security-scan \
  --block-on-critical
```

### Curate Knowledge Base
```bash
# Using MCP Knowledge Curator
mcp-knowledge-curator curate \
  --scope full \
  --deduplicate \
  --categorize \
  --score-quality
```

### Orchestrate Complex Workflow
```bash
# Using MCP Orchestrator
mcp-orchestrator execute \
  --workflow full_intelligence_pipeline \
  --keywords "react,hooks,typescript" \
  --parallel \
  --auto-curate
```

## Integration with Development Agents

All development agents (backend, frontend, etc.) automatically consume intelligence from these agents to:
- **Apply Latest Best Practices**: Discovered patterns inform code generation
- **Avoid Known Vulnerabilities**: Security scans prevent vulnerable patterns
- **Use Proven Patterns**: High-confidence patterns recommended first
- **Optimize Performance**: Performance patterns suggested automatically
- **Stay Current**: Innovation tracking keeps agents up-to-date

## Coordination

### Handoff Structure
```
.agent-workspace/
├── handoffs/
│   └── intelligence/
│       ├── pattern-discovery/
│       ├── innovation-tracking/
│       ├── quality-validation/
│       └── curation-reports/
├── outputs/
│   └── intelligence/
│       ├── patterns/
│       ├── innovations/
│       ├── quality-reports/
│       └── curation-reports/
└── cache/
    └── intelligence/
        ├── patterns/
        ├── innovations/
        └── validation-results/
```

### Real-Time Alerts
- **Breakthrough Detection**: Immediate notification of high-impact innovations
- **Critical Security Issues**: Instant alerts for OWASP Top 10 violations
- **Quality Regression**: Warnings when code quality drops below thresholds
- **Pattern Updates**: Notifications when new high-confidence patterns discovered

## Success Metrics

### Pattern Discovery
- Pattern relevance: >80%
- Confidence accuracy: >90%
- Deduplication rate: >90%
- Discovery latency: <30s

### Innovation Tracking
- Breakthrough detection accuracy: >85%
- Impact prediction accuracy: >75%
- Trend forecast accuracy: >70%
- Alert relevance: >4.0/5.0

### Quality Validation
- Vulnerability detection: >95%
- False positive rate: <5%
- Pre-commit validation: <30s
- Quality score accuracy: >90%

### Knowledge Curation
- Categorization accuracy: >90%
- Duplicate detection: >95%
- Quality improvement: +2 points/month
- Gap identification relevance: >80%

## Best Practices

1. **Regular Curation**: Run knowledge base curation weekly
2. **Continuous Monitoring**: Keep innovation tracker running daily
3. **Pre-Commit Validation**: Always enable quality guardian in git hooks
4. **Workflow Orchestration**: Use orchestrator for complex multi-agent tasks
5. **Threshold Tuning**: Adjust confidence/quality thresholds based on project needs

## Troubleshooting

### MCP Connection Issues
- Check `MCP_AUTH_TOKEN` environment variable
- Verify endpoint URL in configuration
- Test WebSocket connectivity
- Check firewall/proxy settings

### Low Pattern Quality
- Increase `min_confidence` threshold
- Enable stricter quality validation
- Review and improve source filters
- Run quality curator improvement tasks

### Slow Performance
- Enable caching for frequent queries
- Increase parallel execution limits
- Use incremental curation instead of full scans
- Optimize deduplication thresholds

## Version History
- v1.0.0 (2025-10-10): Added MCP intelligence agents
  - mcp-pattern-hunter
  - mcp-innovation-tracker
  - mcp-quality-guardian
  - mcp-knowledge-curator
  - mcp-orchestrator
