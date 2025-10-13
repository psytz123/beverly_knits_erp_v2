# Code Duplication Analyst Agent

## Overview

A **read-only analysis agent** that identifies duplicated code, redundant files, and refactoring opportunities across the codebase. This agent operates in pure documentation mode—it never modifies code, only analyzes and delegates to appropriate specialist agents.

## Purpose

Prevent technical debt by catching when AI coding assistants recreate existing functionality instead of reusing code. Creates comprehensive documentation for refactoring teams to improve code reuse and maintainability.

## Quick Start

### 1. Run Full Analysis

```bash
# Navigate to agent directory
cd .claude/agents/08-support/code-duplication-analyst

# Run all analysis tools
python analysis-tools/duplicate_scanner.py --path ./new/src --output-json reports/duplicates.json --output-md reports/duplicates.md

python analysis-tools/similarity_checker.py --path ./new/src --threshold 0.85 --output-json reports/similarity.json --output-md reports/similarity.md

python analysis-tools/pattern_matcher.py --path ./new/src --output-json reports/patterns.json --output-md reports/patterns.md
```

### 2. Review Reports

All reports are generated in the `reports/` directory:
- `duplicates.md` - Exact duplicate files
- `similarity.md` - Semantically similar code (>85%)
- `patterns.md` - Common boilerplate patterns

### 3. Create Handoff Tasks

Use the findings to populate `handoff-manifest.json` for delegation to specialist agents.

## Agent Components

### Analysis Tools

#### 1. **duplicate_scanner.py**
- **Purpose**: Find exact duplicate files using MD5/SHA256 hashing
- **Output**: Groups of identical files with LOC reduction potential
- **Usage**:
  ```bash
  python duplicate_scanner.py --path ./new/src --algorithm md5 --output-json duplicates.json
  ```

#### 2. **similarity_checker.py**
- **Purpose**: Find semantically similar code using AST comparison
- **Output**: Pairs of files with similarity scores and refactoring recommendations
- **Usage**:
  ```bash
  python similarity_checker.py --path ./new/src --threshold 0.85 --output-json similarity.json
  ```

#### 3. **pattern_matcher.py**
- **Purpose**: Detect repeated code patterns and boilerplate
- **Output**: Pattern occurrences with consolidation recommendations
- **Usage**:
  ```bash
  python pattern_matcher.py --path ./new/src --output-json patterns.json
  ```

### Report Templates

#### duplication-report-template.md
Comprehensive report structure covering:
- Executive summary with metrics
- Exact duplicates by category
- Semantic similarity analysis
- Pattern analysis
- Consolidated findings
- Refactoring roadmap
- Implementation phases
- Success metrics

#### handoff-manifest-template.json
Task delegation manifest with:
- Individual tasks for specialist agents
- Priority levels and effort estimates
- Validation criteria
- Dependencies and rollback plans
- Execution phases
- Risk mitigation strategies

## Current Findings (Example Output)

Based on initial scan of the ERP codebase:

### Exact Duplicates Found
- **20+ duplicate file groups**
- **~500-800 redundant LOC**
- Categories: health.py, config.py, database.py, conftest.py, base.py, migrations/env.py

### High Similarity (>85%)
- Multiple config.py files with 90%+ similarity
- Database setup patterns across services
- Test fixtures nearly identical

### Common Patterns
- Health check endpoints (10 occurrences)
- Database session factories (10 occurrences)
- Redis cache setup (8 occurrences)
- Kafka client initialization (6 occurrences)
- Repository patterns (15+ occurrences)

## Delegation Strategy

### Agent Assignments

1. **refactoring-specialist**
   - Health check endpoint consolidation
   - Test fixture extraction
   - API router standardization
   - Logging setup unification

2. **python-pro**
   - Database connection factory design
   - Redis cache client abstraction
   - Base model extraction
   - Repository pattern base class
   - Service layer base class

3. **backend-developer**
   - Configuration base class
   - Kafka client consolidation
   - CORS middleware standardization
   - Alembic environment templates
   - Error handler centralization

4. **code-reviewer**
   - Comprehensive validation of all changes
   - Integration testing
   - Performance benchmarking
   - Security audit

5. **tech-lead-orchestrator**
   - Architecture review
   - Shared library design approval
   - Roadmap validation

## Expected Outcomes

### Metrics Improvement
- **Code Reuse**: From ~60% to >90%
- **Duplicate Code**: From ~15% to <3%
- **LOC Reduction**: 500-800 lines initially, 1500-2000 total
- **Maintenance Overhead**: Reduce by 30-40%

### Quality Improvements
- Single source of truth for common patterns
- Easier onboarding for new developers
- Faster feature development
- Reduced bug surface area
- Improved test coverage

## Usage Examples

### Example 1: Find All Duplicates

```bash
python analysis-tools/duplicate_scanner.py \
  --path ./new/src \
  --patterns "*.py" \
  --algorithm sha256 \
  --output-json ./reports/all-duplicates.json \
  --output-md ./reports/all-duplicates.md
```

### Example 2: High Similarity Analysis

```bash
python analysis-tools/similarity_checker.py \
  --path ./new/src/services \
  --threshold 0.90 \
  --patterns "*.py" \
  --exclude "__init__.py" "test_*.py" \
  --output-md ./reports/high-similarity.md
```

### Example 3: Pattern Detection

```bash
python analysis-tools/pattern_matcher.py \
  --path ./new/src/services \
  --patterns "*.py" \
  --output-json ./reports/patterns.json \
  --output-md ./reports/patterns.md
```

## Integration with Workspace Protocol

### Directory Structure
```
.agent-workspace/
├── outputs/
│   └── duplication-analysis/
│       ├── duplicates.json
│       ├── similarity.json
│       ├── patterns.json
│       └── comprehensive-report.md
├── handoffs/
│   └── active/
│       └── handoff-duplication-refactoring.json
├── context/
│   └── duplication-findings.md
└── logs/
    └── duplication-analyst-YYYY-MM-DD.log
```

### Workflow Integration
1. **Initiation**: Triggered by `workspace-coordinator` or `tech-lead-orchestrator`
2. **Analysis**: Runs all three analysis tools
3. **Documentation**: Generates comprehensive reports
4. **Delegation**: Creates handoff manifest for specialist agents
5. **Tracking**: Monitors refactoring progress (read-only)

## Best Practices

### When to Run Analysis
- ✅ After major feature implementations
- ✅ Before major refactoring initiatives
- ✅ Monthly code health checks
- ✅ When onboarding new team members
- ✅ After AI-assisted development sessions

### What NOT to Do
- ❌ Never modify source code directly
- ❌ Never delete files without delegation
- ❌ Never implement refactoring without handoff
- ❌ Never skip validation step
- ❌ Never deploy changes without testing

### Tips for Effective Analysis
1. **Run incrementally**: Start with one service, then expand
2. **Set realistic thresholds**: 85% similarity is a good starting point
3. **Review patterns manually**: Not all detected patterns need refactoring
4. **Prioritize by impact**: Focus on high-LOC reduction opportunities
5. **Validate assumptions**: Check if "duplicates" are intentional

## Troubleshooting

### Issue: Analysis taking too long
**Solution**: Use more specific file patterns or exclude test directories
```bash
--patterns "app/**/*.py" --exclude "tests" "__pycache__"
```

### Issue: Too many false positives
**Solution**: Increase similarity threshold or exclude common files
```bash
--threshold 0.90 --exclude "__init__.py" "base.py"
```

### Issue: Missing duplicates
**Solution**: Use multiple hash algorithms and lower thresholds
```bash
--algorithm sha256  # More thorough than md5
--threshold 0.80    # Lower threshold catches more
```

## Contributing

### Adding New Pattern Definitions
Edit `analysis-tools/pattern_matcher.py` and add to `PATTERNS` dictionary:

```python
"new_pattern": {
    "name": "Pattern Name",
    "category": "category",
    "regex": r"pattern_regex",
    "description": "What this pattern does",
    "recommendation": "How to refactor",
    "delegate_to": "agent-name",
    "estimated_loc": 20,
}
```

### Improving Detection Algorithms
1. Enhance AST similarity in `similarity_checker.py`
2. Add new hash algorithms to `duplicate_scanner.py`
3. Expand pattern library in `pattern_matcher.py`

## Version History

- **v1.0.0** (2025-10-05): Initial release
  - Exact duplicate detection
  - Semantic similarity analysis
  - Pattern matching
  - Report generation
  - Handoff manifest creation

## Support

For issues or questions:
1. Check this README
2. Review analysis tool source code
3. Consult `code-duplication-analyst.md` for agent definition
4. Contact workspace coordinator

## License

Internal tool for Beverly Knits ERP System development.
