# Code Duplication Analyst - Quick Reference

## 🎯 Purpose
**Read-only** agent that finds duplicated code and delegates refactoring to specialist agents.

## 🚀 Quick Start

```bash
# Run all analysis tools
cd .claude/agents/08-support/code-duplication-analyst

# 1. Find exact duplicates
python analysis-tools/duplicate_scanner.py --path ./new/src --output-md reports/duplicates.md

# 2. Find similar code (>85%)
python analysis-tools/similarity_checker.py --path ./new/src --threshold 0.85 --output-md reports/similarity.md

# 3. Find common patterns
python analysis-tools/pattern_matcher.py --path ./new/src --output-md reports/patterns.md
```

## 📊 Current Findings

| Category | Count | Redundant LOC | Priority |
|----------|-------|---------------|----------|
| Health Endpoints | 10 | 500+ | High |
| Config Files | 10 | 300+ | High |
| Database Setup | 10 | 400+ | High |
| Test Fixtures | 4 | 200+ | Medium |
| Base Models | 5 | 150+ | High |
| Cache Setup | 8 | 250+ | Medium |

**Total Estimated Reduction**: 1,800+ LOC

## 🎯 Top Priorities

### 1. Extract Health Endpoints (500 LOC)
```bash
# Current: 10 duplicate health.py files
# Target: shared/api/health.py
# Delegate to: refactoring-specialist
```

### 2. Unified Database Factory (400 LOC)
```bash
# Current: 10 duplicate database.py files
# Target: shared/database/factory.py
# Delegate to: python-pro
```

### 3. Base Configuration Class (300 LOC)
```bash
# Current: 10 similar config.py files
# Target: shared/config/base.py
# Delegate to: backend-developer
```

## 🔧 Tool Options

### duplicate_scanner.py
```bash
--path ./new/src              # Root path to scan
--patterns "*.py" "*.js"      # File patterns
--exclude "tests" "__pycache__" # Exclude patterns
--algorithm md5|sha256        # Hash algorithm
--output-json results.json    # JSON output
--output-md results.md        # Markdown report
```

### similarity_checker.py
```bash
--path ./new/src              # Root path
--threshold 0.85              # Similarity threshold (0.0-1.0)
--patterns "*.py"             # File patterns
--exclude "__init__.py"       # Exclude patterns
--output-json results.json    # JSON output
--output-md results.md        # Markdown report
```

### pattern_matcher.py
```bash
--path ./new/src              # Root path
--patterns "*.py"             # File patterns
--exclude "tests"             # Exclude patterns
--output-json results.json    # JSON output
--output-md results.md        # Markdown report
```

## 📋 Detected Patterns

| Pattern | Occurrences | LOC | Delegate To |
|---------|------------|-----|-------------|
| Health Check | 10 | 500 | refactoring-specialist |
| Database Session | 10 | 400 | python-pro |
| Redis Cache | 8 | 250 | python-pro |
| Config Settings | 10 | 300 | backend-developer |
| Kafka Client | 6 | 400 | backend-developer |
| Repository Pattern | 15+ | 300 | python-pro |
| Service Pattern | 12+ | 250 | python-pro |
| Alembic Env | 10 | 500 | backend-developer |

## 🤝 Delegation Map

### refactoring-specialist
- Health check endpoints consolidation
- Test fixture extraction
- API router standardization
- Logging setup unification

### python-pro
- Database connection factory
- Redis cache abstraction
- Base model extraction
- Repository base class
- Service layer base class

### backend-developer
- Base configuration class
- Kafka client consolidation
- CORS middleware standardization
- Alembic environment templates
- Error handler centralization

### code-reviewer
- Comprehensive validation
- Integration testing
- Performance benchmarking
- Security audit

## 📈 Success Metrics

### Before (Current)
- Code Reuse: ~60%
- Duplicate Code: ~15%
- Maintenance: High overhead

### After (Target)
- Code Reuse: >90%
- Duplicate Code: <3%
- Maintenance: -40% overhead
- LOC Reduction: 1,800+ lines

## ⚠️ Agent Constraints

### ❌ NEVER
- Modify code
- Delete files
- Refactor directly
- Make architectural changes
- Update dependencies

### ✅ ALWAYS
- Read-only mode
- Document findings
- Delegate to specialists
- Provide recommendations
- Track progress

## 📂 Output Structure

```
reports/
├── duplicates.md        # Exact duplicates
├── similarity.md        # Similar code (>85%)
├── patterns.md          # Common patterns
├── duplicates.json      # Raw duplicate data
├── similarity.json      # Raw similarity data
└── patterns.json        # Raw pattern data
```

## 🔗 Quick Links

- **Full Documentation**: `README.md`
- **Agent Definition**: `code-duplication-analyst.md`
- **Complete Summary**: `AGENT_SUMMARY.md`
- **Report Template**: `report-templates/duplication-report-template.md`
- **Handoff Template**: `report-templates/handoff-manifest-template.json`

## 💡 Pro Tips

1. **Start Small**: Analyze one service first, then expand
2. **Adjust Threshold**: Try 0.80, 0.85, 0.90 for different results
3. **Exclude Tests Initially**: Focus on production code first
4. **Review Manually**: Not all "duplicates" should be consolidated
5. **Prioritize Impact**: High LOC reduction = high priority

## 🐛 Troubleshooting

**Analysis too slow?**
```bash
--patterns "app/**/*.py" --exclude "tests" "__pycache__"
```

**Too many false positives?**
```bash
--threshold 0.90 --exclude "__init__.py" "base.py"
```

**Missing duplicates?**
```bash
--algorithm sha256 --threshold 0.80
```

## ⚡ One-Liner Commands

```bash
# Quick scan for exact duplicates
python analysis-tools/duplicate_scanner.py --path ./new/src --output-md reports/quick-scan.md

# High similarity only (>90%)
python analysis-tools/similarity_checker.py --path ./new/src --threshold 0.90 --output-md reports/high-sim.md

# Pattern detection with JSON output
python analysis-tools/pattern_matcher.py --path ./new/src --output-json reports/patterns.json

# Full analysis suite
for tool in duplicate_scanner similarity_checker pattern_matcher; do
  python analysis-tools/$tool.py --path ./new/src --output-md reports/$tool.md
done
```

## 📞 Support

Questions? Check:
1. This quick reference
2. `README.md` for detailed guide
3. `AGENT_SUMMARY.md` for complete overview
4. Tool source code for implementation details
