# 🤖 AI Agent Refactoring Guide - Beverly Knits ERP v2

## Quick Start for AI Agents

This project requires comprehensive refactoring from a 18,000-line monolithic file to a modular architecture. All necessary scripts and documentation have been prepared for autonomous AI agent execution.

## 📁 Project Structure

```
beverly_knits_erp_v2/
├── docs/
│   ├── ai_agent_execution_plan.md    # Detailed execution plan
│   └── prompts/                      # Analysis prompts
├── scripts/
│   └── ai_tasks/                     # AI agent scripts
│       ├── orchestrator.py           # Master orchestration script
│       ├── phase1_critical_fixes.py  # Security & performance fixes
│       └── phase2_monolith_decomposition.py  # Architecture refactoring
└── src/
    └── core/
        └── beverly_comprehensive_erp.py  # 18,000-line monolith (TARGET)
```

## 🚀 One-Command Execution

```bash
# Run complete refactoring (all phases)
python scripts/ai_tasks/orchestrator.py

# Run specific phase
python scripts/ai_tasks/orchestrator.py --phase phase1
python scripts/ai_tasks/orchestrator.py --phase phase2
python scripts/ai_tasks/orchestrator.py --phase phase3

# Resume from checkpoints
python scripts/ai_tasks/orchestrator.py --resume
```

## 📋 Refactoring Phases

### Phase 1: Critical Fixes (2-4 hours)
- **Remove hardcoded credentials** (5+ locations)
- **Fix database connection bug** (connection_pool.py:46)
- **Optimize performance** (23 iterrows → vectorization)

### Phase 2: Monolith Decomposition (4-6 hours)
- **Extract core components**:
  - InventoryAnalyzer (1,500 lines)
  - SalesForecastingEngine (1,500 lines)
  - CapacityPlanningEngine (800 lines)
- **Extract API routes** (45+ endpoints)
- **Create component registry**

### Phase 3: Testing & Validation (2-3 hours)
- **Generate unit tests** (80% coverage target)
- **Create integration tests**
- **Validate backwards compatibility**

## 🎯 Success Criteria

```yaml
security_fixes:
  - No hardcoded passwords: grep -r "password=" src/ | wc -l == 0
  - Environment variables used: .env.example exists

architecture:
  - Monolith < 5,000 lines: wc -l src/core/beverly_comprehensive_erp.py
  - Components extracted: ls src/components/*.py | wc -l >= 5
  - API routes separated: src/api/routes.py exists

testing:
  - Coverage > 80%: pytest --cov=src
  - All tests pass: pytest tests/
```

## 📊 Current System Metrics

- **Codebase**: 241,532 lines of Python
- **Monolith**: 18,000 lines (needs decomposition)
- **Critical Issues**: 5 (security vulnerabilities)
- **Performance Issues**: 23 (DataFrame iterations)
- **Test Coverage**: ~40% (needs improvement)

## 🔧 Technical Stack

- **Framework**: Flask 3.0+
- **Data Processing**: Pandas 2.0+
- **ML**: Scikit-learn, XGBoost, Prophet
- **Caching**: Redis
- **Testing**: Pytest

## ⚠️ Critical Warnings

1. **DO NOT** modify dashboard UI (locked)
2. **DO NOT** commit credentials to repository
3. **DO NOT** break API compatibility (use feature flags)
4. **ALWAYS** run tests after changes
5. **ALWAYS** create checkpoints between phases

## 📝 AI Agent Instructions

1. **Start with orchestrator.py** - It manages the entire process
2. **Follow phase order** - Phase 1 → Phase 2 → Phase 3
3. **Use checkpoints** - Saves progress between phases
4. **Check logs** - ai_agent_orchestration.log for details
5. **Validate each phase** - Don't proceed if validation fails

## 🏁 Completion Checklist

- [ ] Phase 1: Security & performance fixes applied
- [ ] Phase 2: Monolith decomposed to < 5,000 lines
- [ ] Phase 3: Tests pass with > 80% coverage
- [ ] All validations pass
- [ ] Report generated (ai_refactoring_report.txt)

## 💡 Tips for AI Agents

1. **Use parallel processing** where possible
2. **Commit after each successful component extraction**
3. **Keep detailed logs** of all changes
4. **Test incrementally** - don't wait until the end
5. **Use the component registry** to track progress

## 📞 Support

If issues arise:
1. Check ai_agent_orchestration.log
2. Review checkpoint files in .ai_checkpoints/
3. Run individual phase scripts for debugging
4. Consult docs/ai_agent_execution_plan.md for detailed guidance

---

**Ready to start?** Run `python scripts/ai_tasks/orchestrator.py` and let the AI agents handle the rest! 🚀