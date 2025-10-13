# Code Duplication Analysis Report

**Report ID**: `{report_id}`
**Generated**: `{timestamp}`
**Analyzed Path**: `{root_path}`
**Analysis Type**: Comprehensive (Exact Duplicates + Similarity + Patterns)

---

## Executive Summary

### Key Metrics
- **Total Files Scanned**: `{total_files_scanned}`
- **Exact Duplicate Groups**: `{exact_duplicate_groups}`
- **High Similarity Pairs (>{threshold}%)**: `{high_similarity_pairs}`
- **Common Patterns Detected**: `{pattern_count}`
- **Total Redundant LOC**: `{total_redundant_loc}` lines
- **Potential LOC Reduction**: `{potential_loc_reduction}` lines (`{reduction_percentage}%`)

### Impact Analysis
- **Maintenance Overhead**: `{maintenance_overhead}` (High/Medium/Low)
- **Code Reuse Score**: `{code_reuse_score}%` (Current)
- **Target Code Reuse**: `{target_code_reuse}%`
- **Technical Debt**: `{technical_debt_score}` (0-100 scale)

### Priority Recommendations
1. **Immediate Action**: `{immediate_action_count}` items
2. **Short Term (1-2 weeks)**: `{short_term_count}` items
3. **Long Term (1-3 months)**: `{long_term_count}` items

---

## Part 1: Exact Duplicate Files

### Summary
- **Duplicate Groups Found**: `{duplicate_groups}`
- **Total Duplicate Files**: `{total_duplicate_files}`
- **Redundant Files**: `{redundant_files}` (excluding originals)
- **Redundant LOC**: `{duplicate_redundant_loc}` lines

### Duplicates by Category

#### Health Endpoints (`{health_endpoint_count}` groups)
**Recommended Action**: Extract to `shared/api/health.py`
**Delegate To**: `refactoring-specialist`
**Priority**: High
**Estimated LOC Reduction**: `{health_endpoint_loc}` lines

**Duplicate Sets**:
```
{health_endpoint_duplicates}
```

#### Configuration Files (`{config_count}` groups)
**Recommended Action**: Create base config in `shared/config/base.py`
**Delegate To**: `python-pro`
**Priority**: High
**Estimated LOC Reduction**: `{config_loc}` lines

**Duplicate Sets**:
```
{config_duplicates}
```

#### Database Setup (`{database_count}` groups)
**Recommended Action**: Unified database factory in `shared/database/`
**Delegate To**: `python-pro`
**Priority**: High
**Estimated LOC Reduction**: `{database_loc}` lines

**Duplicate Sets**:
```
{database_duplicates}
```

#### Test Fixtures (`{test_fixture_count}` groups)
**Recommended Action**: Consolidate to `shared/testing/fixtures.py`
**Delegate To**: `refactoring-specialist`
**Priority**: Medium
**Estimated LOC Reduction**: `{test_fixture_loc}` lines

**Duplicate Sets**:
```
{test_fixture_duplicates}
```

#### Other Categories
{other_category_duplicates}

---

## Part 2: Semantic Similarity Analysis

### Summary
- **Similarity Threshold**: `{similarity_threshold}%`
- **High Similarity Matches**: `{similarity_matches}`
- **Average Similarity**: `{average_similarity}%`
- **Affected Files**: `{similarity_affected_files}`

### High Similarity Pairs (>{similarity_threshold}%)

#### Configuration Patterns (`{config_similarity_count}` pairs, avg `{config_avg_similarity}%`)

**Pair 1**: `{similarity_score_1}%` similarity
- File A: `{file_a_1}`
- File B: `{file_b_1}`
- Matching: `{matching_imports_1}` imports, `{matching_classes_1}` classes, `{matching_functions_1}` functions
- **Recommendation**: `{recommendation_1}`
- **Delegate To**: `{delegate_1}`

{additional_similarity_pairs}

---

## Part 3: Common Pattern Analysis

### Summary
- **Patterns Detected**: `{patterns_detected}`
- **Total Pattern Occurrences**: `{total_pattern_occurrences}`
- **Estimated LOC Reduction**: `{pattern_loc_reduction}` lines

### Patterns by Impact

#### Pattern: {pattern_name_1} (`{pattern_occurrences_1}` occurrences)
**Category**: `{pattern_category_1}`
**Description**: `{pattern_description_1}`
**Affected Files**: `{pattern_files_1}`
**Estimated LOC Reduction**: `{pattern_loc_1}` lines
**Recommendation**: `{pattern_recommendation_1}`
**Delegate To**: `{pattern_delegate_1}`
**Priority**: `{pattern_priority_1}`

**Sample Occurrences**:
```python
{pattern_samples_1}
```

{additional_patterns}

---

## Part 4: Consolidated Findings

### By Category

| Category | Duplicates | Similarity | Patterns | Total LOC |
|----------|-----------|------------|----------|-----------|
| Health Endpoints | {cat_health_dup} | {cat_health_sim} | {cat_health_pat} | {cat_health_loc} |
| Configuration | {cat_config_dup} | {cat_config_sim} | {cat_config_pat} | {cat_config_loc} |
| Database Setup | {cat_db_dup} | {cat_db_sim} | {cat_db_pat} | {cat_db_loc} |
| API Patterns | {cat_api_dup} | {cat_api_sim} | {cat_api_pat} | {cat_api_loc} |
| Testing | {cat_test_dup} | {cat_test_sim} | {cat_test_pat} | {cat_test_loc} |
| Middleware | {cat_mid_dup} | {cat_mid_sim} | {cat_mid_pat} | {cat_mid_loc} |
| Other | {cat_other_dup} | {cat_other_sim} | {cat_other_pat} | {cat_other_loc} |

### By Service

| Service | Duplicate Files | Similarity Issues | Pattern Occurrences | Priority |
|---------|----------------|-------------------|---------------------|----------|
{service_breakdown}

---

## Part 5: Refactoring Recommendations

### High Priority (Week 1-2)

#### 1. Extract Health Endpoints
- **Files Affected**: {health_files}
- **Action**: Create `shared/api/health.py` with configurable health checks
- **LOC Reduction**: {health_loc_reduction}
- **Delegate To**: `refactoring-specialist`
- **Validation**: Unit tests + integration tests

#### 2. Consolidate Database Setup
- **Files Affected**: {db_files}
- **Action**: Create unified database factory in `shared/database/`
- **LOC Reduction**: {db_loc_reduction}
- **Delegate To**: `python-pro`
- **Validation**: Connection pool tests

#### 3. Unified Configuration Base
- **Files Affected**: {config_files}
- **Action**: Create `shared/config/base.py` with common settings
- **LOC Reduction**: {config_loc_reduction}
- **Delegate To**: `backend-developer`
- **Validation**: Environment-specific tests

### Medium Priority (Week 3-4)

{medium_priority_tasks}

### Low Priority (Month 2-3)

{low_priority_tasks}

---

## Part 6: Implementation Roadmap

### Phase 1: Foundation (Week 1-2)
**Goal**: Establish shared library structure

1. Create `shared/` directory structure
2. Set up shared library packaging (`pyproject.toml`)
3. Implement high-priority shared components
4. Create comprehensive test suite
5. Update CI/CD for shared library

**Deliverables**:
- `shared/api/health.py`
- `shared/database/factory.py`
- `shared/config/base.py`
- `shared/testing/fixtures.py`

### Phase 2: Migration (Week 3-6)
**Goal**: Migrate all services to shared library

1. Update service dependencies
2. Replace duplicates with shared imports
3. Validate each service independently
4. Run integration tests
5. Monitor for regressions

**Deliverables**:
- All services using shared components
- Updated documentation
- Reduced codebase by {phase2_loc_reduction} LOC

### Phase 3: Optimization (Week 7-8)
**Goal**: Refine and optimize

1. Address medium-priority patterns
2. Optimize shared library performance
3. Add monitoring and observability
4. Document best practices
5. Training for development team

**Deliverables**:
- Optimized shared library
- Development guidelines
- Training materials

### Phase 4: Maintenance (Ongoing)
**Goal**: Prevent future duplication

1. Establish coding standards
2. Implement pre-commit hooks
3. Regular duplication audits
4. Continuous improvement

---

## Part 7: Risk Assessment

### High Risk Items
{high_risk_items}

### Mitigation Strategies
{mitigation_strategies}

### Rollback Plan
{rollback_plan}

---

## Part 8: Success Metrics

### Baseline Metrics (Current State)
- Code Reuse: {current_code_reuse}%
- Duplicate Code: {current_duplicate_code}%
- Maintenance Overhead: {current_maintenance_overhead}
- Average Bug Fix Time: {current_bug_fix_time}

### Target Metrics (Post-Refactoring)
- Code Reuse: {target_code_reuse}%
- Duplicate Code: <3%
- Maintenance Overhead: {target_maintenance_overhead}
- Average Bug Fix Time: {target_bug_fix_time}

### Measurement Plan
- Track metrics weekly
- Compare before/after
- Measure developer productivity
- Monitor system performance

---

## Part 9: Next Steps

### Immediate Actions (This Week)
1. [ ] Review this report with tech lead
2. [ ] Prioritize refactoring tasks
3. [ ] Assign tasks to specialist agents
4. [ ] Create detailed task breakdown
5. [ ] Set up tracking and monitoring

### Short Term (Next 2 Weeks)
1. [ ] Implement Phase 1 deliverables
2. [ ] Create shared library structure
3. [ ] Begin service migration
4. [ ] Comprehensive testing
5. [ ] Documentation updates

### Long Term (Next 2-3 Months)
1. [ ] Complete all service migrations
2. [ ] Optimize shared library
3. [ ] Establish governance
4. [ ] Monitor and improve continuously

---

## Appendix A: Detailed File Listings

### All Duplicate Files
{detailed_duplicate_listing}

### All Similarity Matches
{detailed_similarity_listing}

### All Pattern Occurrences
{detailed_pattern_listing}

---

## Appendix B: Delegation Manifest

See `handoff-manifest.json` for complete task delegation details.

---

*This report was generated by the Code Duplication Analyst Agent*
*Analysis Tools: duplicate_scanner.py, similarity_checker.py, pattern_matcher.py*
*For questions or issues, contact the workspace coordinator*
