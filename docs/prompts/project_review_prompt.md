# Claude.code Comprehensive Project Analysis Prompt

**For use with Claude.code in Cursor IDE with MCP tools**

Please conduct a thorough automated analysis of this project using Claude.code capabilities and available MCP tools to generate a comprehensive strategic assessment:

## MCP Tool Orchestration

### Repository Analysis Pipeline
```python
# Comprehensive project scanning using MCP tools
project_analysis = {
    "codebase_metrics": mcp_filesystem.analyze_structure(),
    "git_history": mcp_git.analyze_evolution(),
    "dependency_audit": mcp_package_manager.scan_dependencies(),
    "security_assessment": mcp_security.vulnerability_scan(),
    "performance_profile": mcp_profiler.analyze_bottlenecks(),
    "test_coverage": mcp_testing.coverage_analysis()
}
```

## Automated Assessment Categories

### 1. Current State Analysis
```
EXECUTE with Claude.code:
1. Analyze overall project health using integrated metrics
2. Map progress against TODO/FIXME comments and incomplete features
3. Generate code quality dashboard using static analysis
4. Assess documentation completeness via content analysis
5. Calculate test coverage and identify gaps
6. Create technical debt inventory with priority scoring
```

### 2. Architecture & Technical Infrastructure Evaluation
```python
# Use Cursor IDE's understanding capabilities
architecture_analysis = {
    "component_mapping": analyze_module_structure(),
    "dependency_graph": map_import_relationships(),
    "data_flow": trace_execution_paths(),
    "api_endpoints": catalog_routes_and_handlers(),
    "database_schema": analyze_data_models(),
    "external_integrations": map_third_party_services()
}
```

### 3. Security & Compliance Assessment
```
SECURITY_AUDIT using MCP tools:
1. Scan for common vulnerabilities (SQL injection, XSS, etc.)
2. Analyze authentication and authorization implementations
3. Check for exposed secrets and credentials
4. Validate input sanitization coverage
5. Assess encryption and data protection measures
6. Review dependency security status
```

### 4. Performance & Scalability Analysis
```python
# Automated performance profiling
performance_metrics = {
    "query_optimization": analyze_database_queries(),
    "memory_usage": profile_memory_patterns(),
    "cpu_bottlenecks": identify_expensive_operations(),
    "io_efficiency": analyze_file_and_network_operations(),
    "caching_opportunities": find_expensive_computations(),
    "scalability_limits": assess_resource_constraints()
}
```

## Claude.code Execution Framework

### Phase 1: Automated Discovery
```python
def comprehensive_project_scan():
    """Execute comprehensive project analysis using all available MCP tools"""
    
    # File system analysis
    filesystem_data = mcp_filesystem.deep_scan(
        include_metrics=True,
        analyze_structure=True,
        find_duplicates=True
    )
    
    # Code quality assessment
    quality_metrics = mcp_linter.full_analysis(
        check_complexity=True,
        analyze_patterns=True,
        find_code_smells=True
    )
    
    # Security evaluation
    security_report = mcp_security.comprehensive_scan(
        check_dependencies=True,
        scan_code_patterns=True,
        validate_configurations=True
    )
    
    return {
        "filesystem": filesystem_data,
        "quality": quality_metrics,
        "security": security_report
    }
```

### Phase 2: Strategic Analysis Generation
```
ANALYSIS_ENGINE:
1. Synthesize findings across all MCP tool outputs
2. Identify critical dependencies and single points of failure
3. Generate risk assessment matrix with impact/probability scoring
4. Create improvement roadmap with effort estimation
5. Prioritize recommendations based on business impact
```

### Phase 3: Actionable Recommendations
```python
def generate_strategic_recommendations(analysis_data):
    """Convert analysis into actionable strategic recommendations"""
    
    recommendations = {
        "immediate_actions": prioritize_critical_issues(),
        "short_term_goals": plan_quick_wins(),
        "long_term_strategy": design_architectural_improvements(),
        "resource_requirements": calculate_effort_estimates(),
        "risk_mitigation": develop_contingency_plans()
    }
    
    return create_execution_roadmap(recommendations)
```

## Cursor IDE Intelligence Integration

### AI-Powered Code Understanding
- Leverage Cursor's semantic understanding for architecture analysis
- Use intelligent code navigation to trace feature implementations
- Apply AI-assisted pattern recognition for identifying technical debt

### Automated Documentation Generation
- Generate API documentation from code analysis
- Create architectural diagrams from component relationships
- Produce user workflow documentation from UI/API interactions

## MCP Tool Coordination

### Multi-Tool Analysis Pipeline
```bash
# Orchestrated tool execution
mcp_filesystem.analyze() | mcp_git.history_analysis() | mcp_security.scan()
mcp_testing.coverage() | mcp_performance.profile() | mcp_dependencies.audit()
```

### Data Aggregation & Synthesis
```python
# Combine insights from multiple tools
synthesis_engine = MCPAnalysisSynthesizer()
comprehensive_report = synthesis_engine.combine([
    filesystem_analysis,
    git_history_analysis,
    security_scan_results,
    performance_profiling,
    dependency_audit,
    test_coverage_report
])
```

## Output Generation

### Executive Summary Dashboard
```markdown
# Project Health Dashboard

## Key Metrics
- Overall Health Score: [Calculated score]
- Technical Debt Index: [Quantified measure]
- Security Risk Level: [Assessment]
- Performance Grade: [Benchmark against standards]
- Maintainability Score: [Code quality metric]

## Critical Findings
- [Top 3 issues requiring immediate attention]
- [Key blockers for production readiness]
- [High-impact improvement opportunities]
```

### Detailed Technical Analysis
```markdown
## Architecture Assessment
### Component Health
[Component-by-component analysis with health scores]

### Dependency Analysis
[Critical dependencies, versions, security status]

### Performance Bottlenecks
[Identified performance issues with impact analysis]

### Security Vulnerabilities
[Security findings with severity ratings and remediation steps]
```

### Strategic Roadmap
```markdown
## Implementation Roadmap

### Phase 1: Critical Fixes (Week 1-2)
- [Immediate security vulnerabilities]
- [Critical performance issues]
- [Blocking technical debt]

### Phase 2: Foundation Improvements (Week 3-6)
- [Architecture optimizations]
- [Testing infrastructure]
- [Documentation completion]

### Phase 3: Enhancement & Optimization (Week 7-12)
- [Performance optimizations]
- [Feature completions]
- [Long-term maintainability improvements]
```

## Continuous Monitoring Setup

### Automated Health Checks
```python
# Set up ongoing project monitoring
monitoring_config = {
    "daily_scans": ["security", "dependencies"],
    "weekly_analysis": ["performance", "code_quality"],
    "monthly_review": ["architecture", "technical_debt"]
}
```

Execute this comprehensive analysis using the full capabilities of Claude.code's understanding, Cursor IDE's intelligence, and the complete suite of available MCP tools to generate actionable strategic insights.