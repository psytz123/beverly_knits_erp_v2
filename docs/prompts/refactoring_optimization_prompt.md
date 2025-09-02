# Claude.code Refactoring & Performance Optimization Prompt

**For use with Claude.code in Cursor IDE with MCP tools**

Please execute comprehensive refactoring and performance optimization using Claude.code's intelligent analysis capabilities and available MCP tools:

## MCP Tool Optimization Pipeline

### Code Analysis & Profiling Setup
```python
# Initialize comprehensive optimization analysis
optimization_pipeline = {
    "complexity_analysis": mcp_analyzer.calculate_cyclomatic_complexity(),
    "performance_profiling": mcp_profiler.deep_performance_scan(),
    "memory_analysis": mcp_memory.usage_patterns(),
    "database_optimization": mcp_database.query_analyzer(),
    "dependency_optimization": mcp_deps.unused_dependency_scan()
}
```

## Automated Refactoring Execution

### 1. Code Structure Optimization
```python
def execute_structural_refactoring():
    """Use Claude.code to intelligently refactor code structure"""
    
    # Identify refactoring opportunities
    refactor_targets = mcp_analyzer.find_refactoring_opportunities([
        "large_functions",      # Functions > 50 lines
        "duplicate_code",       # Code duplication > 80% similarity
        "complex_conditionals", # Nested if statements > 3 levels
        "long_parameter_lists", # Functions with > 5 parameters
        "tight_coupling",       # High coupling between modules
        "code_smells"          # Various anti-patterns
    ])
    
    # Execute automated refactoring
    for target in refactor_targets:
        apply_refactoring_pattern(target)
        validate_refactoring(target)
        commit_changes(target)
```

### 2. Performance Optimization Engine
```python
def performance_optimization_pipeline():
    """Systematic performance optimization using MCP tools"""
    
    # Profile current performance
    performance_baseline = mcp_profiler.comprehensive_profile([
        "cpu_usage",
        "memory_consumption", 
        "database_queries",
        "io_operations",
        "network_requests"
    ])
    
    # Identify optimization opportunities
    optimizations = {
        "algorithm_improvements": find_inefficient_algorithms(),
        "database_optimizations": analyze_query_performance(),
        "caching_opportunities": identify_expensive_operations(),
        "memory_optimizations": find_memory_leaks_and_waste(),
        "io_optimizations": analyze_file_operations()
    }
    
    # Apply optimizations
    for category, items in optimizations.items():
        execute_optimization_category(category, items)
```

## Claude.code Intelligence Integration

### Intelligent Pattern Recognition
```
REFACTORING_PATTERNS with Cursor AI:
1. Extract Method: Automatically break down large functions
2. Extract Class: Identify cohesive functionality for new classes
3. Move Method: Relocate methods to appropriate classes
4. Rename Variable/Method: Improve naming consistency
5. Remove Duplication: Consolidate similar code blocks
6. Simplify Conditional: Reduce complex boolean logic
```

### AI-Powered Architecture Improvements
```python
# Use Cursor's semantic understanding
architectural_improvements = cursor_ai.analyze_architecture([
    "separation_of_concerns",
    "single_responsibility", 
    "open_closed_principle",
    "dependency_inversion",
    "interface_segregation"
])
```

## Performance Optimization Categories

### 1. Database Query Optimization
```python
def optimize_database_performance():
    """Automated database optimization using MCP tools"""
    
    # Analyze all database queries
    query_analysis = mcp_database.analyze_queries([
        "n_plus_one_problems",
        "missing_indexes",
        "inefficient_joins",
        "large_result_sets",
        "unnecessary_queries"
    ])
    
    # Apply optimizations
    optimizations = {
        "add_indexes": create_missing_indexes(),
        "optimize_queries": rewrite_inefficient_queries(),
        "implement_caching": cache_expensive_queries(),
        "batch_operations": group_similar_queries()
    }
    
    return execute_database_optimizations(optimizations)
```

### 2. Algorithm & Data Structure Optimization
```python
def optimize_algorithms():
    """Identify and improve algorithmic complexity"""
    
    complexity_analysis = mcp_analyzer.algorithm_complexity([
        "nested_loops",        # O(n²) patterns
        "linear_searches",     # Replace with binary search/hash
        "recursive_inefficiency", # Memoization opportunities
        "string_concatenation",   # StringBuilder patterns
        "collection_inefficiency" # Wrong data structure usage
    ])
    
    # Apply algorithmic improvements
    for inefficiency in complexity_analysis:
        improved_algorithm = generate_optimized_version(inefficiency)
        benchmark_improvement(inefficiency, improved_algorithm)
        if improvement_significant():
            apply_optimization(improved_algorithm)
```

### 3. Memory Usage Optimization
```python
def optimize_memory_usage():
    """Memory optimization using profiling data"""
    
    memory_issues = mcp_memory.find_issues([
        "memory_leaks",
        "excessive_object_creation",
        "large_object_retention",
        "inefficient_collections",
        "string_memory_waste"
    ])
    
    # Apply memory optimizations
    optimizations = {
        "object_pooling": implement_object_pools(),
        "string_optimization": optimize_string_operations(),
        "collection_tuning": use_efficient_collections(),
        "dispose_patterns": implement_proper_disposal()
    }
    
    return execute_memory_optimizations(optimizations)
```

## Refactoring Execution Framework

### Phase 1: Analysis & Planning
```python
def create_refactoring_plan():
    """Generate comprehensive refactoring strategy"""
    
    analysis = {
        "code_complexity": mcp_analyzer.complexity_metrics(),
        "performance_bottlenecks": mcp_profiler.bottleneck_analysis(),
        "architectural_issues": cursor_ai.architecture_analysis(),
        "code_quality_metrics": mcp_quality.full_assessment()
    }
    
    # Prioritize refactoring tasks
    refactoring_plan = prioritize_refactoring_tasks(analysis)
    return refactoring_plan
```

### Phase 2: Automated Refactoring Execution
```python
def execute_refactoring_phase(refactoring_plan):
    """Execute refactoring with validation at each step"""
    
    for task in refactoring_plan:
        # Pre-refactoring state
        pre_state = capture_system_state()
        
        # Execute refactoring
        refactoring_result = apply_refactoring(task)
        
        # Validate refactoring
        validation_result = validate_refactoring(pre_state, refactoring_result)
        
        if validation_result.success:
            commit_refactoring(task, refactoring_result)
        else:
            rollback_refactoring(task, pre_state)
            log_refactoring_failure(task, validation_result)
```

### Phase 3: Performance Validation
```python
def validate_performance_improvements():
    """Measure and validate performance improvements"""
    
    # Before/after performance comparison
    performance_comparison = {
        "execution_time": compare_execution_times(),
        "memory_usage": compare_memory_consumption(),
        "database_performance": compare_query_performance(),
        "resource_utilization": compare_resource_usage()
    }
    
    # Generate improvement metrics
    improvement_metrics = calculate_improvement_metrics(performance_comparison)
    return improvement_metrics
```

## MCP Tool Coordination

### Multi-Tool Optimization Pipeline
```bash
# Orchestrated optimization execution
mcp_profiler.baseline() → mcp_analyzer.complexity() → cursor_ai.refactor() → mcp_profiler.validate()
```

### Continuous Optimization Monitoring
```python
# Set up ongoing optimization monitoring
optimization_monitoring = {
    "performance_regression_detection": True,
    "code_quality_tracking": True,
    "memory_usage_monitoring": True,
    "database_performance_alerts": True
}
```

## Output Specification

### Optimization Report Generation
```markdown
# Refactoring & Optimization Execution Report

## Performance Improvements
### Execution Time
- Before: [baseline metrics]
- After: [optimized metrics] 
- Improvement: [percentage improvement]

### Memory Usage
- Memory reduction: [MB saved]
- Object creation reduction: [percentage]
- Garbage collection improvement: [metrics]

### Database Performance
- Query execution time improvement: [percentage]
- N+1 queries eliminated: [count]
- Index optimizations applied: [count]

## Code Quality Improvements
- Cyclomatic complexity reduction: [before/after]
- Code duplication eliminated: [percentage]
- Functions refactored: [count]
- Design patterns implemented: [list]

## Architecture Enhancements
- Separation of concerns improvements
- SOLID principle compliance increases
- Dependency coupling reductions
- Interface abstraction implementations
```

### Continuous Monitoring Setup
```python
# Establish ongoing optimization monitoring
monitoring_config = {
    "performance_regression_alerts": True,
    "code_quality_degradation_detection": True,
    "automated_optimization_suggestions": True,
    "periodic_refactoring_opportunities": "weekly"
}
```

Execute this comprehensive refactoring and optimization using the full intelligence of Claude.code, Cursor IDE's semantic understanding, and coordinated MCP tool capabilities to achieve measurable performance improvements and superior code quality.