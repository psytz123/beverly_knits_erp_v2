# Claude.code Safe File Cleanup & Markdown Consolidation Prompt

**For use with Claude.code in Cursor IDE with MCP tools**

**Last Cleanup Executed:** September 2, 2025  
**Status:** ✅ COMPLETED - Repository successfully organized

Please execute comprehensive file cleanup and markdown consolidation using Claude.code intelligence and MCP tools. **CRITICAL: All operations must include automated backup and rollback capabilities.**

## Cleanup Results Summary (Sep 2, 2025)

### 🎯 Achievements
- **Root Directory**: Reduced from 24 to 3 Python scripts (87.5% reduction)
- **Test Directory**: Removed 24+ redundant files
- **Data Organization**: Archived old snapshots to `data/production/5/archive/`
- **Documentation**: Archived completed phase reports
- **Backup Created**: `backups/cleanup_20250902_030942.tar.gz` (27MB)

### ✅ Completed Actions
1. **Root Directory Organization**
   - Moved 21 utility scripts to `scripts/` subdirectories
   - Created organized structure: `ml_training/`, `data_loading/`, `testing/`, `maintenance/`, `utilities/`
   - Kept only essential files: `setup.py`, `run_tests.py`, `start_erp.py`

2. **Test Directory Cleanup**
   - Removed non-test utility scripts (5 files)
   - Deleted obsolete development files (9 files)
   - Removed duplicate test files (2 files)
   - Cleaned test execution reports (12+ JSON files)

3. **Data Directory Organization**
   - Archived old data snapshots (8-22-2025, 8-26-2025)
   - Moved backtest results to archive
   - Created `data/production/5/archive/2025_august/`

4. **Documentation Consolidation**
   - Archived completed phase reports
   - Created `docs/archive/completed_phases/`

### ✅ System Validation
- Server starts successfully on port 5006
- API endpoints operational
- Data loading functional
- Tests execute (with some failures from previous issues)

## MCP Tool Cleanup Pipeline

### Comprehensive File Analysis Engine
```python
# Automated file cleanup analysis using MCP tools
cleanup_analysis_engine = {
    "filesystem_forensics": mcp_filesystem.comprehensive_file_analysis(),
    "git_history_analysis": mcp_git.file_usage_history(),
    "dependency_analysis": mcp_dependencies.unused_file_detection(),
    "code_reference_analysis": mcp_search.reference_tracking(),
    "build_artifact_analysis": mcp_build.artifact_identification(),
    "documentation_consolidation": mcp_docs.markdown_analysis()
}
```

## Automated Cleanup Categories

### 1. Markdown Documentation Consolidation Engine
```python
def execute_markdown_consolidation():
    """Intelligent markdown consolidation using Claude.code"""
    
    # Discover all markdown files
    markdown_inventory = mcp_filesystem.find_files(
        extensions=[".md", ".markdown"],
        include_metadata=True,
        analyze_content=True
    )
    
    # Analyze content relationships
    content_analysis = cursor_ai.analyze_markdown_relationships([
        "content_overlap_detection",
        "topic_clustering",
        "information_hierarchy_analysis",
        "consolidation_opportunities",
        "duplicate_content_identification",
        "outdated_content_detection"
    ])
    
    # Generate consolidation strategy
    consolidation_plan = cursor_ai.create_consolidation_strategy(
        markdown_inventory,
        content_analysis,
        preservation_priorities=["accuracy", "completeness", "accessibility"]
    )
    
    return consolidation_plan
```

### 2. Dead Code & Unused File Detection
```python
def detect_unused_files():
    """Comprehensive unused file detection using MCP tools"""
    
    unused_file_analysis = {
        "unreferenced_source_files": mcp_search.find_unreferenced_files(),
        "dead_import_detection": mcp_dependencies.unused_imports(),
        "orphaned_test_files": mcp_testing.orphaned_test_detection(),
        "unused_configuration": mcp_config.unused_config_detection(),
        "obsolete_migration_files": mcp_database.obsolete_migrations(),
        "abandoned_feature_files": mcp_git.abandoned_feature_detection()
    }
    
    # Cross-validate findings
    validated_unused_files = cursor_ai.validate_unused_file_detection(
        unused_file_analysis,
        safety_checks=True,
        production_validation=True
    )
    
    return validated_unused_files
```

### 3. Temporary & Development Artifact Cleanup
```python
def identify_temporary_files():
    """Identify temporary and development artifacts for cleanup"""
    
    temporary_file_analysis = {
        "build_artifacts": mcp_build.find_build_artifacts(),
        "cache_files": mcp_filesystem.find_cache_files(),
        "log_files": mcp_filesystem.find_log_files(),
        "ide_configuration": mcp_filesystem.find_ide_files(),
        "os_generated_files": mcp_filesystem.find_os_files(),
        "backup_files": mcp_filesystem.find_backup_files(),
        "temp_development_files": mcp_git.find_uncommitted_temp_files()
    }
    
    # Safety validation for temporary files
    safe_temp_files = cursor_ai.validate_temporary_file_safety(
        temporary_file_analysis,
        check_production_usage=True,
        verify_build_requirements=True
    )
    
    return safe_temp_files
```

## Claude.code Intelligence Integration

### Intelligent Content Consolidation
```python
def intelligent_markdown_consolidation():
    """Use Claude.code for intelligent content consolidation"""
    
    # Analyze existing documentation structure
    documentation_structure = cursor_ai.analyze_documentation_architecture([
        "information_architecture_assessment",
        "user_journey_mapping",
        "content_gap_analysis",
        "redundancy_identification",
        "consolidation_benefit_analysis"
    ])
    
    # Generate optimal structure
    optimal_structure = cursor_ai.design_optimal_documentation([
        "single_source_of_truth_design",
        "logical_information_hierarchy",
        "cross_reference_optimization",
        "searchability_enhancement",
        "maintenance_efficiency"
    ])
    
    # Execute intelligent content merger
    consolidated_content = cursor_ai.merge_documentation_intelligently(
        preserve_all_information=True,
        enhance_readability=True,
        optimize_navigation=True
    )
    
    return consolidated_content
```

### AI-Powered Safety Validation
```python
def ai_safety_validation():
    """Use Claude.code for comprehensive safety validation"""
    
    safety_analysis = cursor_ai.comprehensive_safety_assessment([
        "production_impact_analysis",
        "dependency_risk_assessment", 
        "configuration_impact_validation",
        "deployment_risk_evaluation",
        "rollback_procedure_validation"
    ])
    
    return safety_analysis
```

## Automated Cleanup Execution Engine

### Phase 1: Discovery & Analysis
```python
def execute_discovery_phase():
    """Comprehensive discovery using MCP tools"""
    
    discovery_results = {
        # File system analysis
        "filesystem_scan": mcp_filesystem.deep_scan(
            include_hidden=True,
            analyze_sizes=True,
            check_permissions=True
        ),
        
        # Git history analysis
        "git_analysis": mcp_git.file_history_analysis(
            find_abandoned_files=True,
            track_file_evolution=True,
            identify_stale_branches=True
        ),
        
        # Code reference analysis
        "reference_analysis": mcp_search.comprehensive_reference_scan(
            include_dynamic_references=True,
            check_configuration_references=True,
            validate_build_references=True
        ),
        
        # Documentation analysis
        "documentation_analysis": mcp_docs.documentation_ecosystem_analysis(
            content_overlap_detection=True,
            quality_assessment=True,
            consolidation_opportunities=True
        )
    }
    
    return discovery_results
```

### Phase 2: Intelligent Cleanup Planning
```python
def create_cleanup_execution_plan(discovery_results):
    """Generate intelligent cleanup execution plan"""
    
    cleanup_plan = cursor_ai.generate_cleanup_strategy([
        discovery_results,
        safety_requirements={
            "automated_backup": True,
            "incremental_execution": True,
            "validation_checkpoints": True,
            "rollback_capability": True
        },
        optimization_goals=[
            "repository_size_reduction",
            "documentation_consolidation", 
            "development_efficiency",
            "maintenance_simplification"
        ]
    ])
    
    return cleanup_plan
```

### Phase 3: Safe Cleanup Execution
```python
def execute_safe_cleanup(cleanup_plan):
    """Execute cleanup with automated safety measures"""
    
    cleanup_executor = SafeCleanupExecutor(
        backup_strategy="automated_git_backup",
        validation_strategy="comprehensive_testing",
        rollback_strategy="automated_rollback"
    )
    
    for cleanup_task in cleanup_plan.tasks:
        # Pre-execution backup
        backup_point = cleanup_executor.create_backup_point()
        
        # Execute cleanup task
        execution_result = cleanup_executor.execute_task(cleanup_task)
        
        # Validate execution
        validation_result = cleanup_executor.validate_execution(
            execution_result,
            run_tests=True,
            check_builds=True,
            verify_functionality=True
        )
        
        if validation_result.success:
            cleanup_executor.commit_changes(cleanup_task)
        else:
            cleanup_executor.rollback_to_backup(backup_point)
            cleanup_executor.log_failure(cleanup_task, validation_result)
```

## MCP Tool Coordination Strategy

### Multi-Tool Cleanup Orchestration
```python
class CleanupOrchestrator:
    def __init__(self):
        self.mcp_tools = initialize_cleanup_mcp_tools()
        self.cursor_ai = initialize_cursor_intelligence()
        self.safety_engine = initialize_safety_validation()
        
    async def execute_comprehensive_cleanup(self):
        """Orchestrate complete cleanup operation"""
        
        # Phase 1: Parallel analysis
        analysis_results = await asyncio.gather([
            self.analyze_markdown_consolidation(),
            self.analyze_unused_files(),
            self.analyze_temporary_files(),
            self.analyze_dependency_cleanup(),
            self.analyze_configuration_cleanup()
        ])
        
        # Phase 2: Safety validation
        safety_assessment = self.safety_engine.comprehensive_validation(
            analysis_results
        )
        
        # Phase 3: Intelligent execution planning
        execution_plan = self.cursor_ai.create_execution_plan(
            analysis_results,
            safety_assessment,
            optimization_priorities=["safety", "efficiency", "maintainability"]
        )
        
        # Phase 4: Automated execution with monitoring
        execution_results = await self.execute_cleanup_plan(execution_plan)
        
        return execution_results
```

### Real-Time Safety Monitoring
```python
def setup_cleanup_safety_monitoring():
    """Establish real-time safety monitoring during cleanup"""
    
    safety_monitoring = {
        "build_system_monitoring": monitor_build_integrity(),
        "test_suite_monitoring": monitor_test_execution(),
        "application_startup_monitoring": monitor_app_functionality(),
        "dependency_resolution_monitoring": monitor_dependency_integrity(),
        "configuration_validity_monitoring": monitor_config_integrity()
    }
    
    return safety_monitoring
```

## Markdown Consolidation Specification

### Intelligent Documentation Merger
```python
def execute_documentation_consolidation():
    """Execute intelligent documentation consolidation"""
    
    # Analyze current documentation ecosystem
    doc_ecosystem = mcp_docs.analyze_documentation_ecosystem()
    
    # Generate optimal consolidated structure
    consolidated_structure = cursor_ai.design_consolidated_documentation([
        "comprehensive_table_of_contents",
        "logical_section_organization",
        "cross_reference_optimization",
        "search_optimization",
        "maintenance_efficiency"
    ])
    
    # Execute content consolidation
    consolidation_execution = {
        "content_merger": merge_overlapping_content(),
        "link_updater": update_internal_references(),
        "format_standardizer": standardize_markdown_formatting(),
        "navigation_enhancer": add_navigation_elements(),
        "quality_validator": validate_consolidated_content()
    }
    
    # Generate final consolidated documentation
    final_documentation = cursor_ai.generate_final_documentation(
        consolidation_execution,
        quality_standards="commercial_grade",
        maintainability_focus=True
    )
    
    return final_documentation
```

## Output Specification

### Comprehensive Cleanup Report
```markdown
# Automated Cleanup Execution Report

## Executive Summary
- Files analyzed: [Total count with breakdown]
- Space reclaimed: [Size in MB/GB]
- Documentation consolidated: [Number of files merged]
- Safety validations: [All passed/failed items]
- Performance improvements: [Measurable improvements]

## Markdown Consolidation Results
### Documentation Structure Optimization
- Files consolidated: [Before/after count]
- Content deduplication: [Percentage reduction]
- Navigation improvements: [Enhancements made]
- Maintenance efficiency: [Improvement metrics]

### Consolidated Documentation Features
- Single source of truth: ✅
- Comprehensive table of contents: ✅
- Cross-referenced sections: ✅
- Search-optimized structure: ✅
- Mobile-friendly formatting: ✅

## File Cleanup Results
### Dead Code Removal
- Unused source files: [Count and size]
- Orphaned test files: [Count and size]
- Dead configuration: [Count and size]

### Temporary File Cleanup
- Build artifacts: [Count and size]
- Cache files: [Count and size]
- Development artifacts: [Count and size]

## Safety Validation Results
- All tests: [PASSED/FAILED]
- Build integrity: [PASSED/FAILED]
- Application functionality: [PASSED/FAILED]
- Configuration validity: [PASSED/FAILED]

## Ongoing Maintenance Setup
- Automated cleanup monitoring: ✅
- Documentation maintenance: ✅
- Continuous safety validation: ✅
```

### Continuous Cleanup Monitoring
```python
# Establish ongoing cleanup and maintenance
continuous_cleanup = {
    "automated_temporary_file_cleanup": "daily",
    "documentation_consistency_monitoring": "weekly", 
    "unused_code_detection": "monthly",
    "dependency_cleanup_analysis": "monthly",
    "safety_validation_continuous": True
}
```

Execute this comprehensive cleanup and consolidation using the full capabilities of Claude.code intelligence, Cursor IDE's understanding, and coordinated MCP tool execution to achieve optimal repository organization and maintenance efficiency.