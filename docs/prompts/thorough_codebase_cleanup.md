# Claude.code Thorough Codebase Cleanup - Old Folders and Files Removal

**For use with Claude.code in Cursor IDE with MCP tools**

Please execute an exhaustive cleanup of old folders and files using Claude.code intelligence and MCP tools. **CRITICAL: All operations must include comprehensive backup and validation procedures.**

## Deep Cleanup Analysis Pipeline

### Comprehensive Old File Detection Engine
```python
# Ultra-thorough old file and folder detection using MCP tools
deep_cleanup_engine = {
    "filesystem_archaeology": mcp_filesystem.historical_file_analysis(),
    "git_archaeology": mcp_git.file_evolution_analysis(),
    "dependency_archaeology": mcp_dependencies.historical_dependency_analysis(),
    "build_artifact_archaeology": mcp_build.historical_build_analysis(),
    "configuration_archaeology": mcp_config.historical_config_analysis(),
    "documentation_archaeology": mcp_docs.historical_documentation_analysis(),
    "test_archaeology": mcp_testing.historical_test_analysis(),
    "migration_archaeology": mcp_database.historical_migration_analysis()
}
```

## Exhaustive Old File Categories

### 1. Legacy Code and Obsolete Implementations
```python
def identify_legacy_code_structures():
    """Identify and analyze legacy code structures for removal"""
    
    legacy_analysis = {
        # Old version directories
        "version_directories": mcp_filesystem.find_patterns([
            "**/v1/**", "**/v2/**", "**/old/**", "**/legacy/**",
            "**/deprecated/**", "**/archive/**", "**/backup/**"
        ]),
        
        # Date-based old directories
        "dated_directories": mcp_filesystem.find_dated_directories([
            "**/*2019*/**", "**/*2020*/**", "**/*2021*/**", "**/*2022*/**",
            "**/*old*/**", "**/*archive*/**", "**/*bak*/**"
        ]),
        
        # Language/framework migration leftovers
        "migration_leftovers": mcp_filesystem.find_patterns([
            "**/old-src/**", "**/src-old/**", "**/legacy-api/**",
            "**/deprecated-components/**", "**/unused-modules/**"
        ]),
        
        # Experimental/prototype directories
        "experimental_directories": mcp_filesystem.find_patterns([
            "**/experiment/**", "**/proto/**", "**/test-*/**",
            "**/temp-*/**", "**/try-*/**", "**/draft/**"
        ]),
        
        # Vendor/library old versions
        "old_vendor_directories": mcp_filesystem.find_patterns([
            "**/vendor-old/**", "**/lib-*/**", "**/libs-old/**",
            "**/third-party-old/**", "**/external-old/**"
        ])
    }
    
    # Cross-validate with git history
    git_validated_legacy = mcp_git.validate_legacy_directories(legacy_analysis)
    
    # AI analysis for safety validation
    ai_safety_analysis = cursor_ai.validate_legacy_removal_safety([
        legacy_analysis,
        git_validated_legacy,
        check_active_references=True,
        analyze_dependency_impact=True
    ])
    
    return ai_safety_analysis
```

### 2. Build System and Dependency Artifacts
```python
def identify_build_and_dependency_artifacts():
    """Comprehensive build artifact and dependency cleanup analysis"""
    
    build_artifact_analysis = {
        # Node.js ecosystem artifacts
        "nodejs_artifacts": mcp_filesystem.find_patterns([
            "**/node_modules/**", "**/npm-debug.log*", 
            "**/.npm/**", "**/yarn-error.log", "**/yarn-debug.log*",
            "**/package-lock.json.bak", "**/yarn.lock.bak"
        ]),
        
        # Python ecosystem artifacts
        "python_artifacts": mcp_filesystem.find_patterns([
            "**/__pycache__/**", "**/*.pyc", "**/*.pyo", "**/*.pyd",
            "**/pip-log.txt", "**/pip-delete-this-directory.txt",
            "**/.pytest_cache/**", "**/htmlcov/**", "**/.coverage",
            "**/build/**", "**/dist/**", "**/*.egg-info/**"
        ]),
        
        # Java ecosystem artifacts
        "java_artifacts": mcp_filesystem.find_patterns([
            "**/target/**", "**/build/**", "**/.gradle/**",
            "**/gradle-app.setting", "**/gradlew", "**/gradlew.bat",
            "**/*.class", "**/*.jar.bak", "**/*.war.bak"
        ]),
        
        # .NET ecosystem artifacts
        "dotnet_artifacts": mcp_filesystem.find_patterns([
            "**/bin/**", "**/obj/**", "**/packages/**",
            "**/*.user", "**/TestResults/**", "**/.vs/**"
        ]),
        
        # Generic build artifacts
        "generic_build_artifacts": mcp_filesystem.find_patterns([
            "**/build/**", "**/out/**", "**/output/**",
            "**/release/**", "**/debug/**", "**/tmp/**",
            "**/temp/**", "**/.build/**", "**/.output/**"
        ]),
        
        # Package manager artifacts
        "package_manager_artifacts": mcp_filesystem.find_patterns([
            "**/.bundle/**", "**/vendor/bundle/**", 
            "**/Gemfile.lock.bak", "**/composer.lock.bak",
            "**/go.sum.bak", "**/Cargo.lock.bak"
        ])
    }
    
    # Validate artifacts are safe to remove
    artifact_safety_analysis = cursor_ai.validate_artifact_removal_safety(
        build_artifact_analysis,
        check_gitignore_patterns=True,
        verify_rebuild_capability=True
    )
    
    return artifact_safety_analysis
```

### 3. IDE and Editor Configuration Artifacts
```python
def identify_ide_configuration_artifacts():
    """Identify IDE and editor configuration artifacts for cleanup"""
    
    ide_artifact_analysis = {
        # Visual Studio Code artifacts
        "vscode_artifacts": mcp_filesystem.find_patterns([
            "**/.vscode/settings.json.bak", "**/.vscode/*.log",
            "**/.vscode/extensions.json.bak"
        ]),
        
        # JetBrains IDEs artifacts
        "jetbrains_artifacts": mcp_filesystem.find_patterns([
            "**/.idea/workspace.xml.bak", "**/.idea/tasks.xml.bak",
            "**/.idea/*.iml.bak", "**/.idea/dictionaries/**",
            "**/.idea/libraries/**", "**/.idea/artifacts/**"
        ]),
        
        # Vim/Neovim artifacts
        "vim_artifacts": mcp_filesystem.find_patterns([
            "**/.*.swp", "**/.*.swo", "**/.*.un~",
            "**/Session.vim", "**/.netrwhist"
        ]),
        
        # Emacs artifacts
        "emacs_artifacts": mcp_filesystem.find_patterns([
            "**/*~", "**/#*#", "**/.emacs.desktop",
            "**/.emacs.desktop.lock", "**/auto-save-list/**"
        ]),
        
        # Sublime Text artifacts
        "sublime_artifacts": mcp_filesystem.find_patterns([
            "**/*.sublime-workspace.bak", "**/*.sublime-project.bak"
        ]),
        
        # Generic editor artifacts
        "generic_editor_artifacts": mcp_filesystem.find_patterns([
            "**/*.bak", "**/*.backup", "**/*.old", "**/*.orig",
            "**/*~", "**/.DS_Store", "**/Thumbs.db", "**/desktop.ini"
        ])
    }
    
    # Safety validation for IDE artifacts
    ide_safety_analysis = cursor_ai.validate_ide_artifact_removal(
        ide_artifact_analysis,
        preserve_team_settings=True,
        check_version_control_status=True
    )
    
    return ide_safety_analysis
```

### 4. Log Files and Temporary Data
```python
def identify_log_and_temporary_files():
    """Comprehensive log file and temporary data cleanup"""
    
    log_temp_analysis = {
        # Application log files
        "application_logs": mcp_filesystem.find_patterns([
            "**/*.log", "**/*.log.*", "**/logs/**/*.log",
            "**/log/**/*.log", "**/application.log*",
            "**/error.log*", "**/access.log*", "**/debug.log*"
        ]),
        
        # System and service logs
        "system_logs": mcp_filesystem.find_patterns([
            "**/nohup.out", "**/catalina.out", "**/server.log*",
            "**/gc.log*", "**/heap-dump.*", "**/core.*"
        ]),
        
        # Development and testing logs
        "development_logs": mcp_filesystem.find_patterns([
            "**/test.log*", "**/junit.log*", "**/coverage.log*",
            "**/build.log*", "**/install.log*", "**/update.log*"
        ]),
        
        # Database logs and temporary files
        "database_temp": mcp_filesystem.find_patterns([
            "**/*.db-journal", "**/*.db-wal", "**/*.db-shm",
            "**/mysql-bin.*", "**/mysql-slow.log*", 
            "**/postgresql-*.log", "**/*.mdf.bak", "**/*.ldf.bak"
        ]),
        
        # Cache and temporary directories
        "cache_temp_directories": mcp_filesystem.find_patterns([
            "**/cache/**", "**/tmp/**", "**/temp/**",
            "**/.cache/**", "**/.tmp/**", "**/.temp/**",
            "**/runtime/cache/**", "**/var/cache/**"
        ]),
        
        # Session and state files
        "session_state_files": mcp_filesystem.find_patterns([
            "**/sessions/**", "**/.sessions/**", 
            "**/state/**", "**/.state/**",
            "**/cookies.txt", "**/session.save"
        ])
    }
    
    # Age-based filtering for logs and temporary files
    aged_file_analysis = mcp_filesystem.filter_by_age(
        log_temp_analysis,
        age_threshold_days=30,
        preserve_recent=True
    )
    
    return aged_file_analysis
```

### 5. Documentation and Asset Cleanup
```python
def identify_documentation_and_asset_cleanup():
    """Identify outdated documentation and unused assets"""
    
    doc_asset_analysis = {
        # Outdated documentation
        "outdated_documentation": mcp_docs.find_outdated_docs([
            "**/*-old.md", "**/*-backup.md", "**/*-archive.md",
            "**/docs-old/**", "**/documentation-backup/**",
            "**/README-old.md", "**/CHANGELOG-backup.md"
        ]),
        
        # Duplicate documentation
        "duplicate_documentation": mcp_docs.find_duplicate_content([
            "**/README*.md", "**/INSTALL*.md", "**/SETUP*.md",
            "**/DEPLOYMENT*.md", "**/API*.md"
        ]),
        
        # Unused image and media assets
        "unused_media_assets": mcp_analyzer.find_unreferenced_assets([
            "**/*.png", "**/*.jpg", "**/*.jpeg", "**/*.gif",
            "**/*.svg", "**/*.ico", "**/*.webp", "**/*.mp4",
            "**/*.mov", "**/*.avi", "**/*.pdf", "**/*.doc"
        ]),
        
        # Old presentation and design files
        "old_design_files": mcp_filesystem.find_patterns([
            "**/*.psd", "**/*.ai", "**/*.sketch", "**/*.fig",
            "**/*.xd", "**/*.indd", "**/design-old/**",
            "**/mockups-old/**", "**/wireframes-old/**"
        ]),
        
        # Translation and localization old files
        "old_localization_files": mcp_filesystem.find_patterns([
            "**/i18n-old/**", "**/locales-backup/**", 
            "**/translations-old/**", "**/*.po.bak", "**/*.pot.bak"
        ])
    }
    
    # Content analysis for safety
    content_safety_analysis = cursor_ai.analyze_content_removal_safety(
        doc_asset_analysis,
        check_active_references=True,
        preserve_historical_value=True
    )
    
    return content_safety_analysis
```

## Advanced Cleanup Execution Engine

### Phase 1: Comprehensive Discovery and Analysis
```python
def execute_comprehensive_discovery():
    """Execute thorough discovery of all cleanup targets"""
    
    comprehensive_discovery = {
        "legacy_code_analysis": identify_legacy_code_structures(),
        "build_artifact_analysis": identify_build_and_dependency_artifacts(),
        "ide_artifact_analysis": identify_ide_configuration_artifacts(),
        "log_temp_analysis": identify_log_and_temporary_files(),
        "doc_asset_analysis": identify_documentation_and_asset_cleanup(),
        "custom_pattern_analysis": identify_custom_cleanup_patterns()
    }
    
    # Cross-reference analysis for safety
    cross_reference_analysis = cursor_ai.cross_reference_cleanup_targets([
        comprehensive_discovery,
        active_code_references=mcp_search.find_all_references(),
        build_system_requirements=mcp_build.analyze_requirements(),
        deployment_dependencies=mcp_deployment.analyze_dependencies()
    ])
    
    # Generate comprehensive cleanup strategy
    cleanup_strategy = cursor_ai.generate_cleanup_strategy([
        comprehensive_discovery,
        cross_reference_analysis,
        safety_requirements="maximum",
        backup_strategy="comprehensive",
        rollback_capability="full"
    ])
    
    return cleanup_strategy
```

### Phase 2: Risk Assessment and Safety Validation
```python
def execute_comprehensive_safety_validation(cleanup_strategy):
    """Comprehensive safety validation before cleanup execution"""
    
    safety_validation = {
        # Production impact analysis
        "production_impact": cursor_ai.analyze_production_impact([
            cleanup_strategy.targets,
            production_dependencies=mcp_deployment.get_production_deps(),
            runtime_requirements=mcp_runtime.get_runtime_deps()
        ]),
        
        # Build system impact analysis
        "build_impact": mcp_build.analyze_cleanup_impact([
            cleanup_strategy.targets,
            build_scripts=mcp_build.get_build_scripts(),
            dependency_manifests=mcp_dependencies.get_manifests()
        ]),
        
        # Development workflow impact
        "development_impact": cursor_ai.analyze_development_impact([
            cleanup_strategy.targets,
            development_scripts=mcp_filesystem.find_dev_scripts(),
            team_configurations=mcp_config.get_team_configs()
        ]),
        
        # Historical preservation analysis
        "historical_preservation": cursor_ai.analyze_historical_value([
            cleanup_strategy.targets,
            git_history=mcp_git.get_comprehensive_history(),
            project_milestones=mcp_git.get_project_milestones()
        ])
    }
    
    # Generate safety recommendations
    safety_recommendations = cursor_ai.generate_safety_recommendations(
        safety_validation,
        risk_tolerance="conservative",
        preservation_strategy="comprehensive"
    )
    
    return safety_recommendations
```

### Phase 3: Automated Backup and Cleanup Execution
```python
def execute_safe_comprehensive_cleanup(cleanup_strategy, safety_recommendations):
    """Execute comprehensive cleanup with full safety measures"""
    
    cleanup_executor = ComprehensiveCleanupExecutor(
        backup_strategy="multi_layered",
        validation_strategy="comprehensive",
        rollback_strategy="granular"
    )
    
    # Create comprehensive backup
    backup_manifest = cleanup_executor.create_comprehensive_backup([
        cleanup_strategy.all_targets,
        backup_types=["git_backup", "filesystem_backup", "metadata_backup"],
        compression=True,
        verification=True
    ])
    
    # Execute cleanup in phases
    cleanup_phases = [
        "temporary_files_and_logs",
        "build_artifacts_and_cache",
        "ide_configuration_artifacts", 
        "unused_documentation_and_assets",
        "legacy_code_and_old_directories"
    ]
    
    cleanup_results = {}
    for phase in cleanup_phases:
        phase_targets = cleanup_strategy.get_phase_targets(phase)
        
        # Execute phase cleanup
        phase_result = cleanup_executor.execute_phase_cleanup(
            phase=phase,
            targets=phase_targets,
            safety_checks=safety_recommendations.get_phase_safety(phase)
        )
        
        # Validate phase execution
        phase_validation = cleanup_executor.validate_phase_execution(
            phase_result,
            run_build_tests=True,
            check_application_startup=True,
            verify_core_functionality=True
        )
        
        if phase_validation.success:
            cleanup_executor.commit_phase_cleanup(phase, phase_result)
            cleanup_results[phase] = phase_result
        else:
            cleanup_executor.rollback_phase_cleanup(phase, phase_result)
            cleanup_results[phase] = {
                "status": "failed",
                "reason": phase_validation.failure_reason,
                "rollback_completed": True
            }
    
    return cleanup_results
```

## Specialized Cleanup Operations

### Advanced Pattern-Based Cleanup
```python
def execute_advanced_pattern_cleanup():
    """Execute advanced pattern-based cleanup operations"""
    
    advanced_patterns = {
        # Version control artifacts
        "vcs_artifacts": [
            "**/.git/objects/tmp_*", "**/.git/refs/remotes/*/tmp_*",
            "**/.svn/tmp/**", "**/.hg/store/backup.*"
        ],
        
        # Database migration artifacts
        "migration_artifacts": [
            "**/migrations/*.bak", "**/migrate/versions/*.backup",
            "**/db/migrate/*_backup.rb", "**/database/migrations/*_old.php"
        ],
        
        # Configuration backup patterns
        "config_backups": [
            "**/*.conf.bak", "**/*.config.old", "**/*.ini.backup",
            "**/*.yml.orig", "**/*.yaml.bak", "**/*.json.old"
        ],
        
        # Script and automation artifacts
        "script_artifacts": [
            "**/scripts/*.backup", "**/bin/*.old", "**/tools/*.bak",
            "**/automation/*-old.*", "**/deploy/*.backup"
        ],
        
        # Test data and fixtures artifacts
        "test_artifacts": [
            "**/test-data-old/**", "**/fixtures-backup/**",
            "**/mocks/*.old", "**/stubs/*.backup", "**/samples/*.bak"
        ]
    }
    
    # Execute pattern-based cleanup with validation
    pattern_cleanup_results = {}
    for pattern_category, patterns in advanced_patterns.items():
        category_targets = mcp_filesystem.find_patterns(patterns)
        
        # Validate pattern targets
        validated_targets = cursor_ai.validate_pattern_targets(
            category_targets,
            pattern_category=pattern_category,
            safety_checks=True
        )
        
        # Execute cleanup for validated targets
        if validated_targets.safe_to_remove:
            cleanup_result = execute_pattern_cleanup(
                pattern_category,
                validated_targets.targets
            )
            pattern_cleanup_results[pattern_category] = cleanup_result
    
    return pattern_cleanup_results
```

### Large File and Directory Cleanup
```python
def execute_large_file_cleanup():
    """Identify and clean up large files and directories"""
    
    large_file_analysis = {
        # Find large files (>100MB)
        "large_files": mcp_filesystem.find_large_files(
            size_threshold="100MB",
            exclude_patterns=[".git/**", "node_modules/**"]
        ),
        
        # Find large directories (>1GB)
        "large_directories": mcp_filesystem.find_large_directories(
            size_threshold="1GB",
            analyze_contents=True
        ),
        
        # Find old large files (>50MB and >6 months old)
        "old_large_files": mcp_filesystem.find_old_large_files(
            size_threshold="50MB",
            age_threshold_months=6
        )
    }
    
    # Analyze large file usage and necessity
    large_file_necessity = cursor_ai.analyze_large_file_necessity([
        large_file_analysis,
        check_git_lfs_candidates=True,
        analyze_compression_opportunities=True,
        check_active_usage=True
    ])
    
    # Execute large file cleanup strategy
    large_file_cleanup = execute_large_file_optimization(
        large_file_necessity,
        strategies=["compression", "git_lfs_migration", "removal", "archival"]
    )
    
    return large_file_cleanup
```

## Comprehensive Cleanup Validation

### Post-Cleanup System Validation
```python
def execute_post_cleanup_validation():
    """Comprehensive system validation after cleanup"""
    
    validation_suite = {
        # Build system validation
        "build_validation": mcp_build.comprehensive_build_test([
            "clean_build",
            "dependency_resolution",
            "asset_compilation",
            "test_execution"
        ]),
        
        # Application functionality validation
        "application_validation": mcp_testing.comprehensive_functionality_test([
            "application_startup",
            "core_feature_testing",
            "api_endpoint_testing",
            "database_connectivity"
        ]),
        
        # Performance impact validation
        "performance_validation": mcp_profiler.performance_impact_analysis([
            "startup_time_comparison",
            "memory_usage_comparison",
            "build_time_comparison",
            "test_execution_time_comparison"
        ]),
        
        # Security posture validation
        "security_validation": mcp_security.security_impact_analysis([
            "vulnerability_scan",
            "dependency_security_check",
            "configuration_security_review",
            "access_control_validation"
        ])
    }
    
    # Generate comprehensive validation report
    validation_report = cursor_ai.generate_validation_report(
        validation_suite,
        cleanup_impact_analysis=True,
        recommendations=True
    )
    
    return validation_report
```

## Cleanup Results and Monitoring

### Comprehensive Cleanup Report Generation
```markdown
# Comprehensive Codebase Cleanup Execution Report

## Executive Summary
- **Total Files Analyzed:** [count]
- **Files/Directories Removed:** [count]
- **Space Reclaimed:** [size in GB/MB]
- **Cleanup Categories Processed:** [count]
- **Safety Validations Passed:** [count/total]
- **Post-Cleanup System Status:** [HEALTHY/ISSUES]

## Detailed Cleanup Results

### Legacy Code and Directory Cleanup
- **Legacy Directories Removed:** [count] ([size])
  - Version directories: [count] ([size])
  - Experimental directories: [count] ([size])
  - Archive directories: [count] ([size])
- **Obsolete Code Files Removed:** [count] ([size])

### Build Artifact and Dependency Cleanup
- **Build Artifacts Removed:** [count] ([size])
  - Node.js artifacts: [count] ([size])
  - Python artifacts: [count] ([size])
  - Java artifacts: [count] ([size])
  - Generic build artifacts: [count] ([size])
- **Dependency Cache Cleanup:** [count] ([size])

### IDE and Configuration Cleanup
- **IDE Configuration Artifacts:** [count] ([size])
- **Editor Backup Files:** [count] ([size])
- **System Generated Files:** [count] ([size])

### Log and Temporary File Cleanup
- **Log Files Removed:** [count] ([size])
  - Application logs: [count] ([size])
  - System logs: [count] ([size])
  - Development logs: [count] ([size])
- **Temporary Files and Directories:** [count] ([size])
- **Cache Directories Cleaned:** [count] ([size])

### Documentation and Asset Cleanup
- **Outdated Documentation:** [count] ([size])
- **Unused Media Assets:** [count] ([size])
- **Duplicate Content Removed:** [count] ([size])

### Large File Optimization
- **Large Files Processed:** [count] ([size])
- **Compression Applied:** [count] ([space saved])
- **Git LFS Migration:** [count] ([size])

## Safety and Validation Results
- **Pre-Cleanup Backup Created:** ✅ [backup size and location]
- **Build System Validation:** ✅ PASSED
- **Application Functionality Test:** ✅ PASSED
- **Performance Impact Analysis:** ✅ MINIMAL IMPACT
- **Security Posture Check:** ✅ MAINTAINED

## Performance Impact Analysis
- **Repository Size Reduction:** [percentage] ([before] → [after])
- **Build Time Impact:** [improvement/regression percentage]
- **Git Operations Speed:** [improvement percentage]
- **IDE Loading Time:** [improvement percentage]

## Continuous Monitoring Setup
- **Automated Cleanup Monitoring:** ✅ ACTIVE
- **Large File Detection:** ✅ ACTIVE
- **Build Artifact Monitoring:** ✅ ACTIVE
- **Documentation Freshness Monitoring:** ✅ ACTIVE

## Recommendations for Ongoing Maintenance
1. **Automated Cleanup Schedule:** [recommendations]
2. **GitIgnore Updates:** [patterns to add]
3. **Build System Optimization:** [recommendations]
4. **Documentation Maintenance:** [recommendations]
5. **Asset Management:** [recommendations]
```

### Continuous Cleanup Automation Setup
```python
# Establish ongoing automated cleanup
continuous_cleanup_automation = {
    "daily_cleanup": {
        "temporary_files": True,
        "log_file_rotation": True,
        "cache_cleanup": True
    },
    "weekly_cleanup": {
        "build_artifact_cleanup": True,
        "ide_artifact_cleanup": True,
        "large_file_monitoring": True
    },
    "monthly_cleanup": {
        "comprehensive_analysis": True,
        "documentation_freshness_check": True,
        "dependency_cleanup": True
    },
    "safety_monitoring": {
        "continuous_backup": True,
        "cleanup_impact_monitoring": True,
        "rollback_capability": True
    }
}
```

Execute this thorough codebase cleanup using the full capabilities of Claude.code intelligence, Cursor IDE's understanding, and coordinated MCP tool execution to achieve optimal repository cleanliness and organization while maintaining complete safety and functionality.