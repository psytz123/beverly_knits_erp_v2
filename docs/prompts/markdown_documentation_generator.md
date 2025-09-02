# Claude.code Complete Documentation Generator Prompt

**For use with Claude.code in Cursor IDE with MCP tools**

Please execute automated comprehensive documentation generation using Claude.code intelligence and MCP tools to create a definitive single markdown documentation file:

## MCP Tool Documentation Pipeline

### Automated Content Discovery & Generation
```python
# Comprehensive documentation generation using MCP tools
documentation_engine = {
    "content_discovery": mcp_filesystem.documentation_scan(),
    "code_analysis": cursor_ai.semantic_documentation(),
    "api_extraction": mcp_api_scanner.endpoint_documentation(),
    "architecture_mapping": mcp_analyzer.architecture_visualization(),
    "dependency_documentation": mcp_dependencies.dependency_documentation(),
    "configuration_mapping": mcp_config.configuration_documentation(),
    "deployment_analysis": mcp_deployment.deployment_documentation()
}
```

## Intelligent Documentation Structure Generation

### AI-Powered Content Organization
```python
def generate_documentation_structure():
    """Use Claude.code to intelligently organize documentation"""
    
    # Analyze codebase to determine optimal documentation structure
    structure_analysis = cursor_ai.analyze_documentation_needs([
        "project_complexity",
        "user_personas", 
        "feature_complexity",
        "technical_depth_required",
        "onboarding_requirements",
        "operational_complexity"
    ])
    
    # Generate adaptive documentation structure
    documentation_structure = cursor_ai.create_adaptive_structure(structure_analysis)
    return documentation_structure
```

## Automated Content Generation Engine

### 1. Executive Summary Auto-Generation
```python
def generate_executive_summary():
    """Auto-generate executive summary using codebase analysis"""
    
    summary_data = {
        "project_purpose": mcp_analyzer.infer_project_purpose(),
        "technology_stack": mcp_dependencies.stack_analysis(),
        "key_features": mcp_feature_analyzer.feature_catalog(),
        "architecture_type": cursor_ai.classify_architecture(),
        "performance_characteristics": mcp_profiler.performance_summary(),
        "maturity_assessment": cursor_ai.assess_project_maturity()
    }
    
    executive_summary = cursor_ai.generate_executive_summary(summary_data)
    return executive_summary
```

### 2. Quick Start Guide Auto-Generation
```python
def generate_quick_start_guide():
    """Auto-generate setup and quick start instructions"""
    
    setup_analysis = {
        "dependencies": mcp_dependencies.installation_requirements(),
        "configuration": mcp_config.required_configurations(),
        "build_process": mcp_build.build_instructions(),
        "environment_setup": mcp_env.environment_requirements(),
        "verification_steps": mcp_testing.verification_procedures()
    }
    
    quick_start_guide = cursor_ai.generate_setup_guide(setup_analysis)
    return quick_start_guide
```

### 3. Architecture Documentation Auto-Generation
```python
def generate_architecture_documentation():
    """Auto-generate comprehensive architecture documentation"""
    
    architecture_analysis = {
        "system_topology": mcp_analyzer.system_mapping(),
        "component_relationships": mcp_dependencies.component_analysis(),
        "data_flow_patterns": mcp_data_analyzer.flow_analysis(),
        "technology_decisions": cursor_ai.infer_technology_decisions(),
        "scaling_patterns": mcp_analyzer.scalability_analysis()
    }
    
    # Generate visual diagrams using MCP tools
    diagrams = {
        "system_architecture": mcp_diagram.generate_system_diagram(),
        "component_interactions": mcp_diagram.generate_component_diagram(),
        "data_flow": mcp_diagram.generate_data_flow_diagram(),
        "deployment_topology": mcp_diagram.generate_deployment_diagram()
    }
    
    architecture_docs = cursor_ai.generate_architecture_documentation(
        architecture_analysis, 
        diagrams
    )
    return architecture_docs
```

### 4. API Documentation Auto-Generation
```python
def generate_api_documentation():
    """Auto-generate comprehensive API documentation"""
    
    api_analysis = {
        "endpoint_discovery": mcp_api_scanner.discover_endpoints(),
        "schema_extraction": mcp_api_scanner.extract_schemas(),
        "authentication_patterns": mcp_security.auth_analysis(),
        "error_response_patterns": mcp_api_scanner.error_patterns(),
        "rate_limiting": mcp_api_scanner.rate_limit_analysis()
    }
    
    # Generate interactive API documentation
    api_documentation = cursor_ai.generate_api_docs([
        "endpoint_descriptions",
        "request_response_examples", 
        "authentication_flows",
        "error_handling_examples",
        "code_samples_multiple_languages"
    ])
    
    return api_documentation
```

## Claude.code Intelligence Integration

### Semantic Code Documentation
```python
def generate_code_documentation():
    """Use Claude.code semantic understanding for code documentation"""
    
    code_documentation = cursor_ai.semantic_code_analysis([
        "function_purpose_inference",
        "parameter_description_generation",
        "return_value_documentation",
        "side_effect_identification",
        "usage_example_generation",
        "edge_case_documentation"
    ])
    
    return code_documentation
```

### Intelligent Troubleshooting Guide Generation
```python
def generate_troubleshooting_guide():
    """Auto-generate troubleshooting documentation"""
    
    troubleshooting_analysis = {
        "common_error_patterns": mcp_log_analyzer.error_pattern_analysis(),
        "failure_modes": mcp_analyzer.failure_mode_analysis(),
        "dependency_issues": mcp_dependencies.common_issues(),
        "configuration_problems": mcp_config.common_misconfigurations(),
        "performance_issues": mcp_profiler.common_bottlenecks()
    }
    
    troubleshooting_guide = cursor_ai.generate_troubleshooting_docs(
        troubleshooting_analysis,
        include_solutions=True,
        generate_diagnostic_steps=True
    )
    
    return troubleshooting_guide
```

## Automated Documentation Assembly

### Comprehensive Documentation Builder
```python
def build_comprehensive_documentation():
    """Assemble complete documentation using all generated content"""
    
    documentation_sections = {
        "executive_summary": generate_executive_summary(),
        "quick_start": generate_quick_start_guide(),
        "architecture": generate_architecture_documentation(),
        "api_documentation": generate_api_documentation(),
        "database_schema": generate_database_documentation(),
        "feature_documentation": generate_feature_documentation(),
        "development_guide": generate_development_documentation(),
        "deployment_operations": generate_deployment_documentation(),
        "security_implementation": generate_security_documentation(),
        "performance_optimization": generate_performance_documentation(),
        "testing_strategy": generate_testing_documentation(),
        "troubleshooting": generate_troubleshooting_guide(),
        "technical_debt_roadmap": generate_roadmap_documentation()
    }
    
    # Assemble into comprehensive markdown
    comprehensive_docs = cursor_ai.assemble_documentation(
        documentation_sections,
        generate_toc=True,
        cross_reference_links=True,
        include_navigation=True
    )
    
    return comprehensive_docs
```

### Interactive Documentation Features
```python
def enhance_documentation_interactivity():
    """Add interactive elements to documentation"""
    
    interactive_enhancements = {
        "collapsible_sections": add_collapsible_details(),
        "code_syntax_highlighting": apply_syntax_highlighting(),
        "mermaid_diagrams": generate_mermaid_diagrams(),
        "status_badges": generate_status_badges(),
        "cross_references": create_internal_links(),
        "search_functionality": add_search_anchors()
    }
    
    return interactive_enhancements
```

## MCP Tool Coordination for Documentation

### Multi-Source Documentation Aggregation
```python
class DocumentationOrchestrator:
    def __init__(self):
        self.mcp_tools = initialize_documentation_mcp_tools()
        self.cursor_ai = initialize_cursor_intelligence()
        
    async def generate_complete_documentation(self):
        """Orchestrate complete documentation generation"""
        
        # Parallel content generation
        content_generation_tasks = await asyncio.gather([
            self.generate_architectural_content(),
            self.generate_api_content(),
            self.generate_setup_content(),
            self.generate_operational_content(),
            self.generate_troubleshooting_content(),
            self.generate_development_content()
        ])
        
        # Intelligent content synthesis
        synthesized_content = self.cursor_ai.synthesize_documentation(
            content_generation_tasks
        )
        
        # Generate final markdown with navigation
        final_documentation = self.assemble_final_documentation(synthesized_content)
        
        return final_documentation
```

### Real-Time Documentation Updates
```python
def setup_documentation_automation():
    """Establish automated documentation updates"""
    
    documentation_automation = {
        "code_change_documentation_updates": True,
        "api_change_documentation_sync": True,
        "dependency_update_documentation": True,
        "configuration_change_documentation": True,
        "performance_metric_documentation_updates": True
    }
    
    return documentation_automation
```

## Documentation Quality Assurance

### Automated Documentation Validation
```python
def validate_documentation_quality():
    """Validate generated documentation quality"""
    
    validation_checks = {
        "completeness_check": verify_all_sections_complete(),
        "accuracy_validation": cross_check_with_codebase(),
        "link_validation": verify_all_internal_links(),
        "example_validation": test_all_code_examples(),
        "formatting_consistency": check_markdown_formatting(),
        "accessibility_compliance": validate_accessibility_standards()
    }
    
    quality_report = cursor_ai.generate_quality_assessment(validation_checks)
    return quality_report
```

## Output Specification

### Comprehensive Markdown Documentation
```markdown
# [PROJECT_NAME] - Complete Documentation

## Table of Contents
[Auto-generated TOC with intelligent categorization]

## Executive Summary
[AI-generated project overview with key insights]

## Quick Start Guide
[Auto-generated setup instructions with validation steps]

## Architecture Overview
[AI-generated architecture documentation with diagrams]

## API Documentation
[Auto-extracted and documented API endpoints]

## Database Schema
[Auto-generated database documentation]

## Feature Documentation
[AI-generated feature descriptions and workflows]

## Development Guide
[Auto-generated development environment and practices]

## Deployment & Operations
[Auto-generated deployment and operational procedures]

## Security Implementation
[Auto-documented security features and practices]

## Performance & Optimization
[Auto-generated performance characteristics and optimization]

## Testing Strategy
[Auto-documented testing approaches and coverage]

## Troubleshooting
[AI-generated troubleshooting guide with solutions]

## Technical Debt & Roadmap
[Auto-generated improvement recommendations and roadmap]

## Appendices
[Auto-generated reference materials and resources]
```

### Documentation Maintenance System
```python
# Establish ongoing documentation maintenance
documentation_maintenance = {
    "automated_content_updates": True,
    "accuracy_monitoring": True,
    "completeness_tracking": True,
    "user_feedback_integration": True,
    "documentation_analytics": True
}
```

Execute this comprehensive documentation generation using the full intelligence of Claude.code, Cursor IDE's semantic understanding, and coordinated MCP tool execution to create definitive, accurate, and maintainable project documentation.