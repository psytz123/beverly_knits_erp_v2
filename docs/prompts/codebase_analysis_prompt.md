# Claude.code Complete Codebase Analysis & Understanding Prompt

**For use with Claude.code in Cursor IDE with MCP tools**

Please execute a comprehensive automated codebase analysis using Claude.code's semantic understanding and available MCP tools to generate complete system knowledge:

## MCP Tool Orchestration for Deep Analysis

### Comprehensive Discovery Pipeline
```python
# Orchestrated codebase analysis using all available MCP tools
codebase_intelligence = {
    "structural_analysis": mcp_filesystem.deep_structure_analysis(),
    "semantic_analysis": cursor_ai.semantic_understanding(),
    "dependency_mapping": mcp_dependencies.full_relationship_map(),
    "execution_flow": mcp_tracer.execution_path_analysis(),
    "data_flow": mcp_data_analyzer.flow_mapping(),
    "api_discovery": mcp_api_scanner.endpoint_catalog(),
    "security_context": mcp_security.context_analysis()
}
```

## Automated Analysis Categories

### 1. Architectural Intelligence Gathering
```python
def analyze_system_architecture():
    """Deep architectural analysis using Claude.code intelligence"""
    
    # System topology discovery
    architecture_map = {
        "component_hierarchy": mcp_analyzer.component_tree(),
        "service_boundaries": identify_service_boundaries(),
        "communication_patterns": map_inter_component_communication(),
        "data_persistence_layers": analyze_data_storage_patterns(),
        "external_integrations": catalog_external_dependencies(),
        "deployment_topology": understand_deployment_structure()
    }
    
    # Generate architectural insights
    architectural_insights = cursor_ai.synthesize_architecture(architecture_map)
    return architectural_insights
```

### 2. Code Structure Deep Dive
```python
def execute_structural_forensics():
    """Comprehensive code structure analysis"""
    
    structural_analysis = {
        "directory_organization": mcp_filesystem.organization_analysis(),
        "module_relationships": mcp_dependencies.dependency_graph(),
        "class_hierarchies": analyze_inheritance_patterns(),
        "interface_contracts": map_interface_definitions(),
        "configuration_management": discover_config_patterns(),
        "entry_point_mapping": trace_application_entry_points()
    }
    
    # Cross-reference patterns across the codebase
    pattern_analysis = cursor_ai.identify_patterns(structural_analysis)
    return pattern_analysis
```

### 3. Functional Behavior Analysis
```python
def analyze_functional_behavior():
    """Comprehensive functional analysis using execution tracing"""
    
    functional_mapping = {
        "feature_catalog": mcp_feature_analyzer.discover_features(),
        "user_workflow_tracing": trace_user_interactions(),
        "business_logic_mapping": map_business_rules(),
        "api_behavior_analysis": analyze_endpoint_behaviors(),
        "data_transformation_flows": map_data_processing_pipelines(),
        "event_handling_patterns": discover_event_systems()
    }
    
    # Generate comprehensive feature documentation
    feature_documentation = cursor_ai.generate_feature_docs(functional_mapping)
    return feature_documentation
```

## Claude.code Intelligence Integration

### Semantic Code Understanding
```python
def semantic_codebase_analysis():
    """Leverage Cursor AI for deep semantic understanding"""
    
    semantic_insights = cursor_ai.deep_analysis([
        "intent_recognition",      # What code is trying to accomplish
        "pattern_identification",  # Design patterns and architectural patterns
        "complexity_assessment",   # Cognitive complexity analysis
        "maintainability_scoring", # Code maintainability metrics
        "technical_debt_mapping",  # Areas needing improvement
        "best_practice_compliance" # Adherence to coding standards
    ])
    
    return semantic_insights
```

### Intelligent Documentation Generation
```python
def generate_intelligent_documentation():
    """Auto-generate comprehensive documentation using AI understanding"""
    
    documentation_engine = cursor_ai.documentation_generator([
        "api_documentation",       # Auto-generate API docs from code
        "architecture_diagrams",   # Create visual architecture representations
        "data_flow_diagrams",     # Visualize data movement
        "component_interactions", # Show component relationships
        "deployment_guides",      # Generate deployment documentation
        "troubleshooting_guides"  # Create debugging and issue resolution guides
    ])
    
    return documentation_engine
```

## Comprehensive Analysis Execution

### Phase 1: Discovery & Mapping
```python
def execute_discovery_phase():
    """Comprehensive codebase discovery using MCP tools"""
    
    discovery_results = {
        # File system intelligence
        "filesystem_topology": mcp_filesystem.comprehensive_scan(),
        
        # Git history analysis
        "evolution_patterns": mcp_git.evolution_analysis(),
        
        # Dependency intelligence
        "dependency_ecosystem": mcp_dependencies.ecosystem_analysis(),
        
        # Security landscape
        "security_posture": mcp_security.posture_assessment(),
        
        # Performance characteristics
        "performance_profile": mcp_profiler.baseline_characteristics(),
        
        # Testing landscape
        "test_coverage_analysis": mcp_testing.comprehensive_coverage()
    }
    
    return discovery_results
```

### Phase 2: Intelligence Synthesis
```python
def synthesize_codebase_intelligence(discovery_results):
    """Synthesize insights from all analysis tools"""
    
    intelligence_synthesis = cursor_ai.synthesize_insights([
        discovery_results,
        semantic_analysis_results,
        architectural_analysis_results,
        functional_analysis_results
    ])
    
    # Generate comprehensive understanding
    codebase_knowledge = {
        "system_overview": generate_system_overview(),
        "architectural_understanding": create_architecture_model(),
        "functional_capabilities": map_feature_landscape(),
        "technical_characteristics": assess_technical_qualities(),
        "operational_requirements": understand_operational_needs(),
        "improvement_opportunities": identify_enhancement_areas()
    }
    
    return codebase_knowledge
```

### Phase 3: Knowledge Artifact Creation
```python
def create_knowledge_artifacts(codebase_knowledge):
    """Generate comprehensive knowledge artifacts"""
    
    artifacts = {
        "system_architecture_map": generate_visual_architecture(),
        "component_interaction_diagrams": create_component_diagrams(),
        "data_flow_visualizations": generate_data_flow_charts(),
        "api_documentation": create_comprehensive_api_docs(),
        "developer_onboarding_guide": generate_onboarding_materials(),
        "operational_runbooks": create_operational_documentation(),
        "troubleshooting_knowledge_base": generate_troubleshooting_guide()
    }
    
    return artifacts
```

## MCP Tool Coordination Strategy

### Multi-Tool Intelligence Pipeline
```bash
# Coordinated analysis execution
mcp_filesystem.scan() | mcp_git.analyze() | cursor_ai.understand() | mcp_security.assess()
↓
mcp_dependencies.map() | mcp_profiler.baseline() | mcp_testing.coverage()
↓
cursor_ai.synthesize() | generate_documentation() | create_visualizations()
```

### Real-Time Analysis Orchestration
```python
class CodebaseAnalysisOrchestrator:
    def __init__(self):
        self.mcp_tools = initialize_mcp_tools()
        self.cursor_ai = initialize_cursor_intelligence()
        
    def execute_comprehensive_analysis(self):
        """Orchestrate complete codebase analysis"""
        
        # Parallel analysis execution
        analysis_results = asyncio.gather([
            self.structural_analysis(),
            self.functional_analysis(), 
            self.security_analysis(),
            self.performance_analysis(),
            self.quality_analysis()
        ])
        
        # Synthesis and knowledge generation
        comprehensive_understanding = self.synthesize_results(analysis_results)
        knowledge_artifacts = self.generate_artifacts(comprehensive_understanding)
        
        return knowledge_artifacts
```

## Output Generation Framework

### Comprehensive System Documentation
```markdown
# Complete Codebase Intelligence Report

## Executive Summary
- System Type: [Auto-identified system category]
- Architecture Pattern: [Detected architectural pattern]
- Technology Stack: [Complete stack analysis]
- Complexity Score: [Calculated complexity metric]
- Maintainability Index: [Automated assessment]

## System Architecture
### Component Topology
[Auto-generated architecture diagram]

### Service Interactions
[Component interaction visualization]

### Data Flow Architecture
[Data flow diagram with intelligent annotations]

## Feature Landscape
### Core Capabilities
[Auto-discovered feature catalog with descriptions]

### User Workflows
[Traced user interaction patterns]

### Business Logic Mapping
[Identified business rules and logic]

## Technical Deep Dive
### Code Organization Patterns
[Analysis of code structure and organization]

### Design Pattern Usage
[Identified design patterns and their implementations]

### Performance Characteristics
[Performance profile and bottleneck analysis]

### Security Implementation
[Security feature analysis and recommendations]

## Developer Knowledge Base
### Onboarding Guide
[Auto-generated getting started guide]

### Architecture Understanding
[Detailed architectural explanations]

### Common Tasks Documentation
[Frequently needed operations and procedures]

### Troubleshooting Guide
[Common issues and resolution procedures]
```

### Interactive Knowledge System
```python
# Create interactive documentation system
interactive_docs = cursor_ai.create_interactive_system([
    "searchable_code_explanations",
    "context_aware_documentation",
    "intelligent_code_navigation",
    "automated_example_generation",
    "real_time_impact_analysis"
])
```

Execute this comprehensive analysis using the full intelligence capabilities of Claude.code, Cursor IDE's semantic understanding, and coordinated MCP tool execution to generate complete codebase knowledge and understanding.