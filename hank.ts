import type { AgentDefinition } from './types/agent-definition'

const definition: AgentDefinition = {
  id: 'hank',
  displayName: 'Hank - Expert Codebase Analyzer',
  model: 'anthropic/claude-4-sonnet-20250522',
  
  reasoningOptions: {
    enabled: true,
    effort: 'high',
    exclude: false
  },
  
  toolNames: [
    'read_files',
    'write_file',
    'str_replace',
    'code_search', 
    'find_files',
    'run_terminal_command',
    'web_search',
    'read_docs',
    'spawn_agents',
    'think_deeply',
    'set_output',
    'add_message',
    'end_turn'
  ],
  
  spawnableAgents: [
    'codebuff/file-picker@0.0.2',
    'codebuff/file-explorer@0.0.4',
    'codebuff/reviewer@0.0.8',
    'codebuff/researcher@0.0.2',
    'codebuff/thinker@0.0.2'
  ],
  
  spawnerPrompt: 'Spawn Hank when you need expert codebase analysis, architecture review, or deep code understanding',
  
  systemPrompt: `You are an Expert Codebase Analysis and Systems Integration Specialist with deep expertise in enterprise software development, technical debt assessment, and large-scale system migrations. Your mission is to perform comprehensive analysis of development documentation against actual code implementations.`,
  
  instructionsPrompt: `
AGENT ROLE & EXPERTISE:
- Senior Software Architect with 10+ years enterprise experience
- Systems Integration Specialist (API integrations, data migrations)
- Code Quality Expert (technical debt, security, performance analysis)
- Project Planning Specialist (gap analysis, risk assessment, implementation planning)

YOUR ANALYSIS OBJECTIVES:
1. **DOCUMENTATION vs CODE VALIDATION**
   - Parse all development plans and specifications
   - Map documented requirements to actual implementations
   - Identify gaps, inconsistencies, and missing functionality

2. **IMPLEMENTATION STATUS ASSESSMENT** 
   - Categorize each requirement: Missing/Partial/Incorrect/Complete
   - Assess code quality, security, and performance issues
   - Evaluate technical debt and architectural concerns

3. **RISK & IMPACT ANALYSIS**
   - Prioritize gaps by business impact and implementation complexity
   - Identify dependencies and implementation order
   - Assess risks to production system stability

4. **ACTIONABLE IMPLEMENTATION PLAN**
   - Create detailed, prioritized action plan with specific tasks
   - Provide implementation guidance with code examples
   - Define acceptance criteria and validation procedures

ANALYSIS METHODOLOGY:
- Phase 1: Documentation parsing and requirement extraction
- Phase 2: Codebase scanning and feature mapping  
- Phase 3: Gap analysis and categorization
- Phase 4: Priority assessment and implementation planning

CRITICAL CONSTRAINTS:
- This is a PRODUCTION system - zero tolerance for downtime
- Maintain 100% backward compatibility during transition
- All existing functionality must continue working
- Performance requirements: API <500ms, Dashboard <3s
- Security is paramount - enterprise-grade authentication required

OUTPUT REQUIREMENTS:
1. **Executive Summary** - High-level findings and recommendations
2. **Detailed Gap Analysis** - Specific missing/incorrect implementations
3. **Prioritized Action Plan** - Phased implementation with timelines
4. **Risk Assessment** - Potential issues and mitigation strategies
5. **Implementation Tasks** - Specific code changes with acceptance criteria

QUALITY STANDARDS:
- Every finding must be specific with file/line references
- Every recommendation must include implementation steps
- Every high-priority item must include risk assessment
- Every task must have clear acceptance criteria
- Evidence-based analysis with concrete examples
  `,
  
  stepPrompt: 'Continue your codebase analysis systematically. Use deep thinking for complex architectural decisions.',
  
  inputSchema: {
    prompt: { type: 'string', description: 'Analysis request or focus area' },
    params: {
      type: 'object',
      properties: {
        scopePaths: { type: 'array', items: { type: 'string' }, description: 'Limit analysis to specific paths' },
        writeMode: { type: 'string', enum: ['read-only', 'dry-run', 'apply'], description: 'Output mode for recommendations' },
        maxFiles: { type: 'number', description: 'Maximum files to analyze' },
        riskThreshold: { type: 'string', enum: ['low', 'medium', 'high'], description: 'Minimum risk level to report' },
        includeTests: { type: 'boolean', description: 'Include test files in analysis' }
      }
    }
  },
  
  outputMode: 'structured_output',
  
  outputSchema: {
    type: 'object',
    properties: {
      executive_summary: { type: 'string', description: 'High-level findings and recommendations' },
      architecture_overview: { type: 'string', description: 'System architecture and key components' },
      gaps: {
        type: 'array',
        items: {
          type: 'object',
          properties: {
            id: { type: 'string' },
            file: { type: 'string' },
            line: { type: 'number' },
            category: { type: 'string', enum: ['Missing', 'Partial', 'Incorrect', 'Complete'] },
            severity: { type: 'string', enum: ['Critical', 'High', 'Medium', 'Low'] },
            description: { type: 'string' },
            rationale: { type: 'string' }
          }
        }
      },
      risks: {
        type: 'array',
        items: {
          type: 'object',
          properties: {
            id: { type: 'string' },
            risk: { type: 'string' },
            impact: { type: 'string', enum: ['High', 'Medium', 'Low'] },
            probability: { type: 'string', enum: ['High', 'Medium', 'Low'] },
            mitigation: { type: 'string' }
          }
        }
      },
      implementation_plan: {
        type: 'array',
        items: {
          type: 'object',
          properties: {
            phase: { type: 'string' },
            tasks: { type: 'array', items: { type: 'string' } },
            dependencies: { type: 'array', items: { type: 'string' } },
            estimated_effort: { type: 'string' }
          }
        }
      }
    },
    required: ['executive_summary', 'gaps', 'risks', 'implementation_plan']
  },
  
  async *handleSteps({ agentState, prompt, params = {} }) {
    // Phase 1: Documentation Discovery and Requirements Extraction
    yield {
      toolName: 'add_message',
      input: { content: '🔍 **Phase 1: Documentation Analysis**\n\nScanning for documentation and requirements...' }
    }
    
    // Find documentation files
    yield {
      toolName: 'find_files',
      input: {
        patterns: ['*.md', '*.rst', '*.txt'],
        excludePatterns: ['node_modules/**', '.git/**']
      }
    }
    
    // Read key documentation
    const { toolResult: docFiles } = yield 'STEP'
    if (docFiles && docFiles[0]?.files?.length > 0) {
      const importantDocs = docFiles[0].files.slice(0, 10) // Limit to first 10 docs
      yield {
        toolName: 'read_files',
        input: { paths: importantDocs.map(f => f.path) }
      }
    }
    
    // Phase 2: Codebase Structure Analysis  
    yield {
      toolName: 'add_message',
      input: { content: '🏗️ **Phase 2: Codebase Structure Analysis**\n\nAnalyzing project structure and key components...' }
    }
    
    // Spawn file explorer for comprehensive discovery
    yield {
      toolName: 'spawn_agents',
      input: {
        agents: [{
          agent_type: 'codebuff/file-explorer@0.0.4',
          prompt: 'Explore codebase structure and identify key architectural components',
          params: {
            prompts: [
              'Find main application entry points and configuration files',
              'Locate API endpoints and service definitions', 
              'Identify test files and testing infrastructure',
              'Find database schemas and migration files'
            ]
          }
        }]
      }
    }
    
    // Phase 3: Deep Code Analysis
    yield {
      toolName: 'add_message', 
      input: { content: '🔬 **Phase 3: Deep Code Analysis**\n\nPerforming detailed code quality and security analysis...' }
    }
    
    // Think deeply about findings
    yield {
      toolName: 'think_deeply',
      input: {
        thought: 'Analyzing codebase patterns, technical debt, security issues, and architectural concerns based on documentation and structure discovery'
      }
    }
    
    // Phase 4: Risk Assessment and Planning
    yield {
      toolName: 'add_message',
      input: { content: '📋 **Phase 4: Implementation Planning**\n\nGenerating actionable recommendations and implementation plan...' }
    }
    
    // Generate final structured output
    yield 'STEP_ALL'
  }
}

export default definition