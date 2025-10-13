# AI Workspace App Development Templates

This directory contains templates for rapid application development using the AI Workspace framework.

## Templates Included

### 1. System Prompt Template (`system-prompt-template.md`)

**Purpose**: Comprehensive system prompt that guides AI assistants in building production-ready applications from specifications.

**Key Features**:
- Role definition and core responsibilities
- AI Workspace agent integration
- Development workflow (5-phase approach)
- Quality standards and enforcement
- Technology stack templates
- Error handling patterns
- Output format specifications

**When to Use**: Provide this prompt to your AI assistant (Claude, GPT-4, etc.) at the start of any app development project to ensure consistent, high-quality output.

**Usage**:
```bash
# Copy the system prompt to your conversation
cat .ai-workspace/templates/system-prompt-template.md

# Then provide your app specification
cat my-app-spec.md
```

---

### 2. App Specification Template (`app-spec-template.md`)

**Purpose**: Structured template for documenting complete application requirements, ensuring nothing is missed during development.

**Key Sections**:
1. **Project Overview** - Name, description, audience, business goals
2. **Functional Requirements** - Features, user stories, acceptance criteria
3. **UI Requirements** - User flows, pages, design system
4. **Technical Requirements** - Stack, integrations, auth
5. **Data Model** - Entities, relationships, validation
6. **API Specification** - Endpoints, request/response formats
7. **Non-Functional Requirements** - Performance, security, scalability
8. **Quality Standards** - Code quality levels, testing, documentation
9. **Development Phases** - MVP, post-MVP roadmap
10. **Constraints & Assumptions** - Limitations, risks, out-of-scope
11. **Success Metrics** - Technical, business, UX metrics
12. **Deployment & Operations** - Environments, monitoring, alerting

**When to Use**: Fill out this template BEFORE starting development to ensure complete requirements gathering.

**Usage**:
```bash
# Copy template to your project
cp .ai-workspace/templates/app-spec-template.md ./docs/app-spec.md

# Edit with your app details
code ./docs/app-spec.md
```

---

## Complete Development Workflow

### Step 1: Fill Out App Specification

1. Copy the app spec template to your project
2. Complete all relevant sections
3. Be as specific as possible (reduces ambiguity)
4. Include mockups/wireframes if available
5. Review and validate completeness

**Example**:
```bash
# Create docs directory
mkdir -p ./docs

# Copy template
cp .ai-workspace/templates/app-spec-template.md ./docs/my-saas-app-spec.md

# Fill it out
# ... (edit the file with your app details)
```

### Step 2: Initialize AI Workspace

```bash
# Initialize workspace with appropriate preset
ai-workspace init . --preset typescript-nextjs --interactive

# Or let it auto-detect your stack
ai-workspace init .
```

### Step 3: Start AI-Assisted Development

**Option A: Use with Claude Code or similar AI assistant**

1. Start a new conversation
2. Paste the system prompt template as initial context
3. Provide your completed app specification
4. Let the AI guide you through the 5 phases

**Example Conversation**:
```
User: [Paste system-prompt-template.md]

User: Here is my app specification:
[Paste completed app-spec.md]

User: Please proceed with building this application following the AI Workspace methodology.

AI: [Analyzes spec, recommends agents, creates plan, begins implementation]
```

**Option B: Use CLI Recommendations**

```bash
# Get recommended agents for specific features
ai-workspace recommend "user authentication with OAuth"
ai-workspace recommend "real-time chat" --category 01-development
ai-workspace recommend "PostgreSQL optimization" --top 3

# Use recommended agents in your workflow
# ... (follow agent-specific guidance)
```

### Step 4: Follow the 5-Phase Development Process

The system prompt guides development through these phases:

**Phase 1: Discovery** (Days 1-2)
- Analyze requirements from app spec
- Detect technology stack
- Get agent recommendations
- Create initial ADRs
- Validate completeness

**Phase 2: Design** (Days 3-5)
- Design database schema
- Define API contracts
- Create architecture diagram
- Design component hierarchy
- Document decisions

**Phase 3: Implementation** (Days 6-15)
- Search for reusable code (`search_codebase.py`)
- Generate backend code (models → services → routes)
- Generate frontend code (components → pages → state)
- Write tests alongside features
- Enforce quality standards

**Phase 4: Verification** (Days 16-18)
- Run test suites
- Security audit
- Performance testing
- Code review
- Documentation review

**Phase 5: Integration** (Days 19-20)
- Setup CI/CD
- Deploy to staging
- Integration testing
- Deploy to production
- Create handoff docs

### Step 5: Quality Enforcement

Throughout development, AI Workspace enforces quality:

```bash
# Check before creating new code (automatic)
python .ai-workspace/scripts/search_codebase.py "authentication"
python .ai-workspace/scripts/analyze_reuse.py

# Validate phase gates
python .ai-workspace/scripts/validate_gates.py

# Pre-commit hooks run automatically
git commit -m "Add user authentication"
# → Checks complexity, duplication, file size
```

---

## Example: Building a SaaS Dashboard

### 1. App Specification (Condensed)

```yaml
name: Analytics Dashboard Pro
description: Real-time analytics dashboard for SaaS metrics
stack: typescript-nextjs
quality: balanced

features:
  - User authentication (OAuth + Email)
  - Real-time metric visualization
  - Custom dashboard builder
  - Team collaboration
  - API for data ingestion

mvp_timeline: 4 weeks
```

### 2. Initialization

```bash
ai-workspace init . --preset typescript-nextjs
# Creates:
# - .ai-workspace/ (with all agents)
# - .claude/ (symlink)
# - .cursor/ (symlink)
# - CLAUDE.md
# - Pre-commit hooks
```

### 3. AI-Assisted Development

```
User: [System prompt] + [App spec]

AI:
## Discovery Phase Complete

**Stack Detected**: Next.js 14, TypeScript, PostgreSQL, Redis

**Recommended Agents**:
1. nextjs-developer - Full-stack development
2. typescript-pro - Type-safe implementation
3. postgres-pro - Database optimization
4. react-specialist - Component architecture
5. api-designer - REST API design

**Reuse Analysis**: Found 3 similar authentication patterns
- Reusing: OAuth implementation (85% match)
- Reusing: Dashboard layout component (72% match)
- Creating new: Real-time metrics engine

**Next**: Proceeding to Design Phase...

## Design Phase

**Database Schema**:
[Generated schema with relationships]

**API Specification**:
[Generated OpenAPI spec]

**Component Architecture**:
[Component hierarchy]

**ADRs Created**:
- ADR-001: Use Prisma for ORM
- ADR-002: Use TanStack Query for state management
- ADR-003: Use WebSocket for real-time updates

**Next**: Proceeding to Implementation...

[Continues through all phases]
```

### 4. Output

After ~20 days (or appropriate timeline):

```
Generated Files:
├── src/
│   ├── app/              (Next.js 14 app directory)
│   ├── components/       (42 React components)
│   ├── lib/              (Utilities, API clients)
│   ├── server/           (API routes, database)
│   └── types/            (TypeScript definitions)
├── tests/                (85% coverage achieved)
├── docs/
│   ├── api/              (OpenAPI documentation)
│   └── architecture/     (ADRs, diagrams)
├── .github/workflows/    (CI/CD pipelines)
└── README.md             (Complete setup guide)

Quality Metrics:
- Complexity: 7.2 (target ≤10) ✓
- Duplication: 2.1% (target <3%) ✓
- Test Coverage: 87% (target ≥85%) ✓
- Performance: p95 < 150ms ✓

Ready for deployment!
```

---

## Customizing Templates

### Modify System Prompt

Edit `.ai-workspace/templates/system-prompt-template.md` to:
- Add company-specific coding standards
- Include custom quality metrics
- Add proprietary tools/frameworks
- Adjust agent recommendations
- Change output formats

### Extend App Spec Template

Edit `.ai-workspace/templates/app-spec-template.md` to:
- Add industry-specific sections (e.g., compliance for healthcare)
- Include custom quality gates
- Add company-specific metadata
- Incorporate domain-specific requirements

### Create Domain-Specific Templates

Create specialized versions for common project types:

```bash
# E-commerce template
cp app-spec-template.md ecommerce-app-spec-template.md
# ... customize for e-commerce (products, cart, checkout)

# Mobile app template
cp app-spec-template.md mobile-app-spec-template.md
# ... customize for mobile (native features, app store)

# API-only template
cp app-spec-template.md api-only-spec-template.md
# ... remove frontend sections, focus on endpoints
```

---

## Best Practices

### For App Specifications

1. **Be Specific**: Vague requirements lead to incorrect implementations
2. **Include Examples**: Show sample data, API responses, UI mockups
3. **Define Success**: Clear acceptance criteria for each feature
4. **Prioritize Ruthlessly**: Not everything needs to be in MVP
5. **Document Constraints**: What can't you use? What must you use?
6. **Validate Early**: Review spec before starting implementation

### For System Prompts

1. **Start General**: Use the template as-is for first project
2. **Iterate**: Refine based on actual output quality
3. **Be Consistent**: Keep the same prompt across similar projects
4. **Version Control**: Track prompt changes in git
5. **Share Knowledge**: Document what works in your team

### For AI-Assisted Development

1. **Search First**: Always check for reusable code before creating
2. **Follow Phases**: Don't skip ahead (especially Design)
3. **Review Output**: AI-generated code still needs human review
4. **Test Everything**: Don't trust without verification
5. **Document Decisions**: Capture why, not just what

---

## Troubleshooting

### "AI doesn't follow the system prompt"

**Solution**:
- Break down large specs into smaller sections
- Provide the prompt at the start of EVERY conversation
- Be explicit: "Follow the system prompt template exactly"
- Use Claude Code or GPT-4 (better instruction following)

### "Generated code doesn't match spec"

**Solution**:
- Review app spec for ambiguity
- Add more detailed acceptance criteria
- Provide examples of expected behavior
- Use iterative refinement

### "Quality gates failing"

**Solution**:
```bash
# Check what's failing
git commit -m "test" --dry-run

# Fix complexity
# → Refactor complex functions into smaller ones

# Fix duplication
# → Extract repeated code into utilities

# Fix file size
# → Split large files into modules
```

### "Can't find reusable code"

**Solution**:
```bash
# Try different search terms
python .ai-workspace/scripts/search_codebase.py "auth"
python .ai-workspace/scripts/search_codebase.py "login"
python .ai-workspace/scripts/search_codebase.py "user management"

# Search in specific languages
# Edit search_codebase.py to filter by file extension
```

---

## Additional Resources

### AI Workspace Documentation
- **Main README**: `E:\agents\README.md`
- **Agent Library**: `E:\agents\.ai-workspace\agents\*/README.md`
- **Scripts Documentation**: `E:\agents\.ai-workspace\scripts\README.md`

### Example Projects
```bash
# Create example projects
ai-workspace init /tmp/example-fastapi --preset python-fastapi
ai-workspace init /tmp/example-nextjs --preset typescript-nextjs
ai-workspace init /tmp/example-microservices --preset microservices
```

### Community
- **GitHub Issues**: Report bugs, request features
- **Discussions**: Share your app specs and results
- **Examples**: Contribute successful app specs

---

## Changelog

### v1.2.0 (Current)
- Added system prompt template
- Added comprehensive app spec template
- Integrated with lazy loading and presets
- Added 5-phase development methodology

### Future Enhancements
- [ ] Domain-specific templates (e-commerce, fintech, healthcare)
- [ ] Visual spec builder (web UI)
- [ ] Automated spec validation
- [ ] Template marketplace
- [ ] IDE integration (VS Code extension)

---

## License

These templates are part of the AI Workspace package and follow the same license.

---

**Need Help?**

```bash
ai-workspace --help
ai-workspace recommend --help
```

Or consult the main documentation in `E:\agents\README.md`.
