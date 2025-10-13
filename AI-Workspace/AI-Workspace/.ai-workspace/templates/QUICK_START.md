# Quick Start: Build an App with AI Workspace Templates

**Time to First App**: ~30 minutes to spec, ~2-20 days to build (depending on complexity)

---

## 3-Step Process

### Step 1: Create Your App Specification (15-30 min)

```bash
# Copy the template
cp .ai-workspace/templates/app-spec-template.md ./my-app-spec.md

# Fill it out (or use your editor)
code ./my-app-spec.md
```

**Minimum Required Sections** (for MVP):
1. ✅ Project Overview (name, description, audience)
2. ✅ Core Features (3-5 main features with acceptance criteria)
3. ✅ Technology Stack (backend, frontend, database)
4. ✅ Data Model (main entities and relationships)
5. ✅ API Endpoints (key routes and responses)

**Optional but Recommended**:
- UI mockups/wireframes
- Success metrics
- MVP timeline

---

### Step 2: Initialize AI Workspace (2 min)

```bash
# Option A: Use a preset
ai-workspace init . --preset typescript-nextjs

# Option B: Let it detect your stack
ai-workspace init .

# Option C: Interactive wizard
ai-workspace init . --interactive
```

**Available Presets**:
- `python-fastapi` - FastAPI + PostgreSQL + Redis
- `typescript-nextjs` - Next.js + React + Prisma
- `django-postgres` - Django + Celery + PostgreSQL
- `rust-actix` - Actix-web + SQLx + PostgreSQL
- `microservices` - Kubernetes + gRPC + multiple services

---

### Step 3: Start AI-Assisted Development

**Copy this prompt to Claude Code (or similar):**

```markdown
I want to build an application following the AI Workspace methodology.

System Prompt:
[Paste contents of .ai-workspace/templates/system-prompt-template.md]

App Specification:
[Paste contents of your my-app-spec.md]

Please proceed with the Discovery Phase.
```

**The AI will**:
1. ✅ Analyze your specification
2. ✅ Recommend appropriate agents
3. ✅ Search for reusable code
4. ✅ Generate architecture design
5. ✅ Implement features with tests
6. ✅ Enforce quality standards
7. ✅ Create documentation

---

## Example: To-Do App (Minimal)

### Step 1: Spec (5 min)

```yaml
# my-todo-app-spec.md

name: Simple Todo App
description: A basic todo list with user accounts
stack: typescript-nextjs
quality: balanced

features:
  1. User Registration/Login (Email + Password)
  2. Create/Read/Update/Delete Todos
  3. Mark todos as complete/incomplete
  4. Filter by status (all/active/completed)

data_model:
  User:
    - id: UUID
    - email: String (unique)
    - password_hash: String

  Todo:
    - id: UUID
    - user_id: UUID (FK)
    - title: String
    - completed: Boolean
    - created_at: Timestamp

api_endpoints:
  - POST /api/auth/register
  - POST /api/auth/login
  - GET /api/todos (list user's todos)
  - POST /api/todos (create todo)
  - PATCH /api/todos/:id (update todo)
  - DELETE /api/todos/:id (delete todo)

mvp_timeline: 1 week
```

### Step 2: Initialize (1 min)

```bash
ai-workspace init . --preset typescript-nextjs
```

### Step 3: Build (1 week with AI)

```
User: [System prompt + App spec]

AI:
## Discovery Phase
Stack: Next.js 14, TypeScript, Prisma, PostgreSQL
Recommended: nextjs-developer, typescript-pro, postgres-pro

## Design Phase
Database schema created ✓
API spec generated ✓
Component architecture defined ✓

## Implementation Phase
[Generates ~30 files]
- Authentication system ✓
- Todo CRUD operations ✓
- Frontend components ✓
- Tests (88% coverage) ✓

## Verification Phase
All tests passing ✓
Security audit complete ✓
Performance benchmarks met ✓

## Integration Phase
CI/CD pipeline configured ✓
Deployed to Vercel ✓
Documentation complete ✓

🎉 Your todo app is ready!
```

---

## Tips for Success

### Writing Good Specs

✅ **Do**:
- Be specific about user flows
- Include example data
- Define clear acceptance criteria
- Prioritize features (MVP vs nice-to-have)
- Specify technology constraints

❌ **Don't**:
- Leave requirements vague ("should be fast")
- Skip the data model section
- Forget to define API contracts
- Over-engineer the MVP
- Mix features with implementation details

### Working with AI

✅ **Do**:
- Follow the 5-phase process
- Review generated code
- Ask for explanations
- Iterate on designs
- Test everything

❌ **Don't**:
- Skip the Discovery/Design phases
- Blindly accept all generated code
- Ignore quality gate failures
- Create code without searching first
- Deploy without testing

### Quality Enforcement

AI Workspace automatically enforces:

```bash
# Pre-commit hooks check:
✓ Complexity ≤10 per function
✓ Duplication <3% across codebase
✓ File size ≤500 lines
✓ Test coverage ≥85%

# Search-before-create ensures:
✓ Reuse existing patterns
✓ Don't duplicate logic
✓ Learn from existing code
```

---

## Common Patterns

### Single Page App (SPA)

```yaml
stack: typescript-nextjs
features:
  - Client-side routing
  - State management (React Query)
  - API integration
  - Authentication (JWT)
```

### REST API (Backend Only)

```yaml
stack: python-fastapi
features:
  - RESTful endpoints
  - Database ORM (SQLAlchemy)
  - Authentication (OAuth2)
  - API documentation (OpenAPI)
```

### Full-Stack Monolith

```yaml
stack: django-postgres
features:
  - Server-side rendering
  - Admin panel
  - Background jobs (Celery)
  - WebSocket support
```

### Microservices

```yaml
stack: microservices
services:
  - auth-service (user management)
  - api-gateway (routing)
  - data-service (business logic)
infrastructure:
  - Kubernetes orchestration
  - gRPC inter-service communication
  - Redis caching
```

---

## Next Steps After MVP

Once your MVP is built:

1. **Gather Feedback**
   ```bash
   # Deploy to staging
   # Share with beta users
   # Collect metrics
   ```

2. **Plan Phase 2**
   ```bash
   # Update app spec with new features
   # Re-run AI workflow for additions
   ```

3. **Optimize**
   ```bash
   # Use performance-engineer agent
   ai-workspace recommend "optimize database queries"
   ai-workspace recommend "reduce bundle size"
   ```

4. **Scale**
   ```bash
   # Use devops-engineer agent
   ai-workspace recommend "kubernetes deployment"
   ai-workspace recommend "auto-scaling setup"
   ```

---

## Troubleshooting

### "AI generates incorrect code"

**Check**:
- Is your app spec detailed enough?
- Did you provide example data?
- Did you define acceptance criteria?

**Fix**:
- Add more details to spec
- Provide sample inputs/outputs
- Be explicit about edge cases

### "Quality gates failing"

```bash
# Check what's failing
git commit -m "test"

# Common fixes:
# - Complexity: Break up large functions
# - Duplication: Extract to utilities
# - Coverage: Add more tests
```

### "Build errors"

```bash
# Let AI fix it
"The build is failing with error: [paste error]. Please fix."

# AI will:
# 1. Analyze the error
# 2. Search for solutions
# 3. Apply fix
# 4. Verify build passes
```

---

## Resources

**Templates**:
- `.ai-workspace/templates/system-prompt-template.md` - AI instructions
- `.ai-workspace/templates/app-spec-template.md` - Full spec template
- `.ai-workspace/templates/README.md` - Detailed guide

**Tools**:
```bash
ai-workspace recommend "feature description"  # Get agent recommendations
ai-workspace status                            # Check installation
ai-workspace info                              # Show version info
```

**Examples**:
```bash
# See example projects
ls .ai-workspace/presets/

# Create test project
ai-workspace init /tmp/test-app --preset python-fastapi
```

---

## Success Metrics

**You know it's working when**:

✅ Generated code passes all quality gates
✅ Tests achieve ≥85% coverage
✅ App matches specification requirements
✅ Performance meets targets
✅ Documentation is complete
✅ Deployment succeeds

**Typical Results**:
- **Time Saved**: 60-80% vs manual coding
- **Code Quality**: Consistent, well-tested
- **Documentation**: Always up-to-date
- **Reuse**: 40-70% of code reused from patterns

---

## Get Help

```bash
# CLI help
ai-workspace --help

# Documentation
cat .ai-workspace/templates/README.md

# Community
# → GitHub Issues
# → Discussions
```

---

**Ready to build? Start with Step 1! 🚀**
