# 🤖 AI Workspace - Universal AI Development System

**Version:** 1.0.0
**License:** MIT
**Status:** Production Ready

A portable, framework-agnostic AI development orchestration system that brings intelligent multi-agent coordination, quality gates, and automated workflows to any project.

---

## 🎯 What is AI Workspace?

AI Workspace is a **drag-and-drop folder** that gives you:

- ✅ **156 Specialized AI Agents** - Database experts, DevOps engineers, security analysts, and more
- ✅ **5 Core Principles Enforcement** - Automated quality gates and workflow validation
- ✅ **Multi-Agent Orchestration** - Coordinate agents through handoffs and phase gates
- ✅ **Auto-Detected Stack Configuration** - Works with Python, TypeScript, Rust, Go, Java, etc.
- ✅ **Reuse-First Workflow** - Semantic code search before creating new code
- ✅ **Phase Gate Reviews** - 5-stage workflow with mandatory validations
- ✅ **Automated Documentation** - ADRs, handoffs, decisions tracked automatically

---

## 🚀 Quick Start (30 seconds)

### 1. Installation

```bash
# Copy .ai-workspace into your project root
cp -r /path/to/.ai-workspace your-project/

# Run setup
cd your-project/
python .ai-workspace/scripts/setup.py

# Done! ✨
```

### 2. What Gets Installed

After setup, your project will have:

```
your-project/
├── .ai-workspace/           # ✨ AI orchestration (this folder)
├── CLAUDE.md                # ✨ Auto-generated AI team guide
├── .cursor/rules/           # ✨ AI behavior rules (symlinked)
├── .claude/agents/          # ✨ 156 agents (symlinked)
├── .git/hooks/pre-commit    # ✨ Quality gates
└── .ai-workspace-config.yml # ✨ Project configuration
```

### 3. First Commands

```bash
# View your AI team configuration
cat CLAUDE.md

# Search for existing code before creating new
python .ai-workspace/scripts/search_codebase.py "user authentication"

# Analyze reuse potential
python .ai-workspace/scripts/analyze_reuse.py "email validation" src/validators.py

# Plan a new task
python .ai-workspace/scripts/plan_task.py --task "Add payment integration"

# Create architectural decision record
python .ai-workspace/scripts/create_adr.py --title "Use PostgreSQL for data storage"

# Validate phase gates
python .ai-workspace/scripts/validate_gates.py --check-all
```

---

## 📚 The 5 Core Principles

AI Workspace enforces quality through **5 Core Principles**:

### **Principle 1: Less is More**
- ✅ Reuse before create (mandatory search workflow)
- ✅ Smallest viable solution
- ✅ Complexity ≤10 (enforced via pre-commit)
- ✅ Duplication <3% (measured automatically)

### **Principle 2: Document Everything**
- ✅ ADR for all architectural decisions
- ✅ Complete docstrings and comments
- ✅ Handoffs tracked in `.agent-workspace/handoffs/`
- ✅ Decision history in `.agent-workspace/decisions/`

### **Principle 3: Check Before Create**
- ✅ **Automatic enforcement** - BLOCKS code creation without reuse analysis
- ✅ Mandatory reuse workflow:
  1. Search codebase (search_codebase.py) - auto-recorded ✓
  2. Analyze reuse % (analyze_reuse.py) - auto-recorded ✓
  3. If ≥70%: wrapper/adapter (auto-approved)
  4. If <70%: new code + ADR (requires manual approval)
- ✅ **Cannot bypass** - enforcement script validates before any code writing
- ✅ **30-minute cache** - reuse analysis expires after 30 min (must re-check)

**How it works:**
```bash
# Step 1: Agent searches (automatically recorded)
python .ai-workspace/scripts/search_codebase.py "user authentication"
# → Recorded in .ai-workspace/cache/reuse_checks.json

# Step 2: Agent analyzes (automatically recorded)
python .ai-workspace/scripts/analyze_reuse.py "user auth" src/auth.py
# → Records reuse %, auto-approves if ≥70%

# Step 3: Before writing ANY new code, enforcement check runs
python .ai-workspace/scripts/enforce_check_before_create.py --check user-auth "user authentication"
# → ✅ APPROVED (78% reuse) or ❌ BLOCKED (need ADR)
```

### **Principle 4: Phase Gate Reviews**
- ✅ 5-stage workflow: Discovery → Design → Implementation → Verification → Integration
- ✅ Cannot skip gates (validated by validate_gates.py)
- ✅ Real data validation (no mocks in verification phase)
- ✅ Exit criteria enforced

### **Principle 5: Plan Before Act**
- ✅ Think → Research → Plan → Execute
- ✅ Structured planning (plan_task.py)
- ✅ Task decomposition
- ✅ Risk assessment

---

## 🛠️ Core Tools

### 1. **search_codebase.py** - Semantic Code Search
Enforces: **Principle 3 (Check Before Create)**

**Multi-Language Support:** Python, TypeScript, JavaScript, Rust, Go, Java (+ generic fallback)

```bash
# Search for similar implementations
python .ai-workspace/scripts/search_codebase.py "email validation"

# Output:
# 🟢 1. validate_email_format (85% match)
#    📁 src/validators.py:42
# 🟡 2. check_email_syntax (62% match)
#    📁 src/utils/email.py:15
```

**Language-Specific Features:**
- **Python:** AST-based parsing (functions, classes, docstrings, arguments)
- **TypeScript/JavaScript:** Regex extraction (functions, classes, arrow functions)
- **Rust:** Function, struct, and impl block detection
- **Go:** Function, struct, and interface detection
- **Java:** Method, class, and interface detection (excludes constructors)
- **Others:** Generic text-based matching

### 2. **analyze_reuse.py** - Reuse Analysis
Enforces: **Principles 1 & 3 (Less is More + Check Before Create)**

**Multi-Language Support:** Python, TypeScript, JavaScript, Rust, Go, Java (+ generic fallback)

```bash
# Calculate reuse potential
python .ai-workspace/scripts/analyze_reuse.py "user auth" src/auth/existing.py

# Output:
# 🟢 Reuse Potential: 78%
# 🎯 Action: Create wrapper/adapter around existing code
```

**Analysis Capabilities:**
- **Python:** Deep AST analysis with function/class inspection
- **TypeScript/JavaScript:** Function and class extraction
- **Rust:** Function, struct, and implementation analysis
- **Go:** Function, struct, and interface analysis
- **Java:** Method, class, and interface analysis
- Intelligent missing feature detection across all languages

### 3. **validate_gates.py** - Phase Gate Validator
Enforces: **Principle 4 (Phase Gate Reviews)**

```bash
# Validate all phase gates
python .ai-workspace/scripts/validate_gates.py --check-all

# Validate specific phase
python .ai-workspace/scripts/validate_gates.py --phase verification

# Create gate template
python .ai-workspace/scripts/validate_gates.py --create design
```

### 4. **create_adr.py** - ADR Creator
Enforces: **Principle 2 (Document Everything)**

```bash
# Interactive wizard
python .ai-workspace/scripts/create_adr.py

# Quick template
python .ai-workspace/scripts/create_adr.py --title "Use Redis for caching"

# From reuse analysis
python .ai-workspace/scripts/create_adr.py --from-reuse reuse_results.json
```

### 5. **plan_task.py** - Task Planner
Enforces: **Principle 5 (Plan Before Act)**

```bash
# Interactive planning wizard
python .ai-workspace/scripts/plan_task.py

# Quick plan
python .ai-workspace/scripts/plan_task.py --quick "Fix email bug"
```

### 6. **enforce_check_before_create.py** - Automatic Enforcement
Enforces: **Principle 3 (Check Before Create) - AUTOMATIC**

**🚫 BLOCKS code creation** until reuse analysis is complete!

```bash
# Check if new code creation is allowed
python .ai-workspace/scripts/enforce_check_before_create.py --check user-auth "user authentication"

# Output (if NOT allowed):
# ❌ BLOCKED: No reuse analysis found for 'user-auth'
# → REQUIRED: Run search_codebase.py first

# Output (if allowed):
# ✅ APPROVED: Reuse analysis complete (78% reuse)
# → You may proceed with wrapper/adapter

# Manually approve after creating ADR (for <70% reuse)
python .ai-workspace/scripts/enforce_check_before_create.py --approve user-auth path/to/ADR-XXX.md
```

**Enforcement Rules:**
- **Auto-records** when search_codebase.py and analyze_reuse.py run
- **Auto-approves** if reuse ≥70%
- **Requires ADR** if reuse <70% (manual approval needed)
- **30-min expiry** - old analyses must be refreshed
- **Cannot bypass** - agents MUST check before creating

---

## 🤝 Multi-Agent System

AI Workspace includes **156 specialized agents** organized into 10 categories:

### **Orchestration (9 agents)**
- `@workspace-initializer` - Setup workspace structure
- `@tech-lead-orchestrator` - Plan agent sequences
- `@multi-agent-coordinator` - Complex workflow orchestration
- `@context-compressor` - Prevent context overload
- `@agent-health-monitor` - Track agent status
- And 4 more...

### **Development (19 agents)**
- `@fullstack-developer` - End-to-end features
- `@backend-developer` - Server-side solutions
- `@frontend-developer` - UI engineering
- `@api-designer` - API architecture
- And 15 more...

### **Languages (11 agents)**
- `@python-pro` - Python 3.11+ expert
- `@typescript-pro` - TypeScript/advanced types
- `@rust-engineer` - Systems programming
- `@golang-pro` - Go microservices
- And 7 more...

### **Frameworks (9 agents)**
- `@nextjs-developer` - Next.js 14+ with App Router
- `@react-specialist` - React 18+ hooks/patterns
- `@django-developer` - Django 4+ development
- `@spring-boot-engineer` - Spring Boot 3+
- And 5 more...

### **Infrastructure (11 agents)**
- `@devops-engineer` - CI/CD automation
- `@kubernetes-specialist` - K8s orchestration
- `@terraform-engineer` - Infrastructure as code
- `@sre-engineer` - Site reliability
- And 7 more...

### **Quality (8 agents)**
- `@code-reviewer` - Code quality analysis
- `@test-automator` - Test frameworks
- `@qa-expert` - Quality assurance
- `@security-auditor` - Security assessment
- And 4 more...

### **Data & AI (6 agents)**
- `@data-engineer` - Data pipelines
- `@data-scientist` - ML/analysis
- `@ml-engineer` - ML lifecycle
- `@llm-architect` - LLM systems
- And 2 more...

### **Specialized (15 agents)**
- `@database-optimizer` - Query optimization
- `@performance-engineer` - System performance
- `@documentation-engineer` - Technical docs
- `@api-documenter` - API documentation
- And 11 more...

### **Support (53 agents)**
- `@debugger` - Complex issue diagnosis
- `@project-analyst` - Project analysis
- `@architect-reviewer` - Architecture review
- `@refactoring-specialist` - Code refactoring
- And 49 more...

### **Utilities (15 agents)**
- `@search-specialist` - Information retrieval
- `@research-analyst` - Research synthesis
- `@trend-analyst` - Trend analysis
- And 12 more...

---

## 📋 Typical Workflows

### **Starting a New Feature**

```bash
# 1. Plan the task
python .ai-workspace/scripts/plan_task.py --task "User notification system"

# 2. Search for existing code
python .ai-workspace/scripts/search_codebase.py "notifications"

# 3. Analyze reuse potential
python .ai-workspace/scripts/analyze_reuse.py "notifications" src/messaging/

# 4. Create ADR if needed
python .ai-workspace/scripts/create_adr.py --title "Notification architecture"

# 5. Implement (using multi-agent system)
# Use @python-pro, @database-optimizer, @test-automator

# 6. Validate gates
python .ai-workspace/scripts/validate_gates.py --check-all

# 7. Commit (quality gates run automatically)
git add .
git commit -m "feat: add user notification system"
```

### **Fixing a Bug**

```bash
# 1. Quick plan
python .ai-workspace/scripts/plan_task.py --quick "Fix email sending bug"

# 2. Use debugger agent
# @debugger analyze why emails not sending

# 3. Implement fix
# @python-pro fix email sending issue

# 4. Create gate file for verification phase
python .ai-workspace/scripts/validate_gates.py --create verification

# 5. Test with REAL data (Principle 4)
# Verify, then mark verification gate complete

# 6. Commit
git commit -m "fix: resolve email sending issue"
```

### **Refactoring Code**

```bash
# 1. Analyze for opportunities
# @code-reviewer identify refactoring opportunities in user_service.py

# 2. Plan refactoring
python .ai-workspace/scripts/plan_task.py --task "Refactor user service"

# 3. Create ADR
python .ai-workspace/scripts/create_adr.py --title "User service refactoring"

# 4. Implement incrementally
# @refactoring-specialist plan safe refactoring
# @python-pro implement changes

# 5. Verify tests pass
pytest tests/

# 6. Commit
git commit -m "refactor: simplify user service"
```

---

## 🎨 Quality Gates (Pre-Commit Hook)

Every commit automatically runs:

### **Required (Always Enforced)**
- ✅ File size ≤500 LOC (soft limit)

### **Optional (Skip if tools not installed)**
- ✅ Code duplication <3% (jscpd)
- ✅ Type coverage (mypy/tsc)
- ✅ Code formatting (black/prettier)

### **Bypass (Emergency Only)**
```bash
git commit --no-verify
```

---

## 📖 Documentation

- **[Quick Start Guide](docs/QUICK_START.md)** - Get running in 5 minutes
- **[Operating Charter](docs/OPERATING_CHARTER.md)** - Quality philosophy
- **[5 Core Principles](docs/PRINCIPLES.md)** - Detailed principle explanations
- **[Architecture](docs/ARCHITECTURE.md)** - System design
- **[Agent Guide](docs/AGENT_GUIDE.md)** - Using specialized agents
- **[Customization](docs/CUSTOMIZATION.md)** - Tailoring to your needs

---

## 🔧 Directory Structure

```
.ai-workspace/
├── README.md                    # This file
├── agents/                      # 156 specialized agents
│   ├── orchestration/          # Meta-agents (9)
│   ├── development/            # Dev agents (19)
│   ├── languages/              # Language experts (11)
│   ├── frameworks/             # Framework specialists (9)
│   ├── infrastructure/         # DevOps/SRE (11)
│   ├── quality/                # QA/testing (8)
│   ├── data-ai/                # Data/ML (6)
│   ├── specialized/            # Domain experts (15)
│   ├── support/                # Utilities (53)
│   └── utilities/              # Research/analysis (15)
│
├── cursor/                      # Cursor AI rules
│   ├── rules/                  # Context-aware rules
│   │   ├── 00-core/           # Universal rules
│   │   ├── 01-language/       # Python/TS/Rust
│   │   └── 02-framework/      # FastAPI/Next.js
│   └── patterns/               # Reusable patterns
│
├── workspace/                   # Multi-agent workspace
│   ├── templates/              # Code/doc templates
│   ├── examples/               # Example implementations
│   └── handoffs/               # Agent coordination
│
├── hooks/                       # Git hooks
│   └── pre-commit.template     # Quality gates
│
├── scripts/                     # Automation tools
│   ├── setup.py                # Installer
│   ├── search_codebase.py      # Semantic search
│   ├── analyze_reuse.py        # Reuse analyzer
│   ├── validate_gates.py       # Gate validator
│   ├── create_adr.py           # ADR creator
│   └── plan_task.py            # Task planner
│
├── config/                      # Configuration
│   ├── stack-configs/          # Stack-specific configs
│   └── settings.local.json     # Claude settings
│
├── docs/                        # Documentation
│   ├── QUICK_START.md          # Quick start guide
│   ├── OPERATING_CHARTER.md    # Quality philosophy
│   ├── PRINCIPLES.md           # 5 core principles
│   ├── ARCHITECTURE.md         # System architecture
│   ├── AGENT_GUIDE.md          # Agent usage guide
│   └── CUSTOMIZATION.md        # Customization guide
│
├── plugins/                     # Extensibility
│   └── custom/                 # Custom plugins
│
└── dashboard/                   # Live monitoring
    └── index.html              # Agent status UI
```

---

## ⚙️ Configuration

Customize AI Workspace via `.ai-workspace-config.yml` in your project root:

```yaml
extends: ".ai-workspace/defaults"

project:
  name: "My Project"
  type: "microservices"

overrides:
  quality_gates:
    test_coverage: 90          # Stricter than default 85%
    max_complexity: 8          # Stricter than default 10
    max_file_loc: 300          # Stricter than default 500

  agents:
    custom:
      - name: "domain-expert"
        description: "Expert in my domain"
        triggers: ["domain", "business"]

  rules:
    disable:
      - "02-framework/django-rules.md"
    enable:
      - "custom/my-rules.md"
```

---

## 🔄 Update & Maintenance

### **Update AI Workspace**
```bash
# Pull latest changes
cd .ai-workspace
git pull origin main

# Re-run setup
python scripts/setup.py
```

### **Version Check**
```bash
cat .ai-workspace/.version
```

### **Health Check**
```bash
# Validate installation
python .ai-workspace/scripts/setup.py --verify

# Check agent availability
ls .claude/agents/
```

---

## 🐛 Troubleshooting

### **"Setup failed"**
```bash
# Ensure Python 3.10+
python --version

# Run with verbose logging
python .ai-workspace/scripts/setup.py --verbose
```

### **"Symlinks not created (Windows)"**
```bash
# Run as administrator, or setup will use junctions/hard links automatically
```

### **"Pre-commit hook failing"**
```bash
# Check what's failing
.git/hooks/pre-commit

# Temporarily bypass (not recommended)
git commit --no-verify
```

### **"Agent not responding correctly"**
```bash
# Verify CLAUDE.md exists
ls CLAUDE.md

# Check agent symlinks
ls -la .claude/agents/
```

---

## 🤝 Contributing

AI Workspace is designed to be extended:

1. **Custom Agents**: Add `.md` files to `.ai-workspace/agents/custom/`
2. **Custom Rules**: Add `.md` files to `.cursor/rules/custom/`
3. **Custom Scripts**: Add to `.ai-workspace/scripts/`
4. **Custom Templates**: Add to `.ai-workspace/workspace/templates/`

---

## 📄 License

MIT License - See [LICENSE](LICENSE)

---

## 🙏 Credits

Built on principles from:
- AI Coding Agent Operating Charter
- Multi-agent orchestration patterns
- Software engineering best practices

Designed for developers who want **quality automation** without sacrificing **flexibility**.

---

## 📞 Support

- **Documentation**: `.ai-workspace/docs/`
- **Examples**: `.ai-workspace/workspace/examples/`
- **Issues**: Create issue in repository

---

**Ready to transform your development workflow?**

```bash
python .ai-workspace/scripts/setup.py
```

**Happy coding! 🤖**
