# AI Workspace Runtime System

**Version**: 1.0.0
**Status**: Phase 1 Complete (Week 1)

The AI Workspace Runtime System provides a robust, production-ready execution environment for AI agents with process isolation, resource enforcement, and high-performance process pooling.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     AgentRuntime (Orchestrator)                  │
│  ┌────────────┐  ┌──────────────┐  ┌───────────────────────┐  │
│  │   Agent    │  │   Process    │  │    Resource           │  │
│  │  Factory   │  │     Pool     │  │   Enforcer            │  │
│  └────────────┘  └──────────────┘  └───────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
         │                  │                      │
         │                  │                      │
         ▼                  ▼                      ▼
┌─────────────────┐  ┌──────────────┐  ┌───────────────────────┐
│ Agent Markdown  │  │   Worker     │  │   Platform-Specific   │
│    Parser       │  │  Processes   │  │   Limit Enforcement   │
│                 │  │              │  │                       │
│ - Frontmatter   │  │ - Pre-spawn  │  │ - Linux: cgroups v2   │
│ - Tool perms    │  │ - Warmup     │  │ - macOS: setrlimit    │
│ - Metadata      │  │ - Reuse      │  │ - Windows: polling    │
└─────────────────┘  └──────────────┘  └───────────────────────┘
         │                  │                      │
         │                  │                      │
         ▼                  ▼                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                     IPC Communication Layer                      │
│  ┌────────────┐  ┌──────────────┐  ┌───────────────────────┐  │
│  │  Command   │  │    Result    │  │   Logs/Metrics/       │  │
│  │  Messages  │  │   Messages   │  │   Artifacts           │  │
│  └────────────┘  └──────────────┘  └───────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. AgentRuntime

The main execution orchestrator that coordinates all runtime operations.

**Responsibilities**:
- Agent discovery and validation
- Process lifecycle management
- Resource limit enforcement
- Result collection and error handling
- Graceful shutdown and cleanup

**Key Features**:
- **Process Isolation**: Each agent runs in a separate process
- **Fault Tolerance**: Agent crashes don't affect orchestrator
- **Resource Limits**: Memory, CPU, and timeout enforcement
- **Automatic Cleanup**: atexit handlers ensure no orphaned processes

**Example**:
```python
from ai_workspace.runtime import AgentRuntime, Task, ResourceLimits

# Initialize runtime
runtime = AgentRuntime()
runtime.start()

# Execute agent with custom limits
task = Task(
    action="write_function",
    spec="Create Fibonacci calculator with memoization"
)
limits = ResourceLimits(memory_mb=1024, timeout_seconds=300)
result = runtime.execute("python-pro", task, limits)

# Check results
if result.success:
    print(f"Output: {result.output}")
    print(f"Duration: {result.duration_seconds:.2f}s")
    print(f"Memory: {result.resource_usage['memory_mb']:.1f}MB")
else:
    print(f"Error: {result.error}")

# Cleanup
runtime.stop()
```

### 2. AgentFactory

Dynamic agent instantiation from markdown specifications.

**Responsibilities**:
- Parse agent markdown files (YAML frontmatter + content)
- Extract metadata (name, description, tools, model)
- Validate agent specifications
- Cache agents for performance
- List all available agents

**Supported Frontmatter**:
```yaml
---
name: python-pro
description: Expert Python developer specializing in clean code
tools: Read, Write, Bash, MultiEdit
model: claude-sonnet-4
---
# Agent content here...
```

**Example**:
```python
from ai_workspace.runtime import AgentFactory

factory = AgentFactory()

# Create agent from markdown
agent = factory.create_agent("python-pro")
print(agent.metadata.name)        # "python-pro"
print(agent.metadata.tools)       # ["Read", "Write", "Bash", "MultiEdit"]
print(agent.get_prompt()[:100])   # First 100 chars of agent content

# Check tool permissions
if agent.has_tool("Bash"):
    print("Agent can execute shell commands")

# List all agents
all_agents = factory.list_all_agents()
print(f"Found {len(all_agents)} agents")
```

### 3. AgentProcessPool

Pre-spawned worker processes to reduce startup latency.

**Objective**: Reduce startup time from ~5s → <500ms

**Strategy**:
- Maintain pool of warm worker processes
- Pre-initialize Python environment
- Reuse processes for multiple tasks (future)
- Background pool maintenance

**Current Status**: On-demand spawning (pre-spawning planned for Phase 2)

**Example**:
```python
# Process pool is managed automatically by AgentRuntime
runtime = AgentRuntime()
runtime.start()  # Initializes pool

# Pool automatically provides workers for execution
result = runtime.execute("python-pro", task)

runtime.stop()  # Cleans up all pool workers
```

### 4. ResourceEnforcer

Cross-platform resource limit enforcement.

**Platform Support**:

| Platform | Implementation | Method | Enforcement |
|----------|---------------|---------|-------------|
| Linux | `CgroupEnforcer` | cgroups v2 | Kernel-level (best) |
| macOS | `RlimitEnforcer` | setrlimit | OS-level (good) |
| Windows | `PollingEnforcer` | psutil | Monitor + terminate |

**Limits Enforced**:
- **Memory**: Hard limit in MB (terminates on exceed)
- **CPU**: Percentage of one core (throttles or warns)
- **Timeout**: Maximum execution time (SIGTERM → SIGKILL)

**Example**:
```python
from ai_workspace.runtime import ResourceEnforcer, ResourceLimits
from multiprocessing import Process

enforcer = ResourceEnforcer()
limits = ResourceLimits(
    memory_mb=512,       # 512 MB max
    cpu_percent=50,      # 50% CPU max
    timeout_seconds=60   # 1 minute max
)

# Enforcement runs concurrently with process
import asyncio
async def enforce_limits(process):
    await enforcer.enforce(process, limits)

# Process will be terminated if limits exceeded
```

### 5. IPC Protocol

Type-safe inter-process communication with structured messages.

**Message Types**:
- `command`: Send execution commands to worker
- `result`: Receive execution results from worker
- `log`: Receive log messages (DEBUG, INFO, WARNING, ERROR)
- `metric`: Receive performance metrics
- `artifact`: Receive generated files/data
- `error`: Receive error information with traceback

**Example**:
```python
from ai_workspace.runtime import IPCMessage, IPCChannel
from multiprocessing import Queue

# Create communication channels
work_queue = Queue()
result_queue = Queue()
channel = IPCChannel(work_queue, result_queue)

# Send command to worker
await channel.send_command("execute", {"code": "print('hello')"})

# Receive single message
message = await channel.recv_message(timeout=5.0)
if message:
    print(f"Type: {message.type}")
    print(f"Payload: {message.payload}")

# Collect all messages
all_results = await channel.collect_all(timeout=10.0)
print(f"Logs: {len(all_results['logs'])}")
print(f"Metrics: {len(all_results['metrics'])}")
print(f"Artifacts: {len(all_results['artifacts'])}")
```

## Performance Characteristics

### Startup Latency

| Mode | Current | Target (Phase 2) |
|------|---------|------------------|
| Cold start | ~5s | ~5s |
| Warm pool | N/A | <500ms |
| Cache hit | <100ms | <50ms |

### Resource Usage

| Component | Memory | CPU |
|-----------|--------|-----|
| Runtime orchestrator | ~50 MB | <5% |
| Agent factory (cached) | ~10 MB | <1% |
| Process pool (4 workers) | ~200 MB | <10% |
| Per agent execution | Configurable (default 2GB) | Configurable (default 80%) |

### Scalability

- **Concurrent agents**: Up to `pool_size` (default: 4)
- **Total agents**: 156 agents supported
- **Agent cache**: Unlimited (LRU eviction planned)

## Configuration Options

### Runtime Configuration

```python
from pathlib import Path

runtime = AgentRuntime(
    workspace_path=Path("/path/to/.ai-workspace")
)
```

### Pool Configuration

```python
# Via environment variable
import os
os.environ["AI_WORKSPACE_POOL_SIZE"] = "8"

# Or pass to runtime (future)
runtime = AgentRuntime(pool_size=8)
```

### Resource Limits

#### Global Defaults

Create `E:\agents\.ai-workspace\config\agent_resources.yml`:

```yaml
memory_mb: 2048        # 2 GB
cpu_percent: 80        # 80% of one core
timeout_seconds: 600   # 10 minutes
```

#### Per-Execution Override

```python
limits = ResourceLimits(
    memory_mb=512,       # Override to 512 MB
    cpu_percent=50,      # Override to 50%
    timeout_seconds=120  # Override to 2 minutes
)
result = runtime.execute("python-pro", task, limits)
```

## Data Structures

### AgentMetadata

```python
@dataclass
class AgentMetadata:
    name: str                    # "python-pro"
    description: str             # "Expert Python developer..."
    tools: List[str]            # ["Read", "Write", "Bash"]
    model: str = "claude-sonnet-4"
    category: str = ""           # "Languages / scripting"
    file_path: str = ""          # "/path/to/python-pro.md"
```

### Task

```python
@dataclass
class Task:
    action: str                  # "write_function"
    spec: str                    # "Create Fibonacci calculator..."
    timeout_seconds: int = 600
    parameters: Optional[Dict[str, Any]] = None
```

### AgentResult

```python
@dataclass
class AgentResult:
    success: bool                # True if exit_code == 0
    exit_code: int               # 0 = success, non-zero = error
    output: str                  # Standard output
    error: str                   # Standard error
    artifacts: List[str] = []    # Created/modified files
    duration_seconds: float = 0.0
    resource_usage: Dict[str, float] = {}  # {"memory_mb": 45.2, "cpu_percent": 12.5}
```

### ResourceLimits

```python
@dataclass
class ResourceLimits:
    memory_mb: int = 2048        # Maximum memory (MB)
    cpu_percent: int = 80        # Maximum CPU (%)
    timeout_seconds: int = 600   # Maximum execution time (seconds)
```

## Error Handling

### Exception Hierarchy

```python
# Agent not found
try:
    result = runtime.execute("nonexistent-agent", task)
except AgentNotFoundError as e:
    print(f"Agent doesn't exist: {e}")

# Resource limit exceeded
try:
    result = runtime.execute("memory-hog", task, limits)
except ResourceLimitError as e:
    print(f"Agent exceeded limits: {e}")

# General execution error
try:
    result = runtime.execute("buggy-agent", task)
except ExecutionError as e:
    print(f"Execution failed: {e}")
```

### Graceful Degradation

The runtime gracefully handles:
- **Agent crashes**: Isolated in separate process
- **Timeout**: SIGTERM (5s grace) → SIGKILL
- **OOM**: Process terminated, error returned
- **Missing agents**: AgentNotFoundError with helpful message
- **Malformed markdown**: AgentParsingError with validation details

## Integration Examples

### CLI Integration

```python
import click
from ai_workspace.runtime import AgentRuntime, Task

@click.command()
@click.argument("agent_name")
@click.argument("task_spec")
def run_agent(agent_name: str, task_spec: str):
    """Execute an agent with a task."""
    runtime = AgentRuntime()
    runtime.start()

    try:
        task = Task(action="execute", spec=task_spec)
        result = runtime.execute(agent_name, task)

        click.echo(f"Success: {result.success}")
        click.echo(f"Output:\n{result.output}")
        if result.error:
            click.echo(f"Errors:\n{result.error}", err=True)
    finally:
        runtime.stop()
```

### Dashboard Integration

```python
from flask import Flask, jsonify, request
from ai_workspace.runtime import AgentRuntime, Task

app = Flask(__name__)
runtime = AgentRuntime()
runtime.start()

@app.route("/api/execute", methods=["POST"])
def execute_agent():
    data = request.json

    task = Task(
        action=data["action"],
        spec=data["spec"]
    )

    result = runtime.execute(data["agent_name"], task)

    return jsonify({
        "success": result.success,
        "output": result.output,
        "error": result.error,
        "duration": result.duration_seconds,
        "resources": result.resource_usage
    })

@app.route("/api/agents", methods=["GET"])
def list_agents():
    from ai_workspace.runtime import AgentFactory
    factory = AgentFactory()
    agents = factory.list_all_agents()

    return jsonify([
        {
            "name": a.name,
            "description": a.description,
            "category": a.category,
            "tools": a.tools
        }
        for a in agents
    ])
```

### Testing Integration

```python
import pytest
from ai_workspace.runtime import AgentRuntime, Task, ResourceLimits

@pytest.fixture
def runtime():
    runtime = AgentRuntime()
    runtime.start()
    yield runtime
    runtime.stop()

def test_agent_execution(runtime):
    task = Task(action="test", spec="Simple test task")
    result = runtime.execute("python-pro", task)

    assert result.success
    assert result.exit_code == 0
    assert result.duration_seconds < 10.0

def test_resource_limits(runtime):
    task = Task(action="memory_test", spec="Allocate 1GB memory")
    limits = ResourceLimits(memory_mb=512)  # Lower than task needs

    result = runtime.execute("memory-tester", task, limits)

    assert not result.success  # Should fail due to limits
    assert "memory" in result.error.lower()
```

## Development

### Running Tests

```bash
# Run all runtime tests
pytest tests/test_agent_runtime.py -v

# Run specific test
pytest tests/test_agent_runtime.py::TestAgentProcessPool -v

# Run with coverage
pytest tests/test_agent_runtime.py --cov=ai_workspace.runtime --cov-report=html
```

### Debugging

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

runtime = AgentRuntime()
runtime.start()
# Debug logs will show process lifecycle, resource enforcement, etc.
```

### Contributing

See Phase 1 implementation checklist in `BACKEND_PHASE1_PLAN.md`.

## Roadmap

### Phase 1 (Complete)
- [x] Process isolation and pooling
- [x] Resource enforcement (all platforms)
- [x] IPC protocol foundation
- [x] Agent factory with markdown parsing

### Phase 2 (Planned)
- [ ] Pre-spawned worker pool
- [ ] IPC message routing
- [ ] Tool permission enforcement
- [ ] Agent context injection

### Phase 3 (Planned)
- [ ] Multi-agent coordination
- [ ] Shared state management
- [ ] Performance monitoring dashboard
- [ ] Advanced caching strategies

## API Reference

For detailed API documentation:
- [Process Pool Documentation](../../docs/runtime/process-pool.md)
- [IPC Protocol Documentation](../../docs/runtime/ipc-protocol.md)
- [Resource Monitoring Documentation](../../docs/runtime/monitoring.md)

## Related Documentation

- [Week 1 Runtime Summary](../../EXECUTION_MANAGER_SUMMARY.md)
- [Architecture Decision Records](../../.ai-workspace/workspace/decisions/)
- [Phase 1 Implementation Plan](../../BACKEND_PHASE1_PLAN.md)

## License

Part of AI Workspace v1.2.0 - See top-level LICENSE file.
