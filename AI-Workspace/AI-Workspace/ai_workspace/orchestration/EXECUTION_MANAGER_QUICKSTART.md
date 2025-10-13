# ExecutionManager Quick Start Guide

## 5-Minute Quick Start

### Installation

```python
# ExecutionManager is part of ai_workspace.orchestration
from ai_workspace.orchestration import ExecutionManager, Task, TaskGraph
from ai_workspace.runtime import InMemoryMessageBroker, ContextStore
```

### Minimal Example

```python
import asyncio

async def quick_start():
    # 1. Setup infrastructure
    broker = InMemoryMessageBroker()
    await broker.start()

    context = ContextStore(":memory:")

    # 2. Create tasks
    graph = TaskGraph(tasks=[
        Task(
            id="task1",
            name="First Task",
            agent_id="my-agent",
            dependencies=[],
            estimated_effort=10,  # minutes
        )
    ])

    # 3. Execute
    manager = ExecutionManager(broker, context, agent_selector)
    result = await manager.execute_graph(graph)

    # 4. Check results
    print(f"Success: {result.success}")

    await broker.stop()

asyncio.run(quick_start())
```

## Common Patterns

### Pattern 1: Sequential Tasks

```python
# Create chain: A → B → C
tasks = [
    Task(id="a", name="Task A", agent_id="agent-1", dependencies=[]),
    Task(id="b", name="Task B", agent_id="agent-2", dependencies=["a"]),
    Task(id="c", name="Task C", agent_id="agent-3", dependencies=["b"]),
]

graph = TaskGraph(tasks=tasks)
result = await manager.execute_graph(graph)
```

### Pattern 2: Parallel Tasks

```python
# Create fork: A → [B, C, D]
tasks = [
    Task(id="a", name="Init", agent_id="init-agent", dependencies=[]),
    Task(id="b", name="Branch 1", agent_id="agent-1", dependencies=["a"]),
    Task(id="c", name="Branch 2", agent_id="agent-2", dependencies=["a"]),
    Task(id="d", name="Branch 3", agent_id="agent-3", dependencies=["a"]),
]

graph = TaskGraph(tasks=tasks)
result = await manager.execute_graph(graph)

# Execution: Wave 1: [A], Wave 2: [B, C, D] in parallel
```

### Pattern 3: Diamond (Fork-Join)

```python
# Create diamond: A → [B, C] → D
tasks = [
    Task(id="a", name="Start", agent_id="agent-1", dependencies=[]),
    Task(id="b", name="Left", agent_id="agent-2", dependencies=["a"]),
    Task(id="c", name="Right", agent_id="agent-3", dependencies=["a"]),
    Task(id="d", name="Join", agent_id="agent-4", dependencies=["b", "c"]),
]

graph = TaskGraph(tasks=tasks)
result = await manager.execute_graph(graph)

# Execution: Wave 1: [A], Wave 2: [B, C], Wave 3: [D]
```

### Pattern 4: Complex Pipeline

```python
# Multi-stage pipeline with parallelism
tasks = [
    # Stage 1: Initialize
    Task(id="init", name="Initialize", agent_id="init-agent", dependencies=[]),

    # Stage 2: Parallel processing
    Task(id="frontend", name="Build Frontend", agent_id="frontend-agent", dependencies=["init"]),
    Task(id="backend", name="Build Backend", agent_id="backend-agent", dependencies=["init"]),
    Task(id="database", name="Setup DB", agent_id="db-agent", dependencies=["init"]),

    # Stage 3: Integration
    Task(id="integrate", name="Integrate", agent_id="integration-agent",
         dependencies=["frontend", "backend", "database"]),

    # Stage 4: Final steps
    Task(id="test", name="Test", agent_id="test-agent", dependencies=["integrate"]),
    Task(id="deploy", name="Deploy", agent_id="deploy-agent", dependencies=["test"]),
]

graph = TaskGraph(tasks=tasks)
result = await manager.execute_graph(graph)
```

## Monitoring Execution

### Real-Time Progress

```python
# Start execution in background
execution_task = asyncio.create_task(manager.execute_graph(graph))

# Monitor progress
while not execution_task.done():
    if manager.active_executions:
        exec_id = list(manager.active_executions.keys())[0]
        status = manager.get_execution_status(exec_id)

        print(f"Progress: {status['progress_percent']:.1f}%")
        print(f"Running: {status['running_tasks']}")
        print(f"Completed: {status['completed_tasks']}/{status['total_tasks']}")

    await asyncio.sleep(1)

# Get result
result = await execution_task
```

### Cancellation

```python
# Start execution
execution_task = asyncio.create_task(manager.execute_graph(graph))

# Wait a bit
await asyncio.sleep(2)

# Cancel if needed
exec_id = list(manager.active_executions.keys())[0]
cancelled = await manager.cancel_execution(exec_id)

print(f"Cancelled: {cancelled}")
```

## Error Handling

### Check for Failures

```python
result = await manager.execute_graph(graph)

if result.success:
    print(f"All {len(result.completed_tasks)} tasks completed!")
else:
    print(f"Execution failed: {result.error_message}")
    print(f"Failed tasks: {result.failed_tasks}")

    # Partial results still available
    print(f"Completed tasks: {result.completed_tasks}")
```

### Handle Timeouts

```python
# Set realistic estimated_effort to avoid timeouts
task = Task(
    id="long-task",
    name="Long Running Task",
    agent_id="worker",
    dependencies=[],
    estimated_effort=60,  # 60 minutes (timeout will be 120 min)
)

# Timeout = estimated_effort * 60 * 2 seconds
```

## Task Configuration

### Task Parameters

```python
task = Task(
    id="my-task",                    # Unique ID (auto-generated if omitted)
    name="Build Application",         # Human-readable name
    description="Compile and build",  # Detailed description
    agent_id="build-agent",          # Agent to execute task
    dependencies=["setup-task"],     # List of task IDs
    estimated_effort=30,             # Minutes (affects timeout)
    complexity=3,                    # 1-5 scale
    priority=7,                      # 0-10 (higher = more urgent)
    metadata={                       # Custom data
        "build_type": "release",
        "target": "production",
    }
)
```

### Task Status Lifecycle

```
PENDING → READY → RUNNING → COMPLETED
                         ↘ FAILED
                         ↘ CANCELLED
```

## Result Processing

### Access Artifacts

```python
result = await manager.execute_graph(graph)

for task_id in result.completed_tasks:
    artifact = result.artifacts.get(task_id)
    if artifact:
        print(f"Task {task_id} output: {artifact}")
```

### Execution Metrics

```python
result = await manager.execute_graph(graph)

print(f"Total Duration: {result.total_duration:.2f}s")
print(f"Success Rate: {len(result.completed_tasks) / len(graph.tasks) * 100:.1f}%")
print(f"Failed Tasks: {len(result.failed_tasks)}")
```

## Agent Implementation

### Minimal Agent

```python
class MyAgent:
    def __init__(self, agent_id: str, broker: InMemoryMessageBroker):
        self.agent_id = agent_id
        self.broker = broker

    async def start(self):
        """Start processing messages."""
        while True:
            message = await self.broker.receive(self.agent_id, timeout=30)

            if message.payload.get("action") == "execute_task":
                # Do work
                result = self.process_task(message.payload["params"]["task"])

                # Send response
                response = AgentMessage.create_response(
                    from_agent=self.agent_id,
                    to_agent=message.from_agent,
                    correlation_id=message.correlation_id,
                    status="success",
                    result=result
                )

                await self.broker.send(response)

    def process_task(self, task_data):
        """Process task and return result."""
        # Your task logic here
        return {"output": "Task completed"}
```

## Best Practices

### 1. Task Granularity

```python
# ✅ Good: Tasks of 10-60 minutes
Task(name="Build Module", estimated_effort=30)

# ❌ Bad: Tasks too small (overhead) or too large (risky)
Task(name="Quick Check", estimated_effort=1)  # Too small
Task(name="Deploy Everything", estimated_effort=480)  # Too large
```

### 2. Dependency Design

```python
# ✅ Good: Minimal dependencies, maximum parallelism
tasks = [
    Task(id="init", dependencies=[]),
    Task(id="a", dependencies=["init"]),  # Can run in parallel
    Task(id="b", dependencies=["init"]),  # with 'a'
    Task(id="c", dependencies=["init"]),  # and 'b'
]

# ❌ Bad: Unnecessary sequential dependencies
tasks = [
    Task(id="a", dependencies=[]),
    Task(id="b", dependencies=["a"]),  # Could be parallel
    Task(id="c", dependencies=["b"]),  # with 'a'
]
```

### 3. Error Resilience

```python
# ✅ Good: Design for partial failures
# Independent branches can fail without affecting others
tasks = [
    Task(id="critical", ...),
    Task(id="optional-a", dependencies=["critical"]),  # Can fail
    Task(id="optional-b", dependencies=["critical"]),  # Independently
]

# ❌ Bad: Single point of failure
tasks = [
    Task(id="risky", ...),  # If this fails, everything fails
    Task(id="depends", dependencies=["risky"]),
]
```

### 4. Monitoring

```python
# ✅ Good: Monitor long-running executions
import logging
logging.basicConfig(level=logging.INFO)

result = await manager.execute_graph(graph)

# ❌ Bad: No monitoring for long workflows
# (Task might timeout without visibility)
```

## Troubleshooting

### Issue: Tasks timeout

```python
# Solution: Increase estimated_effort
task = Task(
    name="Slow Task",
    estimated_effort=60,  # Was 10, now 60
    ...
)
```

### Issue: No parallelism

```python
# Check dependencies - ensure tasks can run in parallel
from ai_workspace.orchestration import DependencyResolver

resolver = DependencyResolver(graph)
batches = resolver.find_parallel_batches()

print(f"Execution plan: {len(batches)} waves")
for i, batch in enumerate(batches, 1):
    print(f"  Wave {i}: {[t.name for t in batch]}")
```

### Issue: Execution hangs

```python
# Validate graph for cycles
from ai_workspace.orchestration import DependencyResolver

resolver = DependencyResolver(graph)
try:
    resolver.validate_graph()
    print("Graph is valid")
except Exception as e:
    print(f"Graph error: {e}")
```

## Next Steps

1. **Read Full Documentation**: See `README_EXECUTION_MANAGER.md`
2. **Run Examples**: Check `examples/execution_manager_example.py`
3. **Run Tests**: Execute `test_execution_manager.py`
4. **Build Your Workflow**: Start with simple graph, add complexity

## Resources

- **API Reference**: `README_EXECUTION_MANAGER.md`
- **Examples**: `examples/execution_manager_example.py`
- **Tests**: `test_execution_manager.py`
- **Source Code**: `ai_workspace/orchestration/execution_manager.py`

## Support

For issues or questions:
1. Check logs with `logging.DEBUG`
2. Validate graph structure
3. Review test examples
4. Read full documentation

---

**Quick Reference Card**

```python
# Initialize
broker = InMemoryMessageBroker()
await broker.start()
context = ContextStore()
manager = ExecutionManager(broker, context, selector)

# Create graph
graph = TaskGraph(tasks=[...])

# Execute
result = await manager.execute_graph(graph)

# Check
if result.success:
    print(f"Completed: {len(result.completed_tasks)}")
else:
    print(f"Failed: {result.error_message}")

# Cleanup
await broker.stop()
```
