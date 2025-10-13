# ExecutionManager - Autonomous Multi-Agent Task Orchestration

## Overview

The `ExecutionManager` coordinates autonomous execution of multi-agent workflows with wave-based parallel processing, automatic dependency resolution, fault tolerance, and comprehensive state tracking.

## Features

### Core Capabilities

- **Wave-Based Parallel Execution**: Automatically identifies and executes independent tasks concurrently
- **Dependency Resolution**: Validates task graphs and ensures correct execution order
- **Fault Tolerance**: Handles task failures gracefully without blocking other tasks
- **State Tracking**: Real-time monitoring of execution progress
- **Result Collection**: Aggregates artifacts and outputs from all tasks
- **Cancellation Support**: Ability to cancel running executions

### Architecture

```
ExecutionManager
├── Graph Validation (DependencyResolver)
├── Batch Computation (Parallel Waves)
├── Agent Communication (Message Broker)
├── State Management (Context Store)
└── Result Aggregation (ExecutionResult)
```

## Usage

### Basic Example

```python
import asyncio
from ai_workspace.orchestration import (
    ExecutionManager,
    Task,
    TaskGraph,
)
from ai_workspace.runtime import (
    InMemoryMessageBroker,
    ContextStore,
)

async def main():
    # Initialize infrastructure
    broker = InMemoryMessageBroker()
    await broker.start()

    context = ContextStore()
    selector = MockAgentSelector()  # Your agent selector

    manager = ExecutionManager(broker, context, selector)

    # Create task graph
    graph = TaskGraph(tasks=[
        Task(
            id="setup",
            name="Setup Environment",
            agent_id="setup-agent",
            dependencies=[],
            estimated_effort=10,  # minutes
        ),
        Task(
            id="build",
            name="Build Application",
            agent_id="build-agent",
            dependencies=["setup"],
            estimated_effort=30,
        ),
    ])

    # Execute graph
    result = await manager.execute_graph(graph)

    # Check results
    if result.success:
        print(f"Completed {len(result.completed_tasks)} tasks")
        print(f"Duration: {result.total_duration:.2f}s")
    else:
        print(f"Failed: {result.error_message}")

    await broker.stop()

asyncio.run(main())
```

### Parallel Execution

```python
# Create diamond pattern: t1 -> [t2, t3] -> t4
graph = TaskGraph(tasks=[
    Task(id="t1", name="Init", agent_id="agent-1", dependencies=[]),
    Task(id="t2", name="Branch A", agent_id="agent-2", dependencies=["t1"]),
    Task(id="t3", name="Branch B", agent_id="agent-3", dependencies=["t1"]),
    Task(id="t4", name="Merge", agent_id="agent-4", dependencies=["t2", "t3"]),
])

result = await manager.execute_graph(graph)

# Execution plan:
# Wave 1: [t1]
# Wave 2: [t2, t3]  <- Parallel execution
# Wave 3: [t4]
```

### Monitoring Execution

```python
# Start execution in background
execution_task = asyncio.create_task(manager.execute_graph(graph))

# Monitor progress
while not execution_task.done():
    if manager.active_executions:
        execution_id = list(manager.active_executions.keys())[0]
        status = manager.get_execution_status(execution_id)

        print(f"Progress: {status['progress_percent']:.1f}%")
        print(f"Active tasks: {len(status['active_task_ids'])}")

    await asyncio.sleep(1)

result = await execution_task
```

### Cancellation

```python
# Start execution
execution_task = asyncio.create_task(manager.execute_graph(graph))

# Get execution ID
await asyncio.sleep(0.1)
execution_id = list(manager.active_executions.keys())[0]

# Cancel if needed
cancelled = await manager.cancel_execution(execution_id)
print(f"Cancelled: {cancelled}")
```

## API Reference

### ExecutionManager

#### `__init__(broker, context, selector)`

Initialize execution manager.

**Parameters:**
- `broker` (InMemoryMessageBroker): Message broker for agent communication
- `context` (ContextStore): Context store for state management
- `selector` (AgentSelector): Agent selector for task assignment

**Raises:**
- `ValueError`: If any argument is None

#### `execute_graph(graph: TaskGraph) -> ExecutionResult`

Execute task graph with wave-based parallel execution.

**Parameters:**
- `graph` (TaskGraph): Task graph to execute

**Returns:**
- `ExecutionResult`: Execution result with success status and artifacts

**Raises:**
- `ValueError`: If graph is invalid

**Process:**
1. Validates graph with DependencyResolver
2. Computes parallel execution batches
3. Executes each batch (wave) in order
4. Waits for all tasks in batch to complete
5. Collects results and artifacts

#### `get_execution_status(execution_id: str) -> Dict[str, Any]`

Get current execution status.

**Parameters:**
- `execution_id` (str): Execution identifier

**Returns:**
- Dictionary with keys:
  - `execution_id` (str): Execution identifier
  - `status` (str): Execution status ("running" or "not_found")
  - `total_tasks` (int): Total number of tasks
  - `completed_tasks` (int): Number of completed tasks
  - `failed_tasks` (int): Number of failed tasks
  - `running_tasks` (int): Number of running tasks
  - `pending_tasks` (int): Number of pending tasks
  - `progress_percent` (float): Progress percentage (0-100)
  - `active_task_ids` (List[str]): IDs of running tasks
  - `completed_task_ids` (List[str]): IDs of completed tasks
  - `failed_task_ids` (List[str]): IDs of failed tasks

#### `cancel_execution(execution_id: str) -> bool`

Cancel running execution.

**Parameters:**
- `execution_id` (str): Execution to cancel

**Returns:**
- `bool`: True if cancelled successfully

### ExecutionResult

Result of task execution with comprehensive metrics.

**Attributes:**
- `success` (bool): True if all tasks completed successfully
- `execution_id` (str): Unique execution identifier
- `completed_tasks` (List[str]): List of successfully completed task IDs
- `failed_tasks` (List[str]): List of failed task IDs
- `artifacts` (Dict[str, Any]): Execution outputs and results
- `total_duration` (float): Total execution time in seconds
- `error_message` (Optional[str]): Error message if execution failed

## Wave-Based Execution

The ExecutionManager uses a wave-based execution model:

### Wave Definition

A **wave** is a batch of tasks that:
1. Have all dependencies satisfied by previous waves
2. Can execute in parallel with other tasks in the same wave

### Execution Algorithm

```
1. Validate graph (detect cycles, validate dependencies)
2. Compute parallel batches using DependencyResolver
3. For each wave:
   a. Execute all tasks in parallel using asyncio.gather()
   b. Wait for all tasks to complete
   c. Collect results and update state
4. Aggregate results into ExecutionResult
```

### Example Execution Plan

```python
# Graph: t1 -> [t2, t3, t4] -> [t5, t6] -> t7
#
# Execution Plan:
# Wave 1: [t1]              (1 task, duration: 10min)
# Wave 2: [t2, t3, t4]      (3 tasks parallel, duration: max(t2, t3, t4))
# Wave 3: [t5, t6]          (2 tasks parallel, duration: max(t5, t6))
# Wave 4: [t7]              (1 task, duration: 15min)
```

### Parallelism Benefits

- **Reduced Total Time**: Parallel waves execute concurrently
- **Resource Efficiency**: Multiple agents work simultaneously
- **Automatic Optimization**: No manual coordination needed
- **Scalability**: Handles large graphs with many parallel tasks

## Agent Communication Protocol

### Request Message

ExecutionManager sends REQUEST messages to agents:

```python
{
    "from_agent": "execution-manager",
    "to_agent": "agent-id",
    "message_type": "REQUEST",
    "payload": {
        "action": "execute_task",
        "params": {
            "task": {
                "id": "task-id",
                "name": "Task Name",
                "description": "Task description",
                "metadata": {...},
                ...
            }
        }
    },
    "priority": 5,
    "correlation_id": "abc123...",
}
```

### Response Message

Agents respond with RESPONSE messages:

**Success Response:**
```python
{
    "from_agent": "agent-id",
    "to_agent": "execution-manager",
    "message_type": "RESPONSE",
    "payload": {
        "status": "success",
        "result": {
            "output": "Task output",
            "artifacts": {...},
        }
    },
    "correlation_id": "abc123...",
}
```

**Failure Response:**
```python
{
    "from_agent": "agent-id",
    "to_agent": "execution-manager",
    "message_type": "RESPONSE",
    "payload": {
        "status": "failure",
        "error": "Error message"
    },
    "correlation_id": "abc123...",
}
```

### Timeout Handling

- **Timeout Duration**: `task.estimated_effort * 60 * 2` seconds (2x buffer)
- **Behavior**: Task marked as FAILED if timeout occurs
- **Other Tasks**: Continue executing (fault isolation)

## State Management

### Task State Tracking

ExecutionManager records task state in ContextStore:

```python
context.set_task_state(
    task_id="task-id",
    status="running",  # pending, running, completed, failed
    progress=0.5,      # 0.0 to 1.0
    metadata={
        "execution_id": "exec-123",
        "agent_id": "agent-1",
        "result": {...},
        "error": None,
    }
)
```

### State Persistence

- **Database**: SQLite (context.db)
- **Caching**: TTL cache for performance
- **History**: Complete execution history maintained
- **Recovery**: State survives manager restarts

## Error Handling

### Fault Tolerance Strategy

1. **Task Isolation**: One task failure doesn't affect others
2. **Graceful Degradation**: Execution continues with remaining tasks
3. **Result Collection**: Partial results available even if some tasks fail
4. **Error Reporting**: Detailed error messages in ExecutionResult

### Failure Scenarios

| Scenario | Behavior |
|----------|----------|
| Task timeout | Mark FAILED, continue with other tasks |
| Agent unresponsive | Timeout → FAILED after 2x estimated effort |
| Invalid response | Mark FAILED, log error |
| Exception in task | Mark FAILED, continue execution |
| Dependency failure | Dependent tasks can still execute if other deps satisfied |

### Future Enhancements

- **Retry Logic**: Automatic retry with alternative agents
- **Dynamic Replanning**: Route around failures
- **Compensation**: Rollback partial failures
- **Circuit Breakers**: Prevent cascade failures

## Performance Optimization

### Parallelism

- **Concurrent Execution**: Uses asyncio.gather() for parallel tasks
- **Wave Batching**: Minimizes coordination overhead
- **Non-Blocking**: Asynchronous throughout

### Resource Management

- **Connection Pooling**: Message broker queues
- **Memory Efficiency**: Streaming results, not buffering
- **CPU Optimization**: Parallel task execution

### Scalability

- **Task Count**: Tested with 100+ tasks
- **Agent Count**: Supports 50+ agents
- **Graph Complexity**: Handles deep dependency trees
- **Execution Time**: Efficient for long-running workflows

## Testing

### Unit Tests

```bash
python test_execution_manager.py
```

**Test Coverage:**
- ✓ Simple linear execution
- ✓ Parallel execution (diamond pattern)
- ✓ Task failure handling
- ✓ Execution status tracking
- ✓ Execution cancellation
- ✓ Complex graph execution

### Example Tests

```python
async def test_parallel_execution():
    """Test parallel task execution."""
    # Create diamond graph: t1 -> [t2, t3] -> t4
    graph = TaskGraph(tasks=[...])

    result = await manager.execute_graph(graph)

    assert result.success
    assert len(result.completed_tasks) == 4
    assert result.total_duration < sequential_duration
```

## Integration

### With Other Components

**DependencyResolver:**
```python
# Automatic integration - ExecutionManager uses DependencyResolver internally
resolver = DependencyResolver(graph)
batches = resolver.find_parallel_batches()
```

**ContextStore:**
```python
# State tracking integration
task_state = context.get_task_state("task-id")
print(task_state["status"])  # "completed"
```

**MessageBroker:**
```python
# Agent communication integration
await broker.send(message)
response = await broker.receive("execution-manager")
```

### Custom Agent Selector

Implement custom agent selection logic:

```python
class SmartAgentSelector:
    def select_agent(self, task: Task) -> str:
        """Select best agent based on performance, availability."""
        # Custom selection logic
        return best_agent_id

manager = ExecutionManager(broker, context, SmartAgentSelector())
```

## Best Practices

### Task Design

1. **Granularity**: Break work into tasks of 10-60 minutes
2. **Dependencies**: Minimize dependencies for better parallelism
3. **Idempotency**: Design tasks to be safely retryable
4. **Metadata**: Include context for agents in task.metadata

### Graph Construction

1. **Validation**: Always validate graphs before execution
2. **Testing**: Test dependency resolution separately
3. **Complexity**: Keep graphs under 100 tasks for optimal performance
4. **Documentation**: Document task purposes and dependencies

### Error Handling

1. **Timeouts**: Set realistic estimated_effort values
2. **Monitoring**: Track execution status regularly
3. **Logging**: Enable debug logging for troubleshooting
4. **Recovery**: Design for partial failures

### Performance

1. **Parallelism**: Maximize independent task branches
2. **Resource Limits**: Consider agent capacity
3. **Batching**: Group related tasks when possible
4. **Caching**: Use context store caching effectively

## Examples

See `examples/execution_manager_example.py` for comprehensive examples:

1. **Linear Workflow**: Sequential task execution
2. **Parallel Workflow**: Concurrent task execution
3. **Monitoring**: Real-time progress tracking

## Troubleshooting

### Common Issues

**Issue: Tasks timing out**
- Solution: Increase estimated_effort or optimize agent processing

**Issue: Low parallelism**
- Solution: Reduce dependencies between tasks

**Issue: Execution hangs**
- Solution: Check for circular dependencies, validate graph

**Issue: Failed tasks**
- Solution: Check agent logs, verify agent availability

### Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Detailed execution logs
logger = logging.getLogger("ai_workspace.orchestration.execution_manager")
logger.setLevel(logging.DEBUG)
```

## Future Roadmap

### Planned Features

- [ ] **Retry Logic**: Automatic retry with alternative agents
- [ ] **Dynamic Replanning**: Adapt to failures in real-time
- [ ] **Priority Scheduling**: Execute high-priority tasks first
- [ ] **Resource Quotas**: Limit concurrent tasks per agent
- [ ] **Progress Callbacks**: Real-time progress notifications
- [ ] **Checkpoint/Resume**: Save and resume long-running executions
- [ ] **Distributed Execution**: Multi-node execution support
- [ ] **Cost Optimization**: Minimize total execution cost

### Enhancement Ideas

- Smart agent selection based on historical performance
- Automatic task decomposition for oversized tasks
- Predictive completion time estimation
- Resource usage analytics and optimization
- Integration with workflow scheduling systems

## License

Part of AI Workspace v1.2.0 - See project LICENSE for details.
