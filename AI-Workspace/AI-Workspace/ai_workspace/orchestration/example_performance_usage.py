#!/usr/bin/env python3
"""
Example usage of PerformanceTracker for agent selection.

Demonstrates how to:
1. Track agent performance over time
2. Get performance metrics for decision-making
3. Select best agents based on historical data
"""

from datetime import datetime, timedelta
from ai_workspace.orchestration.performance_tracker import PerformanceTracker


def example_usage() -> None:
    """Demonstrate PerformanceTracker usage."""
    tracker = PerformanceTracker()

    # Scenario: Track multiple agents performing API development tasks
    print("Tracking agent performance for API development...")

    # Simulate 3 agents performing tasks with different success rates
    agents = [
        {"id": "backend-developer", "success_rate": 0.9, "avg_time": 300},
        {"id": "fullstack-developer", "success_rate": 0.8, "avg_time": 400},
        {"id": "junior-developer", "success_rate": 0.6, "avg_time": 600},
    ]

    # Record historical executions
    base_time = datetime.now() - timedelta(days=20)

    for day in range(20):
        for agent in agents:
            # Skip some days randomly to simulate real usage
            if (day + hash(agent["id"])) % 3 == 0:
                continue

            success = (hash(f"{agent['id']}-{day}") % 100) < (
                agent["success_rate"] * 100
            )
            duration = agent["avg_time"] + (hash(f"{day}") % 60)

            start_time = base_time + timedelta(days=day, hours=9)
            end_time = start_time + timedelta(seconds=duration)

            tracker.record_execution(
                agent_id=agent["id"],
                task_id=f"task-{day}-{agent['id']}",
                task_type="api_development",
                task_name=f"API Development Task {day}",
                start_time=start_time,
                end_time=end_time,
                success=success,
                cpu_percent=40.0 + (hash(f"{day}") % 30),
                memory_mb=200.0 + (hash(f"{day}") % 100),
                output_quality_score=0.7 if success else 0.3,
                user_corrections=0 if success else 2,
            )

    print("\n" + "=" * 60)
    print("AGENT PERFORMANCE ANALYSIS")
    print("=" * 60)

    # Get metrics for each agent
    for agent in agents:
        metrics = tracker.get_agent_metrics(
            agent["id"], task_type="api_development", days=30
        )

        print(f"\nAgent: {metrics.agent_id}")
        print(f"  Total executions: {metrics.total_executions}")
        print(f"  Success rate: {metrics.success_rate:.1%}")
        print(f"  Successful: {metrics.successful_executions}")
        print(f"  Failed: {metrics.failed_executions}")
        print(f"  Avg duration: {metrics.avg_duration_seconds:.1f}s")
        print(f"  Avg quality score: {metrics.avg_quality_score:.2f}")
        print(f"  Avg CPU usage: {metrics.avg_cpu_percent:.1f}%")
        print(f"  Avg memory: {metrics.avg_memory_mb:.1f} MB")

    # Get best agents for task type
    print("\n" + "=" * 60)
    print("BEST AGENTS FOR API DEVELOPMENT")
    print("=" * 60)

    best_agents = tracker.get_best_agent_for_task("api_development", top_n=3)

    for rank, agent_id in enumerate(best_agents, 1):
        metrics = tracker.get_agent_metrics(
            agent_id, task_type="api_development", days=30
        )
        print(
            f"{rank}. {agent_id}: "
            f"{metrics.success_rate:.1%} success, "
            f"{metrics.avg_duration_seconds:.0f}s avg time"
        )

    # Get recent failures for debugging
    print("\n" + "=" * 60)
    print("RECENT FAILURES (Last 7 days)")
    print("=" * 60)

    failures = tracker.get_recent_failures(days=7)
    print(f"\nTotal failures: {len(failures)}")

    for failure in failures[:5]:  # Show first 5
        print(f"\n  Task: {failure['task_name']}")
        print(f"  Agent: {failure['agent_id']}")
        print(f"  Time: {failure['start_time']}")
        print(f"  Duration: {failure['duration_seconds']}s")

    # Overall statistics
    print("\n" + "=" * 60)
    print("OVERALL STATISTICS")
    print("=" * 60)

    all_stats = tracker.get_all_agent_stats(days=30)

    for stat in all_stats:
        print(
            f"\n{stat['agent_id']}:"
            f" {stat['total_executions']} tasks, "
            f"{stat['success_rate']:.1%} success rate, "
            f"{stat['task_types_handled']} task types"
        )

    # Cleanup
    tracker.close()

    print("\n" + "=" * 60)
    print("Recommendation: Use 'backend-developer' for critical API tasks")
    print("=" * 60)


if __name__ == "__main__":
    example_usage()
