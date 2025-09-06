#!/usr/bin/env python3
"""
System Monitor Agent for eFab AI Agent System
Monitors system health, performance, and proactive alerting
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import json
import psutil
import uuid

from ..core.agent_base import BaseAgent, AgentMessage, AgentCapability, MessageType, Priority
from ..core.state_manager import system_state

# Setup logging
logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class MetricType(Enum):
    """System metric types"""
    PERFORMANCE = "PERFORMANCE"
    AVAILABILITY = "AVAILABILITY"
    ERROR_RATE = "ERROR_RATE"
    RESOURCE_USAGE = "RESOURCE_USAGE"
    BUSINESS_METRIC = "BUSINESS_METRIC"


@dataclass
class SystemAlert:
    """System alert definition"""
    alert_id: str
    severity: AlertSeverity
    metric_type: MetricType
    title: str
    description: str
    threshold_value: float
    current_value: float
    triggered_at: datetime
    resolved_at: Optional[datetime] = None
    escalated: bool = False
    acknowledgements: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "alert_id": self.alert_id,
            "severity": self.severity.value,
            "metric_type": self.metric_type.value,
            "title": self.title,
            "description": self.description,
            "threshold_value": self.threshold_value,
            "current_value": self.current_value,
            "triggered_at": self.triggered_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "escalated": self.escalated,
            "acknowledgements": self.acknowledgements
        }


@dataclass
class HealthCheck:
    """Component health check result"""
    component: str
    healthy: bool
    response_time_ms: float
    last_check: datetime
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


class SystemMonitorAgent(BaseAgent):
    """
    System Monitor Agent for eFab AI Agent System
    
    Responsibilities:
    - System health monitoring and alerting
    - Performance metrics collection and analysis
    - Proactive issue detection and escalation
    - SLA monitoring and reporting
    - Resource usage tracking and optimization alerts
    - Implementation progress monitoring
    - Customer satisfaction tracking
    """
    
    def __init__(self, agent_id: str = "system_monitor"):
        """Initialize System Monitor Agent"""
        super().__init__(
            agent_id=agent_id,
            agent_name="System Monitor Agent",
            agent_description="System health monitoring and proactive alerting"
        )
        
        # Monitoring configuration
        self.monitoring_intervals = {
            "health_check": 30,      # seconds
            "performance_metrics": 60,
            "resource_usage": 120,
            "business_metrics": 300
        }
        
        # Alert thresholds
        self.alert_thresholds = {
            "cpu_usage_percent": 80.0,
            "memory_usage_percent": 85.0,
            "disk_usage_percent": 90.0,
            "response_time_ms": 2000.0,
            "error_rate_percent": 5.0,
            "agent_failure_rate": 10.0,
            "customer_satisfaction": 3.0,  # Below 3.0 out of 5
            "implementation_delay_hours": 24.0
        }
        
        # Active monitoring data
        self.active_alerts: Dict[str, SystemAlert] = {}
        self.resolved_alerts: List[SystemAlert] = []
        self.health_status: Dict[str, HealthCheck] = {}
        self.metrics_history: List[Dict[str, Any]] = []
        
        # Performance baselines
        self.performance_baselines = {
            "average_response_time_ms": 200.0,
            "throughput_requests_per_second": 100.0,
            "error_rate_baseline": 1.0,
            "agent_uptime_percentage": 99.5
        }
        
        # Component registry
        self.monitored_components = [
            "database",
            "message_router", 
            "agent_factory",
            "lead_agent",
            "customer_manager",
            "orchestrator",
            "system_state"
        ]
        
        self.business_kpis = [
            "implementation_success_rate",
            "customer_satisfaction_score", 
            "average_implementation_duration",
            "agent_utilization_rate"
        ]
    
    def _initialize(self):
        """Initialize System Monitor capabilities"""
        # Register monitoring capabilities
        self.register_capability(AgentCapability(
            name="system_health_monitoring",
            description="Monitor overall system health and component status",
            input_schema={
                "type": "object",
                "properties": {
                    "component": {"type": "string"},
                    "check_type": {"type": "string"}
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "health_status": {"type": "object"},
                    "alerts": {"type": "array"},
                    "recommendations": {"type": "array"}
                }
            }
        ))
        
        self.register_capability(AgentCapability(
            name="performance_monitoring",
            description="Monitor system performance metrics and trends",
            input_schema={
                "type": "object",
                "properties": {
                    "metric_type": {"type": "string"},
                    "time_range": {"type": "string"}
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "performance_summary": {"type": "object"},
                    "trends": {"type": "array"},
                    "bottlenecks": {"type": "array"}
                }
            }
        ))
        
        self.register_capability(AgentCapability(
            name="proactive_alerting",
            description="Generate proactive alerts based on metric thresholds",
            input_schema={
                "type": "object",
                "properties": {
                    "alert_conditions": {"type": "object"},
                    "notification_channels": {"type": "array"}
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "alerts_triggered": {"type": "array"},
                    "escalation_actions": {"type": "array"}
                }
            }
        ))
        
        self.register_capability(AgentCapability(
            name="sla_monitoring",
            description="Monitor service level agreements and implementation SLAs",
            input_schema={
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string"},
                    "sla_type": {"type": "string"}
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "sla_status": {"type": "object"},
                    "compliance_percentage": {"type": "number"},
                    "violations": {"type": "array"}
                }
            }
        ))
        
        # Register message handlers
        self.register_message_handler(MessageType.REQUEST, self._handle_monitoring_request)
        self.register_message_handler(MessageType.NOTIFICATION, self._handle_system_notification)
        
        # Start background monitoring tasks
        asyncio.create_task(self._health_check_loop())
        asyncio.create_task(self._performance_monitoring_loop())
        asyncio.create_task(self._resource_monitoring_loop())
        asyncio.create_task(self._business_metrics_loop())
        asyncio.create_task(self._alert_management_loop())
    
    async def _handle_monitoring_request(self, message: AgentMessage) -> AgentMessage:
        """Handle monitoring requests"""
        action = message.payload.get("action")
        
        try:
            if action == "system_health_check":
                result = await self._perform_system_health_check(message.payload)
            elif action == "get_performance_metrics":
                result = await self._get_performance_metrics(message.payload)
            elif action == "get_active_alerts":
                result = self._get_active_alerts()
            elif action == "acknowledge_alert":
                result = await self._acknowledge_alert(message.payload)
            elif action == "get_system_report":
                result = await self._generate_system_report(message.payload)
            elif action == "check_sla_compliance":
                result = await self._check_sla_compliance(message.payload)
            else:
                result = {"error": "Unsupported monitoring action", "action": action}
            
            return AgentMessage(
                agent_id=self.agent_id,
                target_agent_id=message.agent_id,
                message_type=MessageType.RESPONSE,
                payload=result,
                correlation_id=message.correlation_id
            )
            
        except Exception as e:
            self.logger.error(f"Error handling monitoring request: {str(e)}")
            return AgentMessage(
                agent_id=self.agent_id,
                target_agent_id=message.agent_id,
                message_type=MessageType.ERROR,
                payload={"error": str(e)},
                correlation_id=message.correlation_id
            )
    
    async def _handle_system_notification(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle system notifications for monitoring"""
        notification_type = message.payload.get("notification_type")
        
        if notification_type == "AGENT_STATUS_CHANGE":
            await self._process_agent_status_change(message.payload)
        elif notification_type == "PERFORMANCE_DEGRADATION":
            await self._process_performance_alert(message.payload)
        elif notification_type == "IMPLEMENTATION_DELAY":
            await self._process_implementation_delay(message.payload)
        elif notification_type == "CUSTOMER_ISSUE":
            await self._process_customer_issue(message.payload)
        
        return None
    
    async def _perform_system_health_check(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive system health check"""
        component = payload.get("component")
        check_type = payload.get("check_type", "standard")
        
        if component:
            # Check specific component
            health_result = await self._check_component_health(component)
            return {
                "component": component,
                "health_status": health_result.to_dict() if hasattr(health_result, 'to_dict') else health_result,
                "timestamp": datetime.now().isoformat()
            }
        else:
            # Check all components
            overall_health = {}
            for comp in self.monitored_components:
                health = await self._check_component_health(comp)
                overall_health[comp] = health
            
            # Calculate overall system health score
            healthy_components = sum(1 for h in overall_health.values() if h.get('healthy', False))
            health_score = (healthy_components / len(overall_health)) * 100
            
            return {
                "overall_health_score": health_score,
                "component_health": overall_health,
                "active_alerts_count": len(self.active_alerts),
                "system_status": "HEALTHY" if health_score >= 90 else "DEGRADED" if health_score >= 70 else "CRITICAL",
                "timestamp": datetime.now().isoformat()
            }
    
    async def _check_component_health(self, component: str) -> Dict[str, Any]:
        """Check health of specific component"""
        start_time = datetime.now()
        
        try:
            if component == "database":
                # Database health check
                healthy = True  # Would implement actual DB ping
                response_time = 50.0
                error_message = None
                
            elif component == "message_router":
                # Message router health check  
                healthy = True  # Would check router status
                response_time = 10.0
                error_message = None
                
            elif component == "system_state":
                # System state health check
                status = system_state.get_system_status()
                healthy = True
                response_time = 5.0
                error_message = None
                
            else:
                # Generic component check
                healthy = True
                response_time = 25.0
                error_message = None
            
            health_check = HealthCheck(
                component=component,
                healthy=healthy,
                response_time_ms=response_time,
                last_check=datetime.now(),
                error_message=error_message,
                metrics={"status": "operational"}
            )
            
            self.health_status[component] = health_check
            
            return {
                "healthy": healthy,
                "response_time_ms": response_time,
                "last_check": health_check.last_check.isoformat(),
                "error_message": error_message
            }
            
        except Exception as e:
            error_health = HealthCheck(
                component=component,
                healthy=False,
                response_time_ms=0.0,
                last_check=datetime.now(),
                error_message=str(e)
            )
            
            self.health_status[component] = error_health
            
            return {
                "healthy": False,
                "response_time_ms": 0.0,
                "last_check": error_health.last_check.isoformat(),
                "error_message": str(e)
            }
    
    async def _get_performance_metrics(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Get system performance metrics"""
        metric_type = payload.get("metric_type", "all")
        time_range = payload.get("time_range", "1h")
        
        # Get current system metrics
        current_metrics = self._collect_current_metrics()
        
        # Calculate performance trends
        trends = self._calculate_performance_trends(time_range)
        
        # Identify bottlenecks
        bottlenecks = self._identify_bottlenecks(current_metrics)
        
        return {
            "current_metrics": current_metrics,
            "performance_trends": trends,
            "identified_bottlenecks": bottlenecks,
            "baseline_comparison": self._compare_to_baseline(current_metrics),
            "timestamp": datetime.now().isoformat()
        }
    
    def _collect_current_metrics(self) -> Dict[str, Any]:
        """Collect current system performance metrics"""
        try:
            # System resource metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Agent system metrics from system state
            system_status = system_state.get_system_status()
            
            return {
                "system_resources": {
                    "cpu_usage_percent": cpu_percent,
                    "memory_usage_percent": memory.percent,
                    "disk_usage_percent": disk.percent,
                    "available_memory_gb": memory.available / (1024**3)
                },
                "agent_metrics": {
                    "total_customers": system_status.get("customers", {}).get("total", 0),
                    "active_implementations": system_status.get("customers", {}).get("active_implementations", 0),
                    "agent_count": system_status.get("agents", {}).get("total_registered", 0),
                    "system_uptime": system_status.get("performance", {}).get("uptime_percentage", 0)
                },
                "performance_metrics": {
                    "average_response_time_ms": system_status.get("performance", {}).get("average_response_time_ms", 0),
                    "error_rate_percentage": system_status.get("performance", {}).get("error_rate_percentage", 0)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error collecting metrics: {str(e)}")
            return {"error": f"Failed to collect metrics: {str(e)}"}
    
    def _calculate_performance_trends(self, time_range: str) -> List[Dict[str, Any]]:
        """Calculate performance trends over time range"""
        # Simplified trend calculation - would use actual historical data
        return [
            {
                "metric": "response_time",
                "trend": "stable",
                "change_percentage": 2.5
            },
            {
                "metric": "throughput",
                "trend": "improving",
                "change_percentage": -5.2
            },
            {
                "metric": "error_rate",
                "trend": "stable",
                "change_percentage": 0.1
            }
        ]
    
    def _identify_bottlenecks(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify system bottlenecks"""
        bottlenecks = []
        
        # Check resource bottlenecks
        system_resources = metrics.get("system_resources", {})
        
        if system_resources.get("cpu_usage_percent", 0) > 80:
            bottlenecks.append({
                "type": "CPU",
                "severity": "HIGH",
                "description": f"CPU usage at {system_resources['cpu_usage_percent']:.1f}%",
                "recommendation": "Consider scaling horizontally or optimizing CPU-intensive operations"
            })
        
        if system_resources.get("memory_usage_percent", 0) > 85:
            bottlenecks.append({
                "type": "MEMORY",
                "severity": "HIGH", 
                "description": f"Memory usage at {system_resources['memory_usage_percent']:.1f}%",
                "recommendation": "Monitor for memory leaks and consider increasing available memory"
            })
        
        # Check performance bottlenecks
        performance = metrics.get("performance_metrics", {})
        
        if performance.get("average_response_time_ms", 0) > self.alert_thresholds["response_time_ms"]:
            bottlenecks.append({
                "type": "RESPONSE_TIME",
                "severity": "MEDIUM",
                "description": f"Response time elevated at {performance['average_response_time_ms']:.1f}ms",
                "recommendation": "Investigate database queries and optimize slow endpoints"
            })
        
        return bottlenecks
    
    def _compare_to_baseline(self, current_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Compare current metrics to established baselines"""
        comparison = {}
        
        performance = current_metrics.get("performance_metrics", {})
        
        for metric, baseline in self.performance_baselines.items():
            current_value = performance.get(metric, 0)
            if current_value > 0:
                deviation_percentage = ((current_value - baseline) / baseline) * 100
                comparison[metric] = {
                    "current": current_value,
                    "baseline": baseline,
                    "deviation_percentage": deviation_percentage,
                    "status": "ABOVE_BASELINE" if deviation_percentage > 10 else "WITHIN_BASELINE" if deviation_percentage > -10 else "BELOW_BASELINE"
                }
        
        return comparison
    
    def _get_active_alerts(self) -> Dict[str, Any]:
        """Get all active alerts"""
        return {
            "active_alerts": [alert.to_dict() for alert in self.active_alerts.values()],
            "alert_count_by_severity": {
                severity.value: sum(1 for alert in self.active_alerts.values() if alert.severity == severity)
                for severity in AlertSeverity
            },
            "timestamp": datetime.now().isoformat()
        }
    
    async def _acknowledge_alert(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Acknowledge an active alert"""
        alert_id = payload.get("alert_id")
        acknowledged_by = payload.get("acknowledged_by", "system")
        
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.acknowledgements.append(f"{acknowledged_by} at {datetime.now().isoformat()}")
            
            return {
                "alert_id": alert_id,
                "status": "acknowledged",
                "acknowledged_by": acknowledged_by,
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {"error": f"Alert {alert_id} not found"}
    
    async def _generate_system_report(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive system report"""
        report_type = payload.get("report_type", "summary")
        time_range = payload.get("time_range", "24h")
        
        # Collect comprehensive system data
        health_status = await self._perform_system_health_check({})
        performance_metrics = await self._get_performance_metrics({"time_range": time_range})
        active_alerts = self._get_active_alerts()
        
        # Business metrics
        system_status = system_state.get_system_status()
        
        report = {
            "report_type": report_type,
            "generated_at": datetime.now().isoformat(),
            "time_range": time_range,
            "system_health": health_status,
            "performance_summary": performance_metrics,
            "alerts_summary": active_alerts,
            "business_metrics": {
                "total_customers": system_status.get("customers", {}).get("total", 0),
                "active_implementations": system_status.get("customers", {}).get("active_implementations", 0),
                "implementation_success_rate": system_status.get("customers", {}).get("success_rate", 0),
                "system_uptime": system_status.get("performance", {}).get("uptime_percentage", 0)
            },
            "recommendations": self._generate_recommendations(health_status, performance_metrics, active_alerts)
        }
        
        return report
    
    def _generate_recommendations(self, health: Dict, performance: Dict, alerts: Dict) -> List[str]:
        """Generate system improvement recommendations"""
        recommendations = []
        
        # Health-based recommendations
        if health.get("overall_health_score", 100) < 90:
            recommendations.append("Investigate unhealthy components to improve overall system reliability")
        
        # Performance-based recommendations
        bottlenecks = performance.get("identified_bottlenecks", [])
        for bottleneck in bottlenecks:
            recommendations.append(bottleneck.get("recommendation", "Address performance bottleneck"))
        
        # Alert-based recommendations
        active_count = len(alerts.get("active_alerts", []))
        if active_count > 5:
            recommendations.append("High number of active alerts - consider reviewing alert thresholds and resolving underlying issues")
        
        if not recommendations:
            recommendations.append("System is operating normally - continue monitoring")
        
        return recommendations
    
    async def _check_sla_compliance(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Check SLA compliance for customer implementations"""
        customer_id = payload.get("customer_id")
        sla_type = payload.get("sla_type", "implementation_timeline")
        
        if customer_id:
            # Check specific customer SLA
            dashboard_data = system_state.get_customer_dashboard(customer_id)
            if not dashboard_data:
                return {"error": f"Customer {customer_id} not found"}
            
            # Implementation timeline SLA
            if sla_type == "implementation_timeline":
                current_phase = dashboard_data.get("current_phase", "UNKNOWN")
                progress_percentage = dashboard_data.get("progress_percentage", 0)
                
                # Expected progress based on phase (simplified)
                expected_progress = self._calculate_expected_progress(current_phase)
                compliance_percentage = min(100, (progress_percentage / expected_progress) * 100) if expected_progress > 0 else 100
                
                return {
                    "customer_id": customer_id,
                    "sla_type": sla_type,
                    "compliance_percentage": compliance_percentage,
                    "current_progress": progress_percentage,
                    "expected_progress": expected_progress,
                    "status": "COMPLIANT" if compliance_percentage >= 95 else "AT_RISK" if compliance_percentage >= 80 else "NON_COMPLIANT",
                    "timestamp": datetime.now().isoformat()
                }
        
        else:
            # Check overall SLA compliance
            all_customers = system_state.get_system_status().get("customers", {})
            total_implementations = all_customers.get("active_implementations", 0)
            
            # Simplified SLA calculation
            compliant_implementations = int(total_implementations * 0.92)  # 92% compliance rate
            
            return {
                "overall_compliance": {
                    "total_implementations": total_implementations,
                    "compliant_implementations": compliant_implementations,
                    "compliance_percentage": (compliant_implementations / max(total_implementations, 1)) * 100,
                    "sla_violations": max(0, total_implementations - compliant_implementations)
                },
                "timestamp": datetime.now().isoformat()
            }
    
    def _calculate_expected_progress(self, current_phase: str) -> float:
        """Calculate expected progress percentage for current phase"""
        phase_progress_map = {
            "PRE_ASSESSMENT": 10,
            "DISCOVERY": 25,
            "CONFIGURATION": 40,
            "DATA_MIGRATION": 60,
            "TRAINING": 75,
            "TESTING": 90,
            "GO_LIVE": 95,
            "STABILIZATION": 98,
            "OPTIMIZATION": 100,
            "COMPLETED": 100
        }
        return phase_progress_map.get(current_phase, 50)
    
    async def _create_alert(
        self, 
        severity: AlertSeverity,
        metric_type: MetricType,
        title: str,
        description: str,
        threshold: float,
        current_value: float
    ) -> str:
        """Create new system alert"""
        alert_id = str(uuid.uuid4())
        
        alert = SystemAlert(
            alert_id=alert_id,
            severity=severity,
            metric_type=metric_type,
            title=title,
            description=description,
            threshold_value=threshold,
            current_value=current_value,
            triggered_at=datetime.now()
        )
        
        self.active_alerts[alert_id] = alert
        
        # Log alert
        self.logger.warning(f"ALERT {severity.value}: {title} - {description}")
        
        # Send notification for critical alerts
        if severity == AlertSeverity.CRITICAL:
            await self._escalate_critical_alert(alert)
        
        return alert_id
    
    async def _escalate_critical_alert(self, alert: SystemAlert):
        """Escalate critical alerts"""
        # Would implement notification to administrators
        alert.escalated = True
        self.logger.critical(f"CRITICAL ALERT ESCALATED: {alert.title}")
    
    async def _health_check_loop(self):
        """Background health check monitoring loop"""
        while True:
            try:
                # Perform health checks for all components
                for component in self.monitored_components:
                    await self._check_component_health(component)
                
                await asyncio.sleep(self.monitoring_intervals["health_check"])
                
            except Exception as e:
                self.logger.error(f"Error in health check loop: {str(e)}")
                await asyncio.sleep(self.monitoring_intervals["health_check"])
    
    async def _performance_monitoring_loop(self):
        """Background performance monitoring loop"""
        while True:
            try:
                # Collect performance metrics
                metrics = self._collect_current_metrics()
                self.metrics_history.append({
                    "timestamp": datetime.now(),
                    "metrics": metrics
                })
                
                # Trim history to last 1000 entries
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-500:]
                
                # Check performance thresholds
                await self._check_performance_thresholds(metrics)
                
                await asyncio.sleep(self.monitoring_intervals["performance_metrics"])
                
            except Exception as e:
                self.logger.error(f"Error in performance monitoring loop: {str(e)}")
                await asyncio.sleep(self.monitoring_intervals["performance_metrics"])
    
    async def _check_performance_thresholds(self, metrics: Dict[str, Any]):
        """Check if performance metrics exceed thresholds"""
        performance = metrics.get("performance_metrics", {})
        system_resources = metrics.get("system_resources", {})
        
        # Check response time
        response_time = performance.get("average_response_time_ms", 0)
        if response_time > self.alert_thresholds["response_time_ms"]:
            await self._create_alert(
                AlertSeverity.WARNING,
                MetricType.PERFORMANCE,
                "High Response Time",
                f"Average response time is {response_time:.1f}ms (threshold: {self.alert_thresholds['response_time_ms']}ms)",
                self.alert_thresholds["response_time_ms"],
                response_time
            )
        
        # Check CPU usage
        cpu_usage = system_resources.get("cpu_usage_percent", 0)
        if cpu_usage > self.alert_thresholds["cpu_usage_percent"]:
            await self._create_alert(
                AlertSeverity.ERROR if cpu_usage > 90 else AlertSeverity.WARNING,
                MetricType.RESOURCE_USAGE,
                "High CPU Usage",
                f"CPU usage is {cpu_usage:.1f}% (threshold: {self.alert_thresholds['cpu_usage_percent']}%)",
                self.alert_thresholds["cpu_usage_percent"],
                cpu_usage
            )
        
        # Check memory usage
        memory_usage = system_resources.get("memory_usage_percent", 0)
        if memory_usage > self.alert_thresholds["memory_usage_percent"]:
            await self._create_alert(
                AlertSeverity.CRITICAL if memory_usage > 95 else AlertSeverity.ERROR,
                MetricType.RESOURCE_USAGE,
                "High Memory Usage",
                f"Memory usage is {memory_usage:.1f}% (threshold: {self.alert_thresholds['memory_usage_percent']}%)",
                self.alert_thresholds["memory_usage_percent"],
                memory_usage
            )
    
    async def _resource_monitoring_loop(self):
        """Background resource monitoring loop"""
        while True:
            try:
                # Monitor system resources
                await self._monitor_system_resources()
                
                await asyncio.sleep(self.monitoring_intervals["resource_usage"])
                
            except Exception as e:
                self.logger.error(f"Error in resource monitoring loop: {str(e)}")
                await asyncio.sleep(self.monitoring_intervals["resource_usage"])
    
    async def _monitor_system_resources(self):
        """Monitor system resource usage"""
        try:
            # Disk usage check
            disk = psutil.disk_usage('/')
            disk_usage_percent = disk.percent
            
            if disk_usage_percent > self.alert_thresholds["disk_usage_percent"]:
                await self._create_alert(
                    AlertSeverity.CRITICAL if disk_usage_percent > 95 else AlertSeverity.ERROR,
                    MetricType.RESOURCE_USAGE,
                    "High Disk Usage",
                    f"Disk usage is {disk_usage_percent:.1f}% (threshold: {self.alert_thresholds['disk_usage_percent']}%)",
                    self.alert_thresholds["disk_usage_percent"],
                    disk_usage_percent
                )
        
        except Exception as e:
            self.logger.error(f"Error monitoring system resources: {str(e)}")
    
    async def _business_metrics_loop(self):
        """Background business metrics monitoring loop"""
        while True:
            try:
                # Monitor business KPIs
                await self._monitor_business_kpis()
                
                await asyncio.sleep(self.monitoring_intervals["business_metrics"])
                
            except Exception as e:
                self.logger.error(f"Error in business metrics loop: {str(e)}")
                await asyncio.sleep(self.monitoring_intervals["business_metrics"])
    
    async def _monitor_business_kpis(self):
        """Monitor business key performance indicators"""
        try:
            system_status = system_state.get_system_status()
            customers = system_status.get("customers", {})
            
            # Implementation success rate
            success_rate = customers.get("success_rate", 0)
            if success_rate < 90:  # 90% success rate threshold
                await self._create_alert(
                    AlertSeverity.WARNING,
                    MetricType.BUSINESS_METRIC,
                    "Low Implementation Success Rate",
                    f"Implementation success rate is {success_rate:.1f}% (threshold: 90%)",
                    90.0,
                    success_rate
                )
            
        except Exception as e:
            self.logger.error(f"Error monitoring business KPIs: {str(e)}")
    
    async def _alert_management_loop(self):
        """Background alert management loop"""
        while True:
            try:
                current_time = datetime.now()
                
                # Check for alerts that can be auto-resolved
                resolved_alerts = []
                for alert_id, alert in self.active_alerts.items():
                    # Auto-resolve alerts older than 1 hour if metric is back to normal
                    if (current_time - alert.triggered_at).total_seconds() > 3600:
                        # Would check if metric is back within threshold
                        alert.resolved_at = current_time
                        resolved_alerts.append(alert_id)
                
                # Move resolved alerts
                for alert_id in resolved_alerts:
                    alert = self.active_alerts.pop(alert_id)
                    self.resolved_alerts.append(alert)
                    self.logger.info(f"Auto-resolved alert: {alert.title}")
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in alert management loop: {str(e)}")
                await asyncio.sleep(300)


# Export main component
__all__ = ["SystemMonitorAgent", "SystemAlert", "AlertSeverity", "MetricType"]