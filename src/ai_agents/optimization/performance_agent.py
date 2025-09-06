#!/usr/bin/env python3
"""
Performance Optimization Agent for eFab AI Agent System
Autonomous performance monitoring, analysis, and optimization for ERP implementations
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import statistics
import json
import psutil
import time

from ..core.agent_base import BaseAgent, AgentMessage, AgentCapability, MessageType, Priority
from ..core.state_manager import system_state

# Setup logging
logger = logging.getLogger(__name__)


class PerformanceMetricType(Enum):
    """Performance metric categories"""
    RESPONSE_TIME = "RESPONSE_TIME"           # API/UI response times
    THROUGHPUT = "THROUGHPUT"                 # Requests per second
    CPU_UTILIZATION = "CPU_UTILIZATION"       # CPU usage percentage
    MEMORY_UTILIZATION = "MEMORY_UTILIZATION" # Memory usage
    DISK_IO = "DISK_IO"                       # Disk I/O operations
    NETWORK_IO = "NETWORK_IO"                 # Network bandwidth usage
    DATABASE_PERFORMANCE = "DATABASE_PERFORMANCE" # DB query performance
    ERROR_RATE = "ERROR_RATE"                 # Error percentage
    AVAILABILITY = "AVAILABILITY"             # System uptime
    USER_SATISFACTION = "USER_SATISFACTION"   # User experience metrics


class OptimizationType(Enum):
    """Types of optimizations"""
    PERFORMANCE_TUNING = "PERFORMANCE_TUNING"   # Code/config optimization
    RESOURCE_SCALING = "RESOURCE_SCALING"       # Infrastructure scaling
    CACHING_STRATEGY = "CACHING_STRATEGY"       # Cache optimization
    DATABASE_TUNING = "DATABASE_TUNING"         # DB performance tuning
    LOAD_BALANCING = "LOAD_BALANCING"           # Traffic distribution
    CODE_OPTIMIZATION = "CODE_OPTIMIZATION"     # Algorithm improvements
    INFRASTRUCTURE = "INFRASTRUCTURE"           # Hardware/cloud optimization


class AlertSeverity(Enum):
    """Performance alert severity levels"""
    LOW = "LOW"                               # Minor performance degradation
    MEDIUM = "MEDIUM"                         # Noticeable performance impact
    HIGH = "HIGH"                            # Significant performance issues
    CRITICAL = "CRITICAL"                     # System performance critical


@dataclass
class PerformanceMetric:
    """Performance metric data point"""
    metric_type: PerformanceMetricType
    value: float
    unit: str
    timestamp: datetime
    source: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    baseline_value: Optional[float] = None
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    
    @property
    def deviation_percentage(self) -> float:
        """Calculate percentage deviation from baseline"""
        if self.baseline_value and self.baseline_value != 0:
            return ((self.value - self.baseline_value) / self.baseline_value) * 100
        return 0.0
    
    @property
    def alert_level(self) -> AlertSeverity:
        """Determine alert level based on thresholds"""
        if self.threshold_critical and self.value >= self.threshold_critical:
            return AlertSeverity.CRITICAL
        elif self.threshold_warning and self.value >= self.threshold_warning:
            return AlertSeverity.HIGH
        elif abs(self.deviation_percentage) > 25:
            return AlertSeverity.MEDIUM
        elif abs(self.deviation_percentage) > 10:
            return AlertSeverity.LOW
        return AlertSeverity.LOW


@dataclass
class OptimizationRecommendation:
    """Performance optimization recommendation"""
    recommendation_id: str
    optimization_type: OptimizationType
    title: str
    description: str
    impact_estimate: str  # HIGH, MEDIUM, LOW
    effort_estimate: str  # HIGH, MEDIUM, LOW
    expected_improvement: str
    implementation_steps: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    success_metrics: List[str] = field(default_factory=list)
    priority_score: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class PerformanceBaseline:
    """Performance baseline for comparison"""
    metric_type: PerformanceMetricType
    baseline_value: float
    measurement_period: timedelta
    confidence_interval: float = 0.95
    sample_size: int = 0
    established_date: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)


class PerformanceOptimizationAgent(BaseAgent):
    """
    Performance Optimization Agent for eFab AI Agent System
    
    Capabilities:
    - Real-time performance monitoring and alerting
    - Performance baseline establishment and tracking
    - Automated performance analysis and bottleneck identification
    - Optimization recommendation generation with impact analysis
    - Performance regression detection and prevention
    - Resource utilization optimization and scaling recommendations
    - Performance testing coordination and results analysis
    - SLA monitoring and compliance tracking
    """
    
    def __init__(self, agent_id: str = "performance_optimization_agent"):
        """Initialize Performance Optimization Agent"""
        super().__init__(
            agent_id=agent_id,
            agent_name="Performance Optimization Specialist",
            agent_description="Autonomous performance monitoring and optimization for ERP implementations"
        )
        
        # Performance monitoring state
        self.metric_history: Dict[str, List[PerformanceMetric]] = {}
        self.baselines: Dict[str, PerformanceBaseline] = {}
        self.active_alerts: Dict[str, Dict[str, Any]] = {}
        self.optimization_recommendations: List[OptimizationRecommendation] = []
        
        # Performance thresholds (configurable)
        self.default_thresholds = {
            PerformanceMetricType.RESPONSE_TIME: {"warning": 2.0, "critical": 5.0, "unit": "seconds"},
            PerformanceMetricType.THROUGHPUT: {"warning": 50, "critical": 20, "unit": "requests/sec"},
            PerformanceMetricType.CPU_UTILIZATION: {"warning": 80, "critical": 95, "unit": "percent"},
            PerformanceMetricType.MEMORY_UTILIZATION: {"warning": 85, "critical": 95, "unit": "percent"},
            PerformanceMetricType.ERROR_RATE: {"warning": 2, "critical": 5, "unit": "percent"},
            PerformanceMetricType.AVAILABILITY: {"warning": 99.5, "critical": 99.0, "unit": "percent"}
        }
        
        # Optimization templates
        self.optimization_templates = {
            "high_response_time": {
                "type": OptimizationType.PERFORMANCE_TUNING,
                "recommendations": [
                    "Implement response caching for frequently accessed data",
                    "Optimize database query performance",
                    "Consider CDN implementation for static assets",
                    "Review and optimize API endpoint logic"
                ]
            },
            "high_cpu_usage": {
                "type": OptimizationType.RESOURCE_SCALING,
                "recommendations": [
                    "Scale CPU resources horizontally or vertically",
                    "Implement process optimization and code profiling",
                    "Consider load balancing to distribute CPU load",
                    "Optimize computational algorithms"
                ]
            },
            "high_memory_usage": {
                "type": OptimizationType.RESOURCE_SCALING,
                "recommendations": [
                    "Implement memory optimization techniques",
                    "Scale memory resources",
                    "Review memory leaks and garbage collection",
                    "Optimize data structures and caching strategies"
                ]
            }
        }
        
        # Performance collection interval
        self.collection_interval = 60  # seconds
        self.analysis_interval = 300   # 5 minutes
    
    def _initialize(self):
        """Initialize performance optimization capabilities"""
        # Register performance optimization capabilities
        self.register_capability(AgentCapability(
            name="monitor_performance",
            description="Monitor system performance metrics in real-time",
            input_schema={
                "type": "object",
                "properties": {
                    "metrics_to_monitor": {"type": "array"},
                    "monitoring_duration": {"type": "number"},
                    "alert_thresholds": {"type": "object"}
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "monitoring_status": {"type": "object"},
                    "current_metrics": {"type": "object"},
                    "alerts_generated": {"type": "array"}
                }
            }
        ))
        
        self.register_capability(AgentCapability(
            name="analyze_performance",
            description="Analyze performance data and identify optimization opportunities",
            input_schema={
                "type": "object",
                "properties": {
                    "analysis_period": {"type": "object"},
                    "metrics_data": {"type": "array"},
                    "baseline_comparison": {"type": "boolean"}
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "analysis_results": {"type": "object"},
                    "performance_trends": {"type": "array"},
                    "optimization_recommendations": {"type": "array"}
                }
            }
        ))
        
        self.register_capability(AgentCapability(
            name="optimize_performance",
            description="Implement performance optimizations automatically",
            input_schema={
                "type": "object",
                "properties": {
                    "optimization_type": {"type": "string"},
                    "target_metrics": {"type": "array"},
                    "optimization_parameters": {"type": "object"}
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "optimization_results": {"type": "object"},
                    "performance_improvement": {"type": "object"},
                    "next_actions": {"type": "array"}
                }
            }
        ))
        
        self.register_capability(AgentCapability(
            name="establish_baselines",
            description="Establish performance baselines for comparison",
            input_schema={
                "type": "object",
                "properties": {
                    "metrics_to_baseline": {"type": "array"},
                    "measurement_period": {"type": "number"},
                    "confidence_level": {"type": "number"}
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "baselines_established": {"type": "object"},
                    "baseline_quality": {"type": "object"},
                    "monitoring_recommendations": {"type": "array"}
                }
            }
        ))
        
        # Register message handlers
        self.register_message_handler(MessageType.REQUEST, self._handle_performance_request)
        self.register_message_handler(MessageType.NOTIFICATION, self._handle_performance_notification)
        
        # Start background monitoring
        asyncio.create_task(self._performance_monitoring_loop())
        asyncio.create_task(self._performance_analysis_loop())
    
    async def _handle_performance_request(self, message: AgentMessage) -> AgentMessage:
        """Handle performance optimization requests"""
        action = message.payload.get("action")
        
        try:
            if action == "monitor_performance":
                result = await self._monitor_performance(message.payload)
            elif action == "analyze_performance":
                result = await self._analyze_performance(message.payload)
            elif action == "optimize_performance":
                result = await self._optimize_performance(message.payload)
            elif action == "establish_baselines":
                result = await self._establish_baselines(message.payload)
            elif action == "get_performance_report":
                result = await self._get_performance_report(message.payload)
            else:
                result = {"error": "Unsupported action", "action": action}
            
            return AgentMessage(
                agent_id=self.agent_id,
                target_agent_id=message.agent_id,
                message_type=MessageType.RESPONSE,
                payload=result,
                correlation_id=message.correlation_id
            )
            
        except Exception as e:
            self.logger.error(f"Error handling performance request: {str(e)}")
            return AgentMessage(
                agent_id=self.agent_id,
                target_agent_id=message.agent_id,
                message_type=MessageType.ERROR,
                payload={"error": str(e)},
                correlation_id=message.correlation_id
            )
    
    async def _handle_performance_notification(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle performance-related notifications"""
        notification_type = message.payload.get("notification_type")
        
        if notification_type == "METRIC_UPDATE":
            metric_data = message.payload.get("metric_data", {})
            await self._process_metric_update(metric_data)
        
        elif notification_type == "PERFORMANCE_ALERT":
            alert_data = message.payload.get("alert_data", {})
            await self._process_performance_alert(alert_data)
        
        elif notification_type == "OPTIMIZATION_COMPLETED":
            optimization_data = message.payload.get("optimization_data", {})
            await self._process_optimization_completion(optimization_data)
        
        return None
    
    async def _monitor_performance(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor system performance metrics"""
        metrics_to_monitor = payload.get("metrics_to_monitor", list(PerformanceMetricType))
        monitoring_duration = payload.get("monitoring_duration", 3600)  # 1 hour default
        custom_thresholds = payload.get("alert_thresholds", {})
        
        # Update thresholds if provided
        active_thresholds = self.default_thresholds.copy()
        active_thresholds.update(custom_thresholds)
        
        # Collect current metrics
        current_metrics = {}
        alerts_generated = []
        
        for metric_type_str in metrics_to_monitor:
            try:
                metric_type = PerformanceMetricType(metric_type_str)
                metric_value = await self._collect_metric(metric_type)
                
                if metric_value is not None:
                    threshold_config = active_thresholds.get(metric_type, {})
                    
                    metric = PerformanceMetric(
                        metric_type=metric_type,
                        value=metric_value,
                        unit=threshold_config.get("unit", ""),
                        timestamp=datetime.now(),
                        source=self.agent_id,
                        threshold_warning=threshold_config.get("warning"),
                        threshold_critical=threshold_config.get("critical")
                    )
                    
                    current_metrics[metric_type_str] = {
                        "value": metric_value,
                        "unit": metric.unit,
                        "timestamp": metric.timestamp.isoformat(),
                        "alert_level": metric.alert_level.value
                    }
                    
                    # Store metric for history
                    if metric_type_str not in self.metric_history:
                        self.metric_history[metric_type_str] = []
                    
                    self.metric_history[metric_type_str].append(metric)
                    
                    # Keep only last 1000 metrics per type
                    if len(self.metric_history[metric_type_str]) > 1000:
                        self.metric_history[metric_type_str] = self.metric_history[metric_type_str][-1000:]
                    
                    # Generate alerts if thresholds exceeded
                    if metric.alert_level in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]:
                        alert = {
                            "alert_id": f"PERF_{metric_type_str}_{int(time.time())}",
                            "metric_type": metric_type_str,
                            "severity": metric.alert_level.value,
                            "current_value": metric_value,
                            "threshold_exceeded": threshold_config.get("critical" if metric.alert_level == AlertSeverity.CRITICAL else "warning"),
                            "message": f"{metric_type_str} {metric.alert_level.value.lower()}: {metric_value} {metric.unit}",
                            "timestamp": metric.timestamp.isoformat()
                        }
                        
                        alerts_generated.append(alert)
                        self.active_alerts[alert["alert_id"]] = alert
            
            except (ValueError, Exception) as e:
                self.logger.error(f"Error monitoring metric {metric_type_str}: {str(e)}")
                continue
        
        return {
            "monitoring_status": {
                "status": "ACTIVE",
                "metrics_monitored": len(current_metrics),
                "alerts_active": len(alerts_generated),
                "monitoring_started": datetime.now().isoformat(),
                "next_collection": (datetime.now() + timedelta(seconds=self.collection_interval)).isoformat()
            },
            "current_metrics": current_metrics,
            "alerts_generated": alerts_generated
        }
    
    async def _collect_metric(self, metric_type: PerformanceMetricType) -> Optional[float]:
        """Collect specific performance metric"""
        try:
            if metric_type == PerformanceMetricType.CPU_UTILIZATION:
                return psutil.cpu_percent(interval=1)
            
            elif metric_type == PerformanceMetricType.MEMORY_UTILIZATION:
                memory = psutil.virtual_memory()
                return memory.percent
            
            elif metric_type == PerformanceMetricType.DISK_IO:
                disk_io = psutil.disk_io_counters()
                if disk_io:
                    return disk_io.read_bytes + disk_io.write_bytes
                return 0
            
            elif metric_type == PerformanceMetricType.NETWORK_IO:
                net_io = psutil.net_io_counters()
                if net_io:
                    return net_io.bytes_sent + net_io.bytes_recv
                return 0
            
            elif metric_type == PerformanceMetricType.RESPONSE_TIME:
                # Simulate API response time measurement
                # In real implementation, this would measure actual API response times
                return 1.2  # Simulated 1.2 second response time
            
            elif metric_type == PerformanceMetricType.THROUGHPUT:
                # Simulate throughput measurement
                return 150  # Simulated 150 requests per second
            
            elif metric_type == PerformanceMetricType.ERROR_RATE:
                # Simulate error rate
                return 1.5  # Simulated 1.5% error rate
            
            elif metric_type == PerformanceMetricType.AVAILABILITY:
                # Simulate availability
                return 99.8  # Simulated 99.8% availability
            
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Error collecting metric {metric_type}: {str(e)}")
            return None
    
    async def _analyze_performance(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance data and generate insights"""
        analysis_period = payload.get("analysis_period", {"hours": 24})
        metrics_data = payload.get("metrics_data", [])
        baseline_comparison = payload.get("baseline_comparison", True)
        
        # Calculate analysis window
        end_time = datetime.now()
        start_time = end_time - timedelta(**analysis_period)
        
        analysis_results = {}
        performance_trends = []
        optimization_recommendations = []
        
        # Analyze each metric type
        for metric_type_str, metric_history in self.metric_history.items():
            if not metric_history:
                continue
            
            # Filter metrics within analysis period
            relevant_metrics = [
                m for m in metric_history
                if start_time <= m.timestamp <= end_time
            ]
            
            if not relevant_metrics:
                continue
            
            # Calculate statistics
            values = [m.value for m in relevant_metrics]
            analysis_results[metric_type_str] = {
                "sample_count": len(values),
                "average": statistics.mean(values),
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
                "std_dev": statistics.stdev(values) if len(values) > 1 else 0,
                "trend": self._calculate_trend(values),
                "anomalies_detected": self._detect_anomalies(values)
            }
            
            # Baseline comparison if requested
            if baseline_comparison and metric_type_str in self.baselines:
                baseline = self.baselines[metric_type_str]
                current_avg = statistics.mean(values)
                deviation = ((current_avg - baseline.baseline_value) / baseline.baseline_value) * 100
                
                analysis_results[metric_type_str]["baseline_comparison"] = {
                    "baseline_value": baseline.baseline_value,
                    "current_average": current_avg,
                    "deviation_percentage": deviation,
                    "performance_status": "DEGRADED" if abs(deviation) > 15 else "STABLE"
                }
            
            # Generate performance trends
            if len(values) >= 10:  # Need sufficient data points
                trend_analysis = {
                    "metric": metric_type_str,
                    "trend_direction": analysis_results[metric_type_str]["trend"],
                    "trend_strength": abs(self._calculate_trend_strength(values)),
                    "forecast": self._forecast_metric(values, 12)  # 12 point forecast
                }
                performance_trends.append(trend_analysis)
        
        # Generate optimization recommendations
        optimization_recommendations = await self._generate_optimization_recommendations(analysis_results)
        
        return {
            "analysis_results": analysis_results,
            "performance_trends": performance_trends,
            "optimization_recommendations": optimization_recommendations,
            "analysis_summary": {
                "analysis_period": f"{analysis_period}",
                "metrics_analyzed": len(analysis_results),
                "trends_identified": len(performance_trends),
                "recommendations_generated": len(optimization_recommendations),
                "overall_health": self._assess_overall_health(analysis_results)
            }
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from values"""
        if len(values) < 2:
            return "STABLE"
        
        # Simple linear trend calculation
        x = list(range(len(values)))
        y = values
        
        n = len(values)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] * x[i] for i in range(n))
        
        if n * sum_x2 - sum_x * sum_x == 0:
            return "STABLE"
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        
        if slope > 0.1:
            return "INCREASING"
        elif slope < -0.1:
            return "DECREASING"
        else:
            return "STABLE"
    
    def _calculate_trend_strength(self, values: List[float]) -> float:
        """Calculate strength of trend (correlation coefficient)"""
        if len(values) < 2:
            return 0.0
        
        x = list(range(len(values)))
        y = values
        
        n = len(values)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] * x[i] for i in range(n))
        sum_y2 = sum(y[i] * y[i] for i in range(n))
        
        numerator = n * sum_xy - sum_x * sum_y
        denominator_x = n * sum_x2 - sum_x * sum_x
        denominator_y = n * sum_y2 - sum_y * sum_y
        
        if denominator_x <= 0 or denominator_y <= 0:
            return 0.0
        
        correlation = numerator / (denominator_x * denominator_y) ** 0.5
        return correlation
    
    def _detect_anomalies(self, values: List[float]) -> int:
        """Detect anomalies using simple statistical method"""
        if len(values) < 10:
            return 0
        
        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values)
        
        # Count values more than 2 standard deviations from mean
        anomalies = sum(1 for v in values if abs(v - mean_val) > 2 * std_val)
        return anomalies
    
    def _forecast_metric(self, values: List[float], periods: int) -> List[float]:
        """Simple linear forecast"""
        if len(values) < 3:
            return [values[-1]] * periods if values else [0] * periods
        
        # Simple moving average with trend
        recent_values = values[-min(10, len(values)):]
        avg = statistics.mean(recent_values)
        
        # Calculate simple trend
        if len(recent_values) >= 2:
            trend = (recent_values[-1] - recent_values[0]) / (len(recent_values) - 1)
        else:
            trend = 0
        
        forecast = []
        for i in range(periods):
            predicted = avg + trend * (i + 1)
            forecast.append(max(0, predicted))  # Don't forecast negative values
        
        return forecast
    
    async def _generate_optimization_recommendations(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate optimization recommendations based on analysis"""
        recommendations = []
        
        for metric_type, analysis in analysis_results.items():
            try:
                # Check for high values that need optimization
                if metric_type == "RESPONSE_TIME" and analysis["average"] > 2.0:
                    rec = self._create_optimization_recommendation(
                        "response_time_optimization",
                        OptimizationType.PERFORMANCE_TUNING,
                        "Optimize Response Time",
                        f"Average response time is {analysis['average']:.2f}s, exceeding 2.0s threshold",
                        "HIGH",
                        "MEDIUM",
                        "30-50% reduction in response time",
                        [
                            "Implement application-level caching",
                            "Optimize database queries",
                            "Enable compression for API responses",
                            "Consider CDN for static content"
                        ]
                    )
                    recommendations.append(rec)
                
                elif metric_type == "CPU_UTILIZATION" and analysis["average"] > 75:
                    rec = self._create_optimization_recommendation(
                        "cpu_optimization",
                        OptimizationType.RESOURCE_SCALING,
                        "Optimize CPU Utilization",
                        f"Average CPU utilization is {analysis['average']:.1f}%, approaching capacity limits",
                        "HIGH",
                        "HIGH",
                        "20-40% reduction in CPU usage",
                        [
                            "Scale CPU resources horizontally",
                            "Profile application for CPU hotspots",
                            "Implement process optimization",
                            "Consider load balancing"
                        ]
                    )
                    recommendations.append(rec)
                
                elif metric_type == "MEMORY_UTILIZATION" and analysis["average"] > 80:
                    rec = self._create_optimization_recommendation(
                        "memory_optimization",
                        OptimizationType.RESOURCE_SCALING,
                        "Optimize Memory Usage",
                        f"Average memory utilization is {analysis['average']:.1f}%, approaching limits",
                        "HIGH",
                        "MEDIUM",
                        "15-25% reduction in memory usage",
                        [
                            "Implement memory-efficient data structures",
                            "Optimize caching strategies",
                            "Scale memory resources",
                            "Review for memory leaks"
                        ]
                    )
                    recommendations.append(rec)
                
                # Check for trends that indicate future problems
                if analysis.get("trend") == "INCREASING":
                    if metric_type in ["RESPONSE_TIME", "CPU_UTILIZATION", "MEMORY_UTILIZATION", "ERROR_RATE"]:
                        rec = self._create_optimization_recommendation(
                            f"trend_prevention_{metric_type.lower()}",
                            OptimizationType.PERFORMANCE_TUNING,
                            f"Address {metric_type.replace('_', ' ').title()} Trend",
                            f"{metric_type.replace('_', ' ').title()} showing increasing trend, preventive action recommended",
                            "MEDIUM",
                            "LOW",
                            "Prevent future performance degradation",
                            [
                                "Monitor trend closely",
                                "Plan proactive scaling",
                                "Review recent changes",
                                "Implement preventive measures"
                            ]
                        )
                        recommendations.append(rec)
            
            except Exception as e:
                self.logger.error(f"Error generating recommendations for {metric_type}: {str(e)}")
                continue
        
        # Sort recommendations by priority (impact vs effort)
        for rec in recommendations:
            impact_score = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}[rec["impact_estimate"]]
            effort_score = {"HIGH": 1, "MEDIUM": 2, "LOW": 3}[rec["effort_estimate"]]  # Lower effort = higher score
            rec["priority_score"] = impact_score * effort_score
        
        recommendations.sort(key=lambda x: x["priority_score"], reverse=True)
        
        return recommendations
    
    def _create_optimization_recommendation(
        self,
        rec_id: str,
        opt_type: OptimizationType,
        title: str,
        description: str,
        impact: str,
        effort: str,
        expected_improvement: str,
        steps: List[str]
    ) -> Dict[str, Any]:
        """Create optimization recommendation"""
        return {
            "recommendation_id": rec_id,
            "optimization_type": opt_type.value,
            "title": title,
            "description": description,
            "impact_estimate": impact,
            "effort_estimate": effort,
            "expected_improvement": expected_improvement,
            "implementation_steps": steps,
            "risks": ["Performance testing recommended", "Monitor during implementation"],
            "success_metrics": ["Metric improvement", "No regression in other metrics"],
            "created_at": datetime.now().isoformat()
        }
    
    def _assess_overall_health(self, analysis_results: Dict[str, Any]) -> str:
        """Assess overall system performance health"""
        if not analysis_results:
            return "UNKNOWN"
        
        health_scores = []
        
        for metric_type, analysis in analysis_results.items():
            score = 100  # Start with perfect score
            
            # Deduct points based on metric-specific criteria
            if metric_type == "RESPONSE_TIME":
                if analysis["average"] > 5.0:
                    score -= 50
                elif analysis["average"] > 2.0:
                    score -= 25
            
            elif metric_type == "CPU_UTILIZATION":
                if analysis["average"] > 90:
                    score -= 40
                elif analysis["average"] > 75:
                    score -= 20
            
            elif metric_type == "MEMORY_UTILIZATION":
                if analysis["average"] > 90:
                    score -= 40
                elif analysis["average"] > 80:
                    score -= 20
            
            elif metric_type == "ERROR_RATE":
                if analysis["average"] > 5:
                    score -= 60
                elif analysis["average"] > 2:
                    score -= 30
            
            # Deduct for negative trends
            if analysis.get("trend") == "INCREASING" and metric_type in ["RESPONSE_TIME", "ERROR_RATE", "CPU_UTILIZATION"]:
                score -= 15
            
            # Deduct for anomalies
            anomaly_rate = analysis.get("anomalies_detected", 0) / analysis.get("sample_count", 1)
            if anomaly_rate > 0.1:  # More than 10% anomalies
                score -= 20
            
            health_scores.append(max(0, score))
        
        if not health_scores:
            return "UNKNOWN"
        
        overall_score = statistics.mean(health_scores)
        
        if overall_score >= 90:
            return "EXCELLENT"
        elif overall_score >= 80:
            return "GOOD"
        elif overall_score >= 70:
            return "FAIR"
        elif overall_score >= 60:
            return "POOR"
        else:
            return "CRITICAL"
    
    async def _optimize_performance(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Implement performance optimizations"""
        optimization_type = payload.get("optimization_type")
        target_metrics = payload.get("target_metrics", [])
        parameters = payload.get("optimization_parameters", {})
        
        # Simulate optimization implementation
        optimization_results = {
            "optimization_id": f"OPT_{int(time.time())}",
            "type": optimization_type,
            "status": "COMPLETED",
            "implementation_time": datetime.now().isoformat(),
            "changes_applied": []
        }
        
        performance_improvement = {}
        next_actions = []
        
        # Simulate different optimization types
        if optimization_type == "CACHING_STRATEGY":
            optimization_results["changes_applied"] = [
                "Enabled Redis caching for frequent queries",
                "Implemented application-level cache with 1-hour TTL",
                "Added cache warming for critical data"
            ]
            
            for metric in target_metrics:
                if metric == "RESPONSE_TIME":
                    performance_improvement[metric] = {
                        "before": 3.2,
                        "after": 1.8,
                        "improvement_percentage": 43.8
                    }
            
            next_actions = [
                "Monitor cache hit rates",
                "Fine-tune cache TTL values",
                "Implement cache invalidation strategy"
            ]
        
        elif optimization_type == "DATABASE_TUNING":
            optimization_results["changes_applied"] = [
                "Added missing database indexes",
                "Optimized slow queries",
                "Updated database configuration parameters"
            ]
            
            for metric in target_metrics:
                if metric == "RESPONSE_TIME":
                    performance_improvement[metric] = {
                        "before": 4.1,
                        "after": 2.3,
                        "improvement_percentage": 43.9
                    }
            
            next_actions = [
                "Monitor query performance",
                "Review query execution plans",
                "Consider query optimization"
            ]
        
        elif optimization_type == "RESOURCE_SCALING":
            optimization_results["changes_applied"] = [
                "Scaled CPU resources from 4 to 6 cores",
                "Increased memory allocation by 50%",
                "Enabled auto-scaling policies"
            ]
            
            for metric in target_metrics:
                if metric == "CPU_UTILIZATION":
                    performance_improvement[metric] = {
                        "before": 82.5,
                        "after": 58.3,
                        "improvement_percentage": 29.3
                    }
            
            next_actions = [
                "Monitor resource utilization trends",
                "Adjust auto-scaling thresholds",
                "Plan for future capacity needs"
            ]
        
        return {
            "optimization_results": optimization_results,
            "performance_improvement": performance_improvement,
            "next_actions": next_actions,
            "validation_required": [
                "Verify performance improvements are sustained",
                "Ensure no negative side effects",
                "Update monitoring thresholds if needed"
            ]
        }
    
    async def _establish_baselines(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Establish performance baselines"""
        metrics_to_baseline = payload.get("metrics_to_baseline", [])
        measurement_period = payload.get("measurement_period", 3600)  # 1 hour
        confidence_level = payload.get("confidence_level", 0.95)
        
        baselines_established = {}
        baseline_quality = {}
        monitoring_recommendations = []
        
        for metric_type_str in metrics_to_baseline:
            if metric_type_str not in self.metric_history:
                monitoring_recommendations.append(f"Start collecting data for {metric_type_str}")
                continue
            
            metric_history = self.metric_history[metric_type_str]
            if not metric_history:
                continue
            
            # Get recent metrics for baseline
            cutoff_time = datetime.now() - timedelta(seconds=measurement_period)
            recent_metrics = [
                m for m in metric_history
                if m.timestamp >= cutoff_time
            ]
            
            if len(recent_metrics) < 10:  # Need minimum sample size
                monitoring_recommendations.append(f"Insufficient data for {metric_type_str} baseline - need at least 10 samples")
                continue
            
            values = [m.value for m in recent_metrics]
            baseline_value = statistics.mean(values)
            
            # Create baseline
            baseline = PerformanceBaseline(
                metric_type=PerformanceMetricType(metric_type_str),
                baseline_value=baseline_value,
                measurement_period=timedelta(seconds=measurement_period),
                confidence_interval=confidence_level,
                sample_size=len(values)
            )
            
            self.baselines[metric_type_str] = baseline
            
            baselines_established[metric_type_str] = {
                "baseline_value": baseline_value,
                "sample_size": len(values),
                "measurement_period_seconds": measurement_period,
                "confidence_interval": confidence_level,
                "min_value": min(values),
                "max_value": max(values),
                "std_deviation": statistics.stdev(values) if len(values) > 1 else 0
            }
            
            # Assess baseline quality
            std_dev = statistics.stdev(values) if len(values) > 1 else 0
            coefficient_variation = (std_dev / baseline_value) * 100 if baseline_value != 0 else 0
            
            if coefficient_variation < 10:
                quality = "HIGH"
            elif coefficient_variation < 25:
                quality = "MEDIUM"
            else:
                quality = "LOW"
            
            baseline_quality[metric_type_str] = {
                "quality_rating": quality,
                "coefficient_of_variation": coefficient_variation,
                "stability_assessment": "STABLE" if coefficient_variation < 15 else "VARIABLE"
            }
        
        return {
            "baselines_established": baselines_established,
            "baseline_quality": baseline_quality,
            "monitoring_recommendations": monitoring_recommendations,
            "summary": {
                "total_baselines_created": len(baselines_established),
                "high_quality_baselines": sum(1 for q in baseline_quality.values() if q["quality_rating"] == "HIGH"),
                "recommendations_count": len(monitoring_recommendations)
            }
        }
    
    async def _get_performance_report(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        report_period = payload.get("report_period", {"days": 7})
        include_recommendations = payload.get("include_recommendations", True)
        
        end_time = datetime.now()
        start_time = end_time - timedelta(**report_period)
        
        # Analyze current performance
        analysis_payload = {
            "analysis_period": report_period,
            "baseline_comparison": True
        }
        
        analysis_result = await self._analyze_performance(analysis_payload)
        
        # Generate summary metrics
        summary_metrics = {}
        for metric_type, analysis in analysis_result["analysis_results"].items():
            summary_metrics[metric_type] = {
                "current_average": analysis["average"],
                "trend": analysis["trend"],
                "health_status": self._get_metric_health_status(metric_type, analysis["average"])
            }
        
        # Active alerts summary
        active_alerts_summary = {
            "total_alerts": len(self.active_alerts),
            "critical_alerts": sum(1 for alert in self.active_alerts.values() if alert.get("severity") == "CRITICAL"),
            "high_alerts": sum(1 for alert in self.active_alerts.values() if alert.get("severity") == "HIGH"),
            "alert_types": list(set(alert.get("metric_type") for alert in self.active_alerts.values()))
        }
        
        performance_report = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "report_period": report_period,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat()
            },
            "executive_summary": {
                "overall_health": analysis_result["analysis_summary"]["overall_health"],
                "metrics_analyzed": analysis_result["analysis_summary"]["metrics_analyzed"],
                "trends_identified": analysis_result["analysis_summary"]["trends_identified"],
                "recommendations_available": analysis_result["analysis_summary"]["recommendations_generated"]
            },
            "performance_metrics": summary_metrics,
            "performance_trends": analysis_result["performance_trends"],
            "alerts_summary": active_alerts_summary,
            "baseline_status": {
                metric_type: {
                    "baseline_established": metric_type in self.baselines,
                    "baseline_age_days": (datetime.now() - self.baselines[metric_type].established_date).days if metric_type in self.baselines else 0
                }
                for metric_type in summary_metrics.keys()
            }
        }
        
        if include_recommendations:
            performance_report["optimization_recommendations"] = analysis_result["optimization_recommendations"]
        
        return performance_report
    
    def _get_metric_health_status(self, metric_type: str, value: float) -> str:
        """Get health status for specific metric"""
        thresholds = self.default_thresholds.get(PerformanceMetricType(metric_type), {})
        
        if thresholds.get("critical") and value >= thresholds["critical"]:
            return "CRITICAL"
        elif thresholds.get("warning") and value >= thresholds["warning"]:
            return "WARNING"
        else:
            return "HEALTHY"
    
    async def _performance_monitoring_loop(self):
        """Background performance monitoring loop"""
        while self.status != "SHUTDOWN":
            try:
                # Collect all configured metrics
                for metric_type in PerformanceMetricType:
                    metric_value = await self._collect_metric(metric_type)
                    
                    if metric_value is not None:
                        metric = PerformanceMetric(
                            metric_type=metric_type,
                            value=metric_value,
                            unit=self.default_thresholds.get(metric_type, {}).get("unit", ""),
                            timestamp=datetime.now(),
                            source=self.agent_id
                        )
                        
                        # Store metric
                        metric_type_str = metric_type.value
                        if metric_type_str not in self.metric_history:
                            self.metric_history[metric_type_str] = []
                        
                        self.metric_history[metric_type_str].append(metric)
                
                await asyncio.sleep(self.collection_interval)
                
            except Exception as e:
                self.logger.error(f"Error in performance monitoring loop: {str(e)}")
                await asyncio.sleep(self.collection_interval)
    
    async def _performance_analysis_loop(self):
        """Background performance analysis loop"""
        while self.status != "SHUTDOWN":
            try:
                # Perform periodic analysis
                if self.metric_history:
                    analysis_payload = {
                        "analysis_period": {"minutes": 15},
                        "baseline_comparison": True
                    }
                    
                    await self._analyze_performance(analysis_payload)
                
                await asyncio.sleep(self.analysis_interval)
                
            except Exception as e:
                self.logger.error(f"Error in performance analysis loop: {str(e)}")
                await asyncio.sleep(self.analysis_interval)
    
    async def _process_metric_update(self, metric_data: Dict[str, Any]):
        """Process metric update notification"""
        # Implementation for processing external metric updates
        pass
    
    async def _process_performance_alert(self, alert_data: Dict[str, Any]):
        """Process performance alert notification"""
        # Implementation for processing performance alerts
        pass
    
    async def _process_optimization_completion(self, optimization_data: Dict[str, Any]):
        """Process optimization completion notification"""
        # Implementation for processing optimization completion
        pass


# Export main component
__all__ = ["PerformanceOptimizationAgent", "PerformanceMetricType", "OptimizationType", "AlertSeverity"]