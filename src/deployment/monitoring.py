"""
Deployment Monitoring Module

This module provides monitoring capabilities for deployed models including
health checks, performance monitoring, and alerting functionality.

Classes:
    DeploymentMonitor: Monitor for deployment health and performance
    HealthChecker: Health check utilities for endpoints
    PerformanceMonitor: Performance monitoring for model serving
    AlertManager: Alert management for deployment issues

Author: MLOps Team
Version: 1.0.0
"""

import os
import json
import logging
import time
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

# Local imports
from .model_deployment import EndpointManager, ModelServingManager
from ..config import Config
from ..utils import setup_logging

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status enumeration."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    endpoint_name: str
    status: HealthStatus
    timestamp: str
    latency_ms: float
    response_code: Optional[int] = None
    error_message: Optional[str] = None
    deployed_models_count: int = 0
    traffic_distribution: Dict[str, int] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Performance metrics for an endpoint."""
    endpoint_name: str
    timestamp: str
    request_count: int
    error_count: int
    average_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_qps: float
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    error_rate: float = 0.0


@dataclass
class Alert:
    """Alert for deployment issues."""
    id: str
    endpoint_name: str
    severity: AlertSeverity
    title: str
    message: str
    timestamp: str
    resolved: bool = False
    resolution_time: Optional[str] = None


class HealthChecker:
    """
    Health checker for model endpoints.
    
    Performs health checks on deployed models and endpoints to ensure
    they are functioning correctly.
    """
    
    def __init__(self, endpoint_manager: EndpointManager):
        """
        Initialize health checker.
        
        Args:
            endpoint_manager: EndpointManager instance
        """
        self.endpoint_manager = endpoint_manager
        
    def check_endpoint_health(self, endpoint_name: str, 
                             test_instances: Optional[List[Dict[str, Any]]] = None) -> HealthCheckResult:
        """
        Perform health check on an endpoint.
        
        Args:
            endpoint_name: Endpoint resource name or ID
            test_instances: Test instances for prediction health check
            
        Returns:
            HealthCheckResult object
        """
        start_time = time.time()
        
        try:
            # Get endpoint info
            endpoint_info = self.endpoint_manager.get_endpoint_info(endpoint_name)
            
            if not endpoint_info:
                return HealthCheckResult(
                    endpoint_name=endpoint_name,
                    status=HealthStatus.CRITICAL,
                    timestamp=datetime.now().isoformat(),
                    latency_ms=0.0,
                    error_message="Endpoint not found or not accessible"
                )
            
            # Check if endpoint has deployed models
            if not endpoint_info.deployed_models:
                return HealthCheckResult(
                    endpoint_name=endpoint_name,
                    status=HealthStatus.CRITICAL,
                    timestamp=datetime.now().isoformat(),
                    latency_ms=time.time() - start_time,
                    error_message="No models deployed to endpoint"
                )
            
            # Perform prediction test if test instances provided
            if test_instances:
                serving_manager = ModelServingManager(
                    self.endpoint_manager.project_id, 
                    self.endpoint_manager.location
                )
                
                prediction_response = serving_manager.predict(
                    endpoint_name, 
                    test_instances
                )
                
                if not prediction_response:
                    return HealthCheckResult(
                        endpoint_name=endpoint_name,
                        status=HealthStatus.CRITICAL,
                        timestamp=datetime.now().isoformat(),
                        latency_ms=(time.time() - start_time) * 1000,
                        error_message="Prediction request failed"
                    )
            
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            
            # Determine health status based on latency and other factors
            status = HealthStatus.HEALTHY
            if latency_ms > 5000:  # 5 seconds
                status = HealthStatus.CRITICAL
            elif latency_ms > 2000:  # 2 seconds
                status = HealthStatus.WARNING
            
            return HealthCheckResult(
                endpoint_name=endpoint_name,
                status=status,
                timestamp=datetime.now().isoformat(),
                latency_ms=latency_ms,
                response_code=200,
                deployed_models_count=len(endpoint_info.deployed_models),
                traffic_distribution=endpoint_info.traffic_split
            )
            
        except Exception as e:
            return HealthCheckResult(
                endpoint_name=endpoint_name,
                status=HealthStatus.CRITICAL,
                timestamp=datetime.now().isoformat(),
                latency_ms=(time.time() - start_time) * 1000,
                error_message=f"Health check failed: {str(e)}"
            )
    
    def check_all_endpoints_health(self, 
                                  test_instances: Optional[List[Dict[str, Any]]] = None) -> List[HealthCheckResult]:
        """
        Check health of all endpoints.
        
        Args:
            test_instances: Test instances for prediction health checks
            
        Returns:
            List of HealthCheckResult objects
        """
        results = []
        
        try:
            endpoints = self.endpoint_manager.list_endpoints()
            
            for endpoint_info in endpoints:
                result = self.check_endpoint_health(endpoint_info.name, test_instances)
                results.append(result)
                
        except Exception as e:
            logger.error(f"Failed to check all endpoints health: {e}")
        
        return results
    
    def get_health_summary(self, results: List[HealthCheckResult]) -> Dict[str, Any]:
        """
        Get summary of health check results.
        
        Args:
            results: List of HealthCheckResult objects
            
        Returns:
            Dictionary with health summary
        """
        total_endpoints = len(results)
        healthy_count = sum(1 for r in results if r.status == HealthStatus.HEALTHY)
        warning_count = sum(1 for r in results if r.status == HealthStatus.WARNING)
        critical_count = sum(1 for r in results if r.status == HealthStatus.CRITICAL)
        
        average_latency = statistics.mean([r.latency_ms for r in results]) if results else 0.0
        
        return {
            'timestamp': datetime.now().isoformat(),
            'total_endpoints': total_endpoints,
            'healthy_count': healthy_count,
            'warning_count': warning_count,
            'critical_count': critical_count,
            'health_percentage': (healthy_count / total_endpoints * 100) if total_endpoints > 0 else 0,
            'average_latency_ms': average_latency,
            'status_distribution': {
                'healthy': healthy_count,
                'warning': warning_count,
                'critical': critical_count
            }
        }


class PerformanceMonitor:
    """
    Performance monitor for model serving.
    
    Tracks performance metrics for deployed models and endpoints.
    """
    
    def __init__(self, project_id: str, location: str = "us-central1"):
        """
        Initialize performance monitor.
        
        Args:
            project_id: Google Cloud project ID
            location: Vertex AI location/region
        """
        self.project_id = project_id
        self.location = location
        self.metrics_history: Dict[str, List[PerformanceMetrics]] = {}
    
    def collect_metrics(self, endpoint_name: str, 
                       time_window_minutes: int = 5) -> PerformanceMetrics:
        """
        Collect performance metrics for an endpoint.
        
        Args:
            endpoint_name: Endpoint resource name or ID
            time_window_minutes: Time window for metrics collection
            
        Returns:
            PerformanceMetrics object
        """
        # In a real implementation, this would query Cloud Monitoring
        # For now, we'll generate mock metrics
        
        import random
        
        current_time = datetime.now()
        
        # Generate realistic mock metrics
        base_latency = 100 + random.uniform(-20, 50)  # Base latency around 100ms
        request_count = random.randint(10, 1000)
        error_count = random.randint(0, int(request_count * 0.05))  # Up to 5% error rate
        
        metrics = PerformanceMetrics(
            endpoint_name=endpoint_name,
            timestamp=current_time.isoformat(),
            request_count=request_count,
            error_count=error_count,
            average_latency_ms=base_latency,
            p95_latency_ms=base_latency * 1.5,
            p99_latency_ms=base_latency * 2.0,
            throughput_qps=request_count / (time_window_minutes * 60),
            cpu_utilization=random.uniform(20, 80),
            memory_utilization=random.uniform(30, 70),
            error_rate=(error_count / request_count * 100) if request_count > 0 else 0
        )
        
        # Store metrics in history
        if endpoint_name not in self.metrics_history:
            self.metrics_history[endpoint_name] = []
        
        self.metrics_history[endpoint_name].append(metrics)
        
        # Keep only last 100 metrics entries per endpoint
        if len(self.metrics_history[endpoint_name]) > 100:
            self.metrics_history[endpoint_name] = self.metrics_history[endpoint_name][-100:]
        
        return metrics
    
    def get_metrics_history(self, endpoint_name: str, 
                           hours_back: int = 24) -> List[PerformanceMetrics]:
        """
        Get metrics history for an endpoint.
        
        Args:
            endpoint_name: Endpoint resource name or ID
            hours_back: Number of hours of history to retrieve
            
        Returns:
            List of PerformanceMetrics objects
        """
        if endpoint_name not in self.metrics_history:
            return []
        
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        filtered_metrics = []
        for metric in self.metrics_history[endpoint_name]:
            metric_time = datetime.fromisoformat(metric.timestamp)
            if metric_time >= cutoff_time:
                filtered_metrics.append(metric)
        
        return filtered_metrics
    
    def analyze_performance_trends(self, endpoint_name: str, 
                                 hours_back: int = 24) -> Dict[str, Any]:
        """
        Analyze performance trends for an endpoint.
        
        Args:
            endpoint_name: Endpoint resource name or ID
            hours_back: Number of hours to analyze
            
        Returns:
            Dictionary with trend analysis
        """
        metrics = self.get_metrics_history(endpoint_name, hours_back)
        
        if len(metrics) < 2:
            return {
                'endpoint_name': endpoint_name,
                'status': 'insufficient_data',
                'message': 'Not enough data for trend analysis'
            }
        
        # Calculate trends
        latencies = [m.average_latency_ms for m in metrics]
        error_rates = [m.error_rate for m in metrics]
        throughputs = [m.throughput_qps for m in metrics]
        
        # Simple trend calculation (recent vs older)
        recent_count = max(1, len(metrics) // 3)
        recent_latency = statistics.mean(latencies[-recent_count:])
        older_latency = statistics.mean(latencies[:recent_count])
        
        recent_error_rate = statistics.mean(error_rates[-recent_count:])
        older_error_rate = statistics.mean(error_rates[:recent_count])
        
        recent_throughput = statistics.mean(throughputs[-recent_count:])
        older_throughput = statistics.mean(throughputs[:recent_count])
        
        return {
            'endpoint_name': endpoint_name,
            'analysis_period_hours': hours_back,
            'data_points': len(metrics),
            'latency_trend': {
                'recent_avg_ms': recent_latency,
                'older_avg_ms': older_latency,
                'change_percent': ((recent_latency - older_latency) / older_latency * 100) if older_latency > 0 else 0,
                'status': 'improving' if recent_latency < older_latency else 'degrading'
            },
            'error_rate_trend': {
                'recent_avg_percent': recent_error_rate,
                'older_avg_percent': older_error_rate,
                'change_percent': ((recent_error_rate - older_error_rate) / max(older_error_rate, 0.1)),
                'status': 'improving' if recent_error_rate < older_error_rate else 'degrading'
            },
            'throughput_trend': {
                'recent_avg_qps': recent_throughput,
                'older_avg_qps': older_throughput,
                'change_percent': ((recent_throughput - older_throughput) / max(older_throughput, 0.1)),
                'status': 'improving' if recent_throughput > older_throughput else 'degrading'
            },
            'overall_status': self._determine_overall_trend_status(recent_latency, older_latency, 
                                                                 recent_error_rate, older_error_rate)
        }
    
    def _determine_overall_trend_status(self, recent_latency: float, older_latency: float,
                                      recent_error_rate: float, older_error_rate: float) -> str:
        """Determine overall trend status based on metrics."""
        latency_improving = recent_latency < older_latency
        error_rate_improving = recent_error_rate < older_error_rate
        
        if latency_improving and error_rate_improving:
            return 'improving'
        elif not latency_improving and not error_rate_improving:
            return 'degrading'
        else:
            return 'mixed'


class AlertManager:
    """
    Alert manager for deployment monitoring.
    
    Manages alerts for deployment issues and performance problems.
    """
    
    def __init__(self):
        """Initialize alert manager."""
        self.alerts: List[Alert] = []
        self.alert_rules: List[Dict[str, Any]] = []
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Setup default alerting rules."""
        self.alert_rules = [
            {
                'name': 'high_latency',
                'condition': lambda metrics: metrics.average_latency_ms > 1000,
                'severity': AlertSeverity.WARNING,
                'title': 'High Latency Detected',
                'message_template': 'Average latency is {latency:.2f}ms (threshold: 1000ms)'
            },
            {
                'name': 'critical_latency',
                'condition': lambda metrics: metrics.average_latency_ms > 5000,
                'severity': AlertSeverity.CRITICAL,
                'title': 'Critical Latency Detected',
                'message_template': 'Average latency is {latency:.2f}ms (threshold: 5000ms)'
            },
            {
                'name': 'high_error_rate',
                'condition': lambda metrics: metrics.error_rate > 5.0,
                'severity': AlertSeverity.ERROR,
                'title': 'High Error Rate Detected',
                'message_template': 'Error rate is {error_rate:.2f}% (threshold: 5%)'
            },
            {
                'name': 'critical_error_rate',
                'condition': lambda metrics: metrics.error_rate > 10.0,
                'severity': AlertSeverity.CRITICAL,
                'title': 'Critical Error Rate Detected',
                'message_template': 'Error rate is {error_rate:.2f}% (threshold: 10%)'
            },
            {
                'name': 'low_throughput',
                'condition': lambda metrics: metrics.throughput_qps < 0.1,
                'severity': AlertSeverity.WARNING,
                'title': 'Low Throughput Detected',
                'message_template': 'Throughput is {throughput:.2f} QPS (threshold: 0.1 QPS)'
            }
        ]
    
    def check_alerts(self, metrics: PerformanceMetrics) -> List[Alert]:
        """
        Check if metrics trigger any alerts.
        
        Args:
            metrics: PerformanceMetrics to check
            
        Returns:
            List of triggered alerts
        """
        triggered_alerts = []
        
        for rule in self.alert_rules:
            try:
                if rule['condition'](metrics):
                    alert_id = f"{rule['name']}_{metrics.endpoint_name}_{int(time.time())}"
                    
                    message = rule['message_template'].format(
                        latency=metrics.average_latency_ms,
                        error_rate=metrics.error_rate,
                        throughput=metrics.throughput_qps
                    )
                    
                    alert = Alert(
                        id=alert_id,
                        endpoint_name=metrics.endpoint_name,
                        severity=rule['severity'],
                        title=rule['title'],
                        message=message,
                        timestamp=datetime.now().isoformat()
                    )
                    
                    triggered_alerts.append(alert)
                    self.alerts.append(alert)
                    
            except Exception as e:
                logger.error(f"Error checking alert rule {rule['name']}: {e}")
        
        return triggered_alerts
    
    def resolve_alert(self, alert_id: str):
        """
        Resolve an alert.
        
        Args:
            alert_id: Alert ID to resolve
        """
        for alert in self.alerts:
            if alert.id == alert_id and not alert.resolved:
                alert.resolved = True
                alert.resolution_time = datetime.now().isoformat()
                logger.info(f"Resolved alert: {alert_id}")
                break
    
    def get_active_alerts(self, endpoint_name: Optional[str] = None) -> List[Alert]:
        """
        Get active (unresolved) alerts.
        
        Args:
            endpoint_name: Optional endpoint name filter
            
        Returns:
            List of active alerts
        """
        active_alerts = [alert for alert in self.alerts if not alert.resolved]
        
        if endpoint_name:
            active_alerts = [alert for alert in active_alerts if alert.endpoint_name == endpoint_name]
        
        return active_alerts
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """
        Get alert summary statistics.
        
        Returns:
            Dictionary with alert summary
        """
        active_alerts = self.get_active_alerts()
        
        severity_counts = {severity.value: 0 for severity in AlertSeverity}
        for alert in active_alerts:
            severity_counts[alert.severity.value] += 1
        
        return {
            'timestamp': datetime.now().isoformat(),
            'total_alerts': len(self.alerts),
            'active_alerts': len(active_alerts),
            'resolved_alerts': len(self.alerts) - len(active_alerts),
            'severity_distribution': severity_counts,
            'endpoints_with_alerts': len(set(alert.endpoint_name for alert in active_alerts))
        }


class DeploymentMonitor:
    """
    Main deployment monitoring class.
    
    Orchestrates health checking, performance monitoring, and alerting
    for deployed models.
    """
    
    def __init__(self, project_id: str, location: str = "us-central1"):
        """
        Initialize deployment monitor.
        
        Args:
            project_id: Google Cloud project ID
            location: Vertex AI location/region
        """
        self.project_id = project_id
        self.location = location
        
        # Initialize components
        self.endpoint_manager = EndpointManager(project_id, location)
        self.health_checker = HealthChecker(self.endpoint_manager)
        self.performance_monitor = PerformanceMonitor(project_id, location)
        self.alert_manager = AlertManager()
        
        logger.info(f"Deployment monitor initialized for {project_id}")
    
    def run_monitoring_cycle(self, endpoints: Optional[List[str]] = None,
                           test_instances: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Run a complete monitoring cycle.
        
        Args:
            endpoints: Optional list of specific endpoints to monitor
            test_instances: Optional test instances for health checks
            
        Returns:
            Dictionary with monitoring results
        """
        start_time = time.time()
        results = {
            'timestamp': datetime.now().isoformat(),
            'monitoring_duration_seconds': 0,
            'health_checks': [],
            'performance_metrics': [],
            'alerts': [],
            'summary': {}
        }
        
        try:
            # Get endpoints to monitor
            if endpoints:
                endpoint_names = endpoints
            else:
                endpoint_infos = self.endpoint_manager.list_endpoints()
                endpoint_names = [info.name for info in endpoint_infos]
            
            # Perform health checks
            health_results = []
            for endpoint_name in endpoint_names:
                health_result = self.health_checker.check_endpoint_health(endpoint_name, test_instances)
                health_results.append(health_result)
            
            results['health_checks'] = [
                {
                    'endpoint_name': r.endpoint_name,
                    'status': r.status.value,
                    'latency_ms': r.latency_ms,
                    'error_message': r.error_message
                } for r in health_results
            ]
            
            # Collect performance metrics and check alerts
            performance_results = []
            alert_results = []
            
            for endpoint_name in endpoint_names:
                # Collect metrics
                metrics = self.performance_monitor.collect_metrics(endpoint_name)
                performance_results.append(metrics)
                
                # Check alerts
                triggered_alerts = self.alert_manager.check_alerts(metrics)
                alert_results.extend(triggered_alerts)
            
            results['performance_metrics'] = [
                {
                    'endpoint_name': m.endpoint_name,
                    'average_latency_ms': m.average_latency_ms,
                    'error_rate': m.error_rate,
                    'throughput_qps': m.throughput_qps
                } for m in performance_results
            ]
            
            results['alerts'] = [
                {
                    'id': a.id,
                    'endpoint_name': a.endpoint_name,
                    'severity': a.severity.value,
                    'title': a.title,
                    'message': a.message
                } for a in alert_results
            ]
            
            # Generate summary
            health_summary = self.health_checker.get_health_summary(health_results)
            alert_summary = self.alert_manager.get_alert_summary()
            
            results['summary'] = {
                'endpoints_monitored': len(endpoint_names),
                'health_summary': health_summary,
                'alert_summary': alert_summary,
                'new_alerts': len(alert_results)
            }
            
        except Exception as e:
            logger.error(f"Error during monitoring cycle: {e}")
            results['error'] = str(e)
        
        finally:
            results['monitoring_duration_seconds'] = time.time() - start_time
        
        return results
    
    def get_monitoring_dashboard_data(self, hours_back: int = 24) -> Dict[str, Any]:
        """
        Get data for monitoring dashboard.
        
        Args:
            hours_back: Hours of historical data to include
            
        Returns:
            Dictionary with dashboard data
        """
        dashboard_data = {
            'timestamp': datetime.now().isoformat(),
            'time_range_hours': hours_back,
            'endpoints': [],
            'overall_stats': {
                'total_endpoints': 0,
                'healthy_endpoints': 0,
                'active_alerts': 0
            }
        }
        
        try:
            # Get all endpoints
            endpoint_infos = self.endpoint_manager.list_endpoints()
            
            for endpoint_info in endpoint_infos:
                endpoint_name = endpoint_info.name
                
                # Get latest health check
                health_result = self.health_checker.check_endpoint_health(endpoint_name)
                
                # Get performance trends
                performance_trends = self.performance_monitor.analyze_performance_trends(
                    endpoint_name, hours_back
                )
                
                # Get active alerts
                active_alerts = self.alert_manager.get_active_alerts(endpoint_name)
                
                endpoint_data = {
                    'name': endpoint_name,
                    'display_name': endpoint_info.display_name,
                    'health_status': health_result.status.value,
                    'last_check_latency_ms': health_result.latency_ms,
                    'deployed_models_count': len(endpoint_info.deployed_models),
                    'performance_trends': performance_trends,
                    'active_alerts_count': len(active_alerts),
                    'alert_severities': [alert.severity.value for alert in active_alerts]
                }
                
                dashboard_data['endpoints'].append(endpoint_data)
                
                # Update overall stats
                dashboard_data['overall_stats']['total_endpoints'] += 1
                if health_result.status == HealthStatus.HEALTHY:
                    dashboard_data['overall_stats']['healthy_endpoints'] += 1
                dashboard_data['overall_stats']['active_alerts'] += len(active_alerts)
            
        except Exception as e:
            logger.error(f"Error generating dashboard data: {e}")
            dashboard_data['error'] = str(e)
        
        return dashboard_data


def create_deployment_monitor(project_id: str, location: str = "us-central1") -> DeploymentMonitor:
    """
    Create a DeploymentMonitor instance.
    
    Args:
        project_id: Google Cloud project ID
        location: Vertex AI location
        
    Returns:
        DeploymentMonitor instance
    """
    return DeploymentMonitor(project_id, location)
