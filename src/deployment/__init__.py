"""
Deployment Package

This package provides comprehensive model deployment capabilities for
the MLOps pipeline including endpoint management, serving, and monitoring.

Modules:
    model_deployment: Core deployment functionality
    monitoring: Deployment monitoring and health checks

Main Classes:
    EndpointManager: Manage Vertex AI endpoints
    ModelServingManager: Handle model serving operations
    DeploymentMonitor: Monitor deployment health and performance

Author: MLOps Team
Version: 1.0.0
"""

from .model_deployment import (
    EndpointManager,
    ModelServingManager,
    ModelDeploymentConfig,
    EndpointInfo,
    PredictionRequest,
    PredictionResponse,
    DeploymentStatus,
    TrafficSplit,
    create_endpoint_manager,
    create_model_serving_manager,
    get_deployment_recommendations
)

from .monitoring import (
    DeploymentMonitor,
    HealthChecker,
    PerformanceMonitor,
    AlertManager,
    HealthStatus,
    AlertSeverity,
    HealthCheckResult,
    PerformanceMetrics,
    Alert,
    create_deployment_monitor
)

from .ab_testing import (
    ABTestConfig,
    ModelVariant,
    ABTestResult,
    ABTestReport,
    StatisticalComparison,
    ABTestManager,
    ABTestAnalyzer,
    ABTestStatus,
    ABTestType,
    StatisticalSignificance,
    create_ab_test_manager,
    validate_ab_test_setup
)

__all__ = [
    # Core deployment classes
    'EndpointManager',
    'ModelServingManager', 
    'ModelDeploymentConfig',
    'EndpointInfo',
    'PredictionRequest',
    'PredictionResponse',
    
    # Monitoring classes
    'DeploymentMonitor',
    'HealthChecker',
    'PerformanceMonitor',
    'AlertManager',
    
    # Enums
    'DeploymentStatus',
    'TrafficSplit',
    'HealthStatus',
    'AlertSeverity',
    
    # Data classes
    'HealthCheckResult',
    'PerformanceMetrics',
    'Alert',
    
    # Factory functions
    'create_endpoint_manager',
    'create_model_serving_manager',
    'create_deployment_monitor',
    'get_deployment_recommendations',
    
    # A/B Testing
    'ABTestConfig',
    'ModelVariant',
    'ABTestResult',
    'ABTestReport',
    'StatisticalComparison',
    'ABTestManager',
    'ABTestAnalyzer',
    'ABTestStatus',
    'ABTestType',
    'StatisticalSignificance',
    'create_ab_test_manager',
    'validate_ab_test_setup'
]
