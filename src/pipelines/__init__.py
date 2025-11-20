"""
Pipelines Package

This package provides pipeline orchestration capabilities for the MLOps workflow,
including both simple local execution and advanced Vertex AI Pipelines integration.

Modules:
    simple_orchestration: Simple local pipeline execution
    orchestration: Advanced Vertex AI Pipelines integration

Classes:
    SimplePipeline: Basic pipeline execution
    LocalPipelineRunner: Local pipeline runner
    PipelineManager: Vertex AI pipeline manager

Author: MLOps Team
Version: 1.0.0
"""

from .simple_orchestration import (
    SimplePipeline,
    LocalPipelineRunner,
    PipelineStep,
    SimplePipelineConfig,
    PipelineResult,
    StepStatus,
    PipelineType,
    create_pipeline_runner,
    run_sample_pipeline
)

# Try to import advanced orchestration
try:
    from .orchestration import (
        PipelineConfig,
        PipelineManager,
        PipelineRun,
        PipelineStatus,
        ComponentType,
        create_pipeline_manager,
        validate_pipeline_config
    )
    ADVANCED_ORCHESTRATION_AVAILABLE = True
except ImportError:
    ADVANCED_ORCHESTRATION_AVAILABLE = False

__all__ = [
    # Simple Orchestration
    'SimplePipeline',
    'LocalPipelineRunner',
    'PipelineStep',
    'SimplePipelineConfig',
    'PipelineResult',
    'StepStatus',
    'PipelineType',
    'create_pipeline_runner',
    'run_sample_pipeline',
]

# Add advanced orchestration to exports if available
if ADVANCED_ORCHESTRATION_AVAILABLE:
    __all__.extend([
        'PipelineConfig',
        'PipelineManager',
        'PipelineRun',
        'PipelineStatus',
        'ComponentType',
        'create_pipeline_manager',
        'validate_pipeline_config'
    ])
