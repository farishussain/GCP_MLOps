"""
Cloud Package

This package provides cloud integration capabilities for the MLOps pipeline,
including Google Cloud Platform services like Vertex AI, Cloud Storage, and more.

Modules:
    vertex_ai: Vertex AI integration for training and deployment
    storage_manager: Google Cloud Storage management
    distributed_training: Distributed training capabilities

Author: MLOps Team
Version: 1.0.0
"""

try:
    from .vertex_ai import CloudTrainingUtils
except ImportError:
    CloudTrainingUtils = None

try:
    from .storage_manager import GCSManager
except ImportError:
    GCSManager = None

try:
    from .distributed_training import DistributedTrainingManager
except ImportError:
    DistributedTrainingManager = None

__all__ = []

if CloudTrainingUtils:
    __all__.append('CloudTrainingUtils')

if GCSManager:
    __all__.append('GCSManager')

if DistributedTrainingManager:
    __all__.append('DistributedTrainingManager')
