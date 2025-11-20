"""
Data loading utilities for the MLOps pipeline.

This module provides classes and functions for loading data from various sources
including local files, Google Cloud Storage, and databases.
"""

import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from io import BytesIO
from google.cloud import storage
from sklearn.datasets import load_iris
from sklearn.utils import Bunch

from ..config import config
from ..utils import setup_logging

logger = setup_logging()


class DataLoader:
    """
    Data loader for handling various data sources.
    
    Supports loading from:
    - Local files (CSV, NPZ, Pickle)
    - Google Cloud Storage
    - Built-in datasets (Iris, etc.)
    """
    
    def __init__(self, project_id: Optional[str] = None, bucket_name: Optional[str] = None):
        """
        Initialize DataLoader.
        
        Args:
            project_id (str): Google Cloud project ID.
            bucket_name (str): GCS bucket name.
        """
        self.project_id = project_id or config.project_id
        self.bucket_name = bucket_name or config.bucket_name
        
        if self.project_id and self.bucket_name:
            try:
                self.storage_client = storage.Client(project=self.project_id)
                self.bucket = self.storage_client.bucket(self.bucket_name)
                logger.info(f"Initialized GCS client for project: {self.project_id}")
            except Exception as e:
                logger.warning(f"Failed to initialize GCS client: {e}")
                self.storage_client = None
                self.bucket = None
    
    def load_iris_dataset(self, include_target_names: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Load the Iris dataset.
        
        Args:
            include_target_names (bool): Whether to include target names in DataFrame.
            
        Returns:
            Tuple[pd.DataFrame, Dict[str, Any]]: DataFrame and metadata.
        """
        logger.info("Loading Iris dataset...")
        
        iris_data = load_iris()
        
        # Create DataFrame
        df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
        df['target'] = iris_data.target
        
        if include_target_names:
            df['target_name'] = df['target'].map(
                {i: name for i, name in enumerate(iris_data.target_names)}
            )
        
        # Create metadata
        metadata = {
            'dataset_name': 'iris',
            'description': 'Iris flower classification dataset',
            'n_samples': len(iris_data.data),
            'n_features': iris_data.data.shape[1],
            'n_classes': len(iris_data.target_names),
            'feature_names': list(iris_data.feature_names),
            'target_names': list(iris_data.target_names),
            'source': 'sklearn.datasets'
        }
        
        logger.info(f"Loaded Iris dataset: {metadata['n_samples']} samples, "
                   f"{metadata['n_features']} features, {metadata['n_classes']} classes")
        
        return df, metadata
    
    def load_from_csv(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            file_path (str): Path to CSV file (local or GCS).
            **kwargs: Additional arguments for pd.read_csv.
            
        Returns:
            pd.DataFrame: Loaded DataFrame.
        """
        if file_path.startswith('gs://'):
            # Load from GCS
            return self._load_csv_from_gcs(file_path, **kwargs)
        else:
            # Load from local file
            logger.info(f"Loading CSV from local file: {file_path}")
            return pd.read_csv(file_path, **kwargs)
    
    def load_from_numpy(self, file_path: str) -> Dict[str, np.ndarray]:
        """
        Load data from NumPy .npz file.
        
        Args:
            file_path (str): Path to .npz file (local or GCS).
            
        Returns:
            Dict[str, np.ndarray]: Dictionary of arrays.
        """
        if file_path.startswith('gs://'):
            # Load from GCS
            return self._load_numpy_from_gcs(file_path)
        else:
            # Load from local file
            logger.info(f"Loading NumPy arrays from: {file_path}")
            return dict(np.load(file_path))
    
    def load_from_pickle(self, file_path: str) -> Any:
        """
        Load data from pickle file.
        
        Args:
            file_path (str): Path to pickle file (local or GCS).
            
        Returns:
            Any: Pickled object.
        """
        if file_path.startswith('gs://'):
            # Load from GCS
            return self._load_pickle_from_gcs(file_path)
        else:
            # Load from local file
            logger.info(f"Loading pickle from: {file_path}")
            with open(file_path, 'rb') as f:
                return pickle.load(f)
    
    def _load_csv_from_gcs(self, gcs_path: str, **kwargs) -> pd.DataFrame:
        """Load CSV from Google Cloud Storage."""
        if not self.bucket:
            raise RuntimeError("GCS client not initialized")
        
        # Remove gs://bucket/ prefix
        blob_name = gcs_path.replace(f'gs://{self.bucket_name}/', '')
        blob = self.bucket.blob(blob_name)
        
        logger.info(f"Loading CSV from GCS: {gcs_path}")
        
        # Download to memory and read with pandas
        content = blob.download_as_bytes()
        return pd.read_csv(BytesIO(content), **kwargs)
    
    def _load_numpy_from_gcs(self, gcs_path: str) -> Dict[str, np.ndarray]:
        """Load NumPy arrays from Google Cloud Storage."""
        if not self.bucket:
            raise RuntimeError("GCS client not initialized")
        
        # Remove gs://bucket/ prefix
        blob_name = gcs_path.replace(f'gs://{self.bucket_name}/', '')
        blob = self.bucket.blob(blob_name)
        
        logger.info(f"Loading NumPy arrays from GCS: {gcs_path}")
        
        # Download to temporary file and load
        import tempfile
        with tempfile.NamedTemporaryFile() as tmp:
            blob.download_to_filename(tmp.name)
            return dict(np.load(tmp.name))
    
    def _load_pickle_from_gcs(self, gcs_path: str) -> Any:
        """Load pickle from Google Cloud Storage."""
        if not self.bucket:
            raise RuntimeError("GCS client not initialized")
        
        # Remove gs://bucket/ prefix
        blob_name = gcs_path.replace(f'gs://{self.bucket_name}/', '')
        blob = self.bucket.blob(blob_name)
        
        logger.info(f"Loading pickle from GCS: {gcs_path}")
        
        # Download to memory and unpickle
        content = blob.download_as_bytes()
        return pickle.loads(content)
    
    def save_to_gcs(self, data: Any, gcs_path: str, format: str = 'pickle') -> None:
        """
        Save data to Google Cloud Storage.
        
        Args:
            data (Any): Data to save.
            gcs_path (str): GCS path (gs://bucket/path).
            format (str): Save format ('pickle', 'csv', 'numpy').
        """
        if not self.bucket:
            raise RuntimeError("GCS client not initialized")
        
        # Remove gs://bucket/ prefix
        blob_name = gcs_path.replace(f'gs://{self.bucket_name}/', '')
        blob = self.bucket.blob(blob_name)
        
        logger.info(f"Saving data to GCS: {gcs_path} (format: {format})")
        
        if format == 'pickle':
            # Save as pickle
            blob.upload_from_string(pickle.dumps(data))
        elif format == 'csv':
            # Save as CSV (assumes data is DataFrame)
            if not isinstance(data, pd.DataFrame):
                raise ValueError("Data must be DataFrame for CSV format")
            blob.upload_from_string(data.to_csv(index=False))
        elif format == 'numpy':
            # Save as NumPy (assumes data is dict of arrays)
            import tempfile
            with tempfile.NamedTemporaryFile() as tmp:
                if isinstance(data, dict):
                    np.savez(tmp.name, **data)
                else:
                    np.save(tmp.name, data)
                tmp.seek(0)
                blob.upload_from_filename(tmp.name)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Successfully saved data to GCS: {gcs_path}")
