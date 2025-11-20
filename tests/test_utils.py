"""
Tests for utility functions.
"""

import pytest
import tempfile
from pathlib import Path
from src.utils import setup_logging, ensure_directory, list_files, get_project_root


def test_setup_logging():
    """Test logging setup."""
    logger = setup_logging('INFO')
    assert logger is not None
    assert logger.name == 'src.utils'


def test_ensure_directory():
    """Test directory creation."""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_path = Path(temp_dir) / 'test' / 'nested' / 'directory'
        
        result = ensure_directory(str(test_path))
        
        assert result.exists()
        assert result.is_dir()
        assert result == test_path


def test_list_files():
    """Test file listing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test files
        (temp_path / 'file1.txt').touch()
        (temp_path / 'file2.py').touch()
        (temp_path / 'file3.txt').touch()
        
        # Test all files
        all_files = list_files(str(temp_path))
        assert len(all_files) == 3
        
        # Test pattern matching
        txt_files = list_files(str(temp_path), '*.txt')
        assert len(txt_files) == 2
        
        py_files = list_files(str(temp_path), '*.py')
        assert len(py_files) == 1


def test_list_files_nonexistent():
    """Test file listing for nonexistent directory."""
    result = list_files('/nonexistent/path')
    assert result == []


def test_get_project_root():
    """Test project root detection."""
    root = get_project_root()
    assert root.is_dir()
    assert root.name == 'GCP_MLOps' or 'src' in str(root)
