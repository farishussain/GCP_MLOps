"""
Data validation utilities for MLOps pipeline.

This module provides data validation capabilities including:
- Data quality checks
- Schema validation
- Statistical validation
- Drift detection
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Data validation result."""
    passed: bool
    issues: List[str]
    warnings: List[str]
    metrics: Dict[str, Any]

class DataValidator:
    """Data validation utility class."""
    
    def __init__(self):
        self.validation_rules = {}
        
    def validate_schema(
        self,
        df: pd.DataFrame,
        expected_columns: List[str],
        required_columns: Optional[List[str]] = None
    ) -> ValidationResult:
        """
        Validate DataFrame schema.
        
        Args:
            df: DataFrame to validate.
            expected_columns: List of expected column names.
            required_columns: List of required column names.
            
        Returns:
            ValidationResult object.
        """
        issues = []
        warnings = []
        metrics = {}
        
        # Check columns
        actual_columns = set(df.columns)
        expected_columns_set = set(expected_columns)
        
        missing_columns = expected_columns_set - actual_columns
        extra_columns = actual_columns - expected_columns_set
        
        if missing_columns:
            issues.append(f"Missing columns: {list(missing_columns)}")
            
        if extra_columns:
            warnings.append(f"Extra columns: {list(extra_columns)}")
            
        # Check required columns
        if required_columns:
            required_set = set(required_columns)
            missing_required = required_set - actual_columns
            if missing_required:
                issues.append(f"Missing required columns: {list(missing_required)}")
        
        metrics.update({
            'total_columns': len(df.columns),
            'expected_columns': len(expected_columns),
            'missing_columns': len(missing_columns),
            'extra_columns': len(extra_columns)
        })
        
        return ValidationResult(
            passed=len(issues) == 0,
            issues=issues,
            warnings=warnings,
            metrics=metrics
        )
    
    def validate_data_quality(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate data quality.
        
        Args:
            df: DataFrame to validate.
            
        Returns:
            ValidationResult object.
        """
        issues = []
        warnings = []
        metrics = {}
        
        # Check for missing values
        missing_counts = df.isnull().sum()
        total_missing = missing_counts.sum()
        
        if total_missing > 0:
            missing_percentage = (total_missing / (len(df) * len(df.columns))) * 100
            if missing_percentage > 50:
                issues.append(f"High percentage of missing values: {missing_percentage:.2f}%")
            elif missing_percentage > 10:
                warnings.append(f"Moderate missing values: {missing_percentage:.2f}%")
        
        # Check for duplicate rows
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            duplicate_percentage = (duplicate_count / len(df)) * 100
            if duplicate_percentage > 10:
                issues.append(f"High percentage of duplicates: {duplicate_percentage:.2f}%")
            else:
                warnings.append(f"Found {duplicate_count} duplicate rows")
        
        # Check data types
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        categorical_columns = df.select_dtypes(include=['object']).columns
        
        metrics.update({
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': int(total_missing),
            'missing_percentage': float((total_missing / (len(df) * len(df.columns))) * 100),
            'duplicate_rows': int(duplicate_count),
            'numeric_columns': len(numeric_columns),
            'categorical_columns': len(categorical_columns)
        })
        
        return ValidationResult(
            passed=len(issues) == 0,
            issues=issues,
            warnings=warnings,
            metrics=metrics
        )
    
    def validate_statistical_properties(
        self,
        df: pd.DataFrame,
        reference_stats: Optional[Dict[str, Dict[str, float]]] = None
    ) -> ValidationResult:
        """
        Validate statistical properties of the data.
        
        Args:
            df: DataFrame to validate.
            reference_stats: Reference statistics for comparison.
            
        Returns:
            ValidationResult object.
        """
        issues = []
        warnings = []
        metrics = {}
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            col_stats = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'median': df[col].median()
            }
            
            metrics[col] = col_stats
            
            # Check for outliers using IQR method
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outlier_percentage = (len(outliers) / len(df)) * 100
            
            if outlier_percentage > 10:
                warnings.append(f"High percentage of outliers in {col}: {outlier_percentage:.2f}%")
            
            # Compare with reference statistics if provided
            if reference_stats and col in reference_stats:
                ref_mean = reference_stats[col]['mean']
                ref_std = reference_stats[col]['std']
                
                # Check for significant drift (using 2-sigma rule)
                mean_diff = abs(col_stats['mean'] - ref_mean)
                if mean_diff > 2 * ref_std:
                    warnings.append(f"Potential data drift in {col}: mean changed by {mean_diff:.4f}")
        
        return ValidationResult(
            passed=len(issues) == 0,
            issues=issues,
            warnings=warnings,
            metrics=metrics
        )
    
    def validate_target_distribution(
        self,
        y: pd.Series,
        reference_distribution: Optional[Dict[Any, float]] = None
    ) -> ValidationResult:
        """
        Validate target variable distribution.
        
        Args:
            y: Target variable series.
            reference_distribution: Reference distribution for comparison.
            
        Returns:
            ValidationResult object.
        """
        issues = []
        warnings = []
        metrics = {}
        
        # Calculate current distribution
        value_counts = y.value_counts()
        distribution = (value_counts / len(y)).to_dict()
        
        metrics['distribution'] = distribution
        metrics['num_classes'] = len(value_counts)
        metrics['class_balance_ratio'] = value_counts.min() / value_counts.max()
        
        # Check for class imbalance
        if metrics['class_balance_ratio'] < 0.1:
            warnings.append(f"Severe class imbalance detected: ratio {metrics['class_balance_ratio']:.3f}")
        elif metrics['class_balance_ratio'] < 0.3:
            warnings.append(f"Moderate class imbalance detected: ratio {metrics['class_balance_ratio']:.3f}")
        
        # Compare with reference distribution if provided
        if reference_distribution:
            for class_value, ref_proportion in reference_distribution.items():
                current_proportion = distribution.get(class_value, 0.0)
                diff = abs(current_proportion - ref_proportion)
                
                if diff > 0.1:  # 10% difference threshold
                    warnings.append(
                        f"Class {class_value} distribution changed: "
                        f"{ref_proportion:.3f} -> {current_proportion:.3f}"
                    )
        
        return ValidationResult(
            passed=len(issues) == 0,
            issues=issues,
            warnings=warnings,
            metrics=metrics
        )
    
    def run_full_validation(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        expected_columns: Optional[List[str]] = None,
        reference_stats: Optional[Dict[str, Any]] = None
    ) -> Dict[str, ValidationResult]:
        """
        Run complete data validation suite.
        
        Args:
            df: DataFrame to validate.
            target_column: Name of target column if present.
            expected_columns: Expected column names.
            reference_stats: Reference statistics for comparison.
            
        Returns:
            Dictionary of validation results.
        """
        logger.info("Running full data validation...")
        
        results = {}
        
        # Schema validation
        if expected_columns:
            results['schema'] = self.validate_schema(df, expected_columns)
        
        # Data quality validation
        results['data_quality'] = self.validate_data_quality(df)
        
        # Statistical validation
        feature_columns = [col for col in df.columns if col != target_column] if target_column else df.columns
        feature_df = df[feature_columns]
        
        feature_reference_stats = None
        if reference_stats and 'features' in reference_stats:
            feature_reference_stats = reference_stats['features']
            
        results['statistical'] = self.validate_statistical_properties(
            feature_df, feature_reference_stats
        )
        
        # Target validation if target column is provided
        if target_column and target_column in df.columns:
            target_reference_dist = None
            if reference_stats and 'target_distribution' in reference_stats:
                target_reference_dist = reference_stats['target_distribution']
                
            results['target'] = self.validate_target_distribution(
                df[target_column], target_reference_dist
            )
        
        # Summary
        total_issues = sum(len(result.issues) for result in results.values())
        total_warnings = sum(len(result.warnings) for result in results.values())
        
        logger.info(f"Validation complete: {total_issues} issues, {total_warnings} warnings")
        
        return results
    
    def generate_report(self, validation_results: Dict[str, ValidationResult]) -> str:
        """
        Generate a readable validation report.
        
        Args:
            validation_results: Validation results dictionary.
            
        Returns:
            Formatted report string.
        """
        report = ["Data Validation Report", "=" * 30, ""]
        
        overall_passed = all(result.passed for result in validation_results.values())
        status = "PASSED" if overall_passed else "FAILED"
        report.append(f"Overall Status: {status}\n")
        
        for check_name, result in validation_results.items():
            report.append(f"{check_name.title()} Validation:")
            report.append(f"  Status: {'PASSED' if result.passed else 'FAILED'}")
            
            if result.issues:
                report.append("  Issues:")
                for issue in result.issues:
                    report.append(f"    - {issue}")
                    
            if result.warnings:
                report.append("  Warnings:")
                for warning in result.warnings:
                    report.append(f"    - {warning}")
                    
            if result.metrics:
                report.append("  Metrics:")
                for metric, value in result.metrics.items():
                    if isinstance(value, (int, float)):
                        report.append(f"    {metric}: {value}")
                    elif isinstance(value, dict) and len(value) < 10:
                        report.append(f"    {metric}: {value}")
                        
            report.append("")
        
        return "\n".join(report)
