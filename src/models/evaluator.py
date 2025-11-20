"""
Model Evaluation and Visualization Module

This module provides comprehensive evaluation metrics and visualization tools
for machine learning models, including performance plots, confusion matrices,
and feature importance visualizations.

Classes:
    ModelEvaluator: Main class for model evaluation and visualization
    
Functions:
    plot_confusion_matrix: Create confusion matrix heatmaps
    plot_roc_curves: Plot ROC curves for multiple models
    plot_feature_importance: Visualize feature importance
    plot_model_comparison: Create comparison charts for multiple models

Author: MLOps Team  
Version: 1.0.0
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import logging
from datetime import datetime
from matplotlib.figure import Figure

# Scientific computing
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Local imports
from .trainer import EvaluationResults, ModelTrainer

logger = logging.getLogger(__name__)

# Set style for better-looking plots
plt.style.use('default')
sns.set_palette("husl")


class ModelEvaluator:
    """
    Comprehensive evaluation and visualization toolkit for machine learning models.
    
    Features:
    - Confusion matrix visualizations
    - ROC curve analysis  
    - Precision-Recall curves
    - Feature importance plots
    - Model performance comparisons
    - Detailed evaluation reports
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), style: str = 'seaborn-v0_8'):
        """
        Initialize the ModelEvaluator.
        
        Args:
            figsize: Default figure size for plots
            style: Matplotlib style to use
        """
        self.figsize = figsize
        try:
            plt.style.use(style)
        except OSError:
            plt.style.use('default')
            logger.warning(f"Style '{style}' not found, using 'default'")
        
        logger.info(f"ModelEvaluator initialized with figsize={figsize}")
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            class_names: Optional[List[str]] = None,
                            title: str = "Confusion Matrix",
                            normalize: bool = False,
                            save_path: Optional[str] = None) -> Figure:
        """
        Create a confusion matrix heatmap.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names of the classes
            title: Title for the plot
            normalize: Whether to normalize values
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Handle class names
        tick_labels = class_names if class_names is not None else True
        
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                   xticklabels=tick_labels, yticklabels=tick_labels, ax=ax)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        return fig
    
    def plot_roc_curves(self, models_data: Dict[str, Dict[str, Any]], 
                       title: str = "ROC Curves Comparison",
                       save_path: Optional[str] = None) -> Figure:
        """
        Plot ROC curves for multiple models.
        
        Args:
            models_data: Dictionary containing model data with 'y_true', 'y_proba', 'name'
            title: Title for the plot  
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot ROC curve for each model
        for model_name, data in models_data.items():
            y_true = data['y_true']
            y_proba = data['y_proba']
            
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            roc_auc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, linewidth=2, 
                   label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        # Plot diagonal line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.7, label='Random')
        
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curves saved to {save_path}")
        
        return fig
    
    def plot_precision_recall_curves(self, models_data: Dict[str, Dict[str, Any]],
                                   title: str = "Precision-Recall Curves",
                                   save_path: Optional[str] = None) -> Figure:
        """
        Plot Precision-Recall curves for multiple models.
        
        Args:
            models_data: Dictionary containing model data
            title: Title for the plot
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        for model_name, data in models_data.items():
            y_true = data['y_true']
            y_proba = data['y_proba']
            
            precision, recall, _ = precision_recall_curve(y_true, y_proba)
            avg_precision = average_precision_score(y_true, y_proba)
            
            ax.plot(recall, precision, linewidth=2,
                   label=f'{model_name} (AP = {avg_precision:.3f})')
        
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.legend(loc='lower left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Precision-recall curves saved to {save_path}")
        
        return fig
    
    def plot_feature_importance(self, feature_importance: Dict[str, float],
                              title: str = "Feature Importance",
                              top_n: int = 20,
                              save_path: Optional[str] = None) -> Optional[Figure]:
        """
        Plot feature importance as a horizontal bar chart.
        
        Args:
            feature_importance: Dictionary of feature names and importance scores
            title: Title for the plot
            top_n: Number of top features to display
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure object or None if no data
        """
        if not feature_importance:
            logger.warning("No feature importance data provided")
            return None
        
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), 
                               key=lambda x: abs(x[1]), reverse=True)[:top_n]
        
        features, importance = zip(*sorted_features)
        
        fig, ax = plt.subplots(figsize=(10, max(6, len(features) * 0.4)))
        
        # Create horizontal bar plot
        y_pos = np.arange(len(features))
        bars = ax.barh(y_pos, importance, alpha=0.8)
        
        # Color bars based on positive/negative importance
        for bar, imp in zip(bars, importance):
            if imp < 0:
                bar.set_color('red')
            else:
                bar.set_color('green')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()  # Top features at the top
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        
        return fig
    
    def plot_model_comparison(self, results: Dict[str, EvaluationResults],
                            metrics: List[str] = ['test_accuracy', 'precision', 'recall', 'f1_score'],
                            title: str = "Model Performance Comparison",
                            save_path: Optional[str] = None) -> Optional[Figure]:
        """
        Create a comparison chart for multiple models.
        
        Args:
            results: Dictionary of evaluation results
            metrics: List of metrics to compare
            title: Title for the plot
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure object or None if no data
        """
        if not results:
            logger.warning("No results provided for comparison")
            return None
        
        # Prepare data for plotting
        models = list(results.keys())
        metric_data = {metric: [] for metric in metrics}
        
        for model_name in models:
            result = results[model_name]
            for metric in metrics:
                value = getattr(result, metric, 0.0)
                metric_data[metric].append(value if value is not None else 0.0)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics[:4]):  # Limit to 4 metrics
            ax = axes[i]
            
            bars = ax.bar(models, metric_data[metric], alpha=0.8)
            
            # Color bars by performance
            max_val = max(metric_data[metric])
            for bar, value in zip(bars, metric_data[metric]):
                if value == max_val:
                    bar.set_color('gold')
                else:
                    bar.set_color('skyblue')
            
            ax.set_title(metric.replace('_', ' ').title(), fontsize=14, fontweight='bold')
            ax.set_ylabel('Score', fontsize=12)
            ax.set_ylim(0, 1.1 * max(metric_data[metric]))
            ax.grid(True, alpha=0.3, axis='y')
            
            # Rotate x-axis labels if needed
            if len(max(models, key=len)) > 10:
                ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, metric_data[metric]):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.suptitle(title, fontsize=18, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Model comparison plot saved to {save_path}")
        
        return fig
    
    def create_evaluation_report(self, results: Dict[str, EvaluationResults],
                               output_dir: str = "evaluation_reports") -> str:
        """
        Create a comprehensive evaluation report with visualizations.
        
        Args:
            results: Dictionary of evaluation results
            output_dir: Directory to save report files
            
        Returns:
            Path to the main report file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = output_path / f"evaluation_report_{timestamp}.md"
        
        # Create comparison plot
        comparison_plot = output_path / f"model_comparison_{timestamp}.png"
        self.plot_model_comparison(results, save_path=str(comparison_plot))
        
        # Generate markdown report
        with open(report_file, 'w') as f:
            f.write(f"# Model Evaluation Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Model Performance Summary\n\n")
            
            # Create summary table
            f.write("| Model | Algorithm | Test Accuracy | Precision | Recall | F1-Score | CV Mean ± Std |\n")
            f.write("|-------|-----------|---------------|-----------|--------|----------|---------------|\n")
            
            # Sort by test accuracy
            sorted_results = sorted(results.items(), 
                                  key=lambda x: x[1].test_accuracy, reverse=True)
            
            for name, result in sorted_results:
                f.write(f"| {result.model_name} | {result.algorithm} | "
                       f"{result.test_accuracy:.4f} | {result.precision:.4f} | "
                       f"{result.recall:.4f} | {result.f1_score:.4f} | "
                       f"{result.cross_val_mean:.4f} ± {result.cross_val_std:.4f} |\n")
            
            f.write(f"\n![Model Comparison](model_comparison_{timestamp}.png)\n\n")
            
            f.write("## Detailed Results\n\n")
            
            for name, result in sorted_results:
                f.write(f"### {result.model_name}\n\n")
                f.write(f"**Algorithm:** {result.algorithm}\n\n")
                
                if result.best_parameters:
                    f.write("**Best Parameters:**\n")
                    for param, value in result.best_parameters.items():
                        f.write(f"- {param}: {value}\n")
                    f.write("\n")
                
                f.write("**Performance Metrics:**\n")
                f.write(f"- Training Accuracy: {result.train_accuracy:.4f}\n")
                f.write(f"- Test Accuracy: {result.test_accuracy:.4f}\n")
                f.write(f"- Precision: {result.precision:.4f}\n")
                f.write(f"- Recall: {result.recall:.4f}\n")
                f.write(f"- F1-Score: {result.f1_score:.4f}\n")
                if result.roc_auc:
                    f.write(f"- ROC AUC: {result.roc_auc:.4f}\n")
                f.write(f"- Cross-Validation: {result.cross_val_mean:.4f} ± {result.cross_val_std:.4f}\n")
                f.write(f"- Training Time: {result.training_time:.2f}s\n")
                f.write(f"- Prediction Time: {result.prediction_time:.4f}s\n\n")
                
                if result.feature_importance:
                    importance_plot = output_path / f"feature_importance_{name.replace(' ', '_')}_{timestamp}.png"
                    self.plot_feature_importance(
                        result.feature_importance,
                        title=f"Feature Importance - {result.model_name}",
                        save_path=str(importance_plot)
                    )
                    f.write(f"![Feature Importance - {result.model_name}](feature_importance_{name.replace(' ', '_')}_{timestamp}.png)\n\n")
            
            f.write("## Recommendations\n\n")
            
            best_model = sorted_results[0][1]
            f.write(f"**Best Overall Model:** {best_model.model_name}\n")
            f.write(f"- Highest test accuracy: {best_model.test_accuracy:.4f}\n")
            f.write(f"- Stable cross-validation: {best_model.cross_val_mean:.4f} ± {best_model.cross_val_std:.4f}\n")
            
            if len(sorted_results) > 1:
                second_best = sorted_results[1][1]
                diff = best_model.test_accuracy - second_best.test_accuracy
                if diff < 0.01:
                    f.write(f"\n**Note:** The difference with {second_best.model_name} is minimal ({diff:.4f}). "
                           f"Consider ensemble methods or further hyperparameter tuning.\n")
        
        logger.info(f"Evaluation report created: {report_file}")
        return str(report_file)
    
    def plot_learning_curves(self, model, X_train: np.ndarray, y_train: np.ndarray,
                           title: str = "Learning Curves",
                           cv_folds: int = 5,
                           save_path: Optional[str] = None) -> Figure:
        """
        Plot learning curves showing training and validation scores vs. training size.
        
        Args:
            model: Trained model object
            X_train: Training features
            y_train: Training targets
            title: Title for the plot
            cv_folds: Number of cross-validation folds
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        from sklearn.model_selection import learning_curve
        
        train_sizes = np.linspace(0.1, 1.0, 10)
        
        learning_results = learning_curve(
            model, X_train, y_train, cv=cv_folds, n_jobs=-1,
            train_sizes=train_sizes, scoring='accuracy'
        )
        train_sizes_abs, train_scores, val_scores = learning_results[0], learning_results[1], learning_results[2]
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        ax.plot(train_sizes_abs, train_mean, 'o-', color='blue', label='Training Score')
        ax.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std,
                       alpha=0.3, color='blue')
        
        ax.plot(train_sizes_abs, val_mean, 'o-', color='red', label='Validation Score')
        ax.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std,
                       alpha=0.3, color='red')
        
        ax.set_xlabel('Training Size', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Learning curves saved to {save_path}")
        
        return fig


def create_model_evaluation_dashboard(trainer: ModelTrainer, 
                                   output_dir: str = "model_evaluation") -> str:
    """
    Create a comprehensive evaluation dashboard for all trained models.
    
    Args:
        trainer: Trained ModelTrainer instance
        output_dir: Directory to save dashboard files
        
    Returns:
        Path to the main dashboard file
    """
    evaluator = ModelEvaluator()
    
    if not trainer.results:
        raise ValueError("No trained models found. Train models first.")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create comprehensive evaluation report
    report_path = evaluator.create_evaluation_report(
        trainer.results, str(output_path)
    )
    
    logger.info(f"Model evaluation dashboard created at {output_path}")
    return report_path
