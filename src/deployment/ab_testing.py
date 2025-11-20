"""
A/B Testing Framework for Model Deployment

This module provides comprehensive A/B testing capabilities for comparing
model performance, analyzing statistical significance, and managing traffic splits.

Classes:
    ABTestConfig: Configuration for A/B testing experiments
    ABTestAnalyzer: Statistical analysis for A/B test results
    ABTestManager: Manager for A/B testing experiments
    ABTestReport: Comprehensive reporting for A/B test outcomes

Author: MLOps Team
Version: 1.0.0
"""

import json
import logging
import time
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
import math

# Local imports
from .model_deployment import EndpointManager, ModelServingManager
from ..utils import setup_logging

logger = logging.getLogger(__name__)


class ABTestStatus(Enum):
    """A/B test status enumeration."""
    PLANNED = "planned"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class StatisticalSignificance(Enum):
    """Statistical significance levels."""
    VERY_HIGH = 0.99    # 99% confidence
    HIGH = 0.95         # 95% confidence
    MEDIUM = 0.90       # 90% confidence
    LOW = 0.80          # 80% confidence


class ABTestType(Enum):
    """Types of A/B tests."""
    CHAMPION_CHALLENGER = "champion_challenger"
    MULTI_VARIANT = "multi_variant"
    CANARY = "canary"
    BLUE_GREEN = "blue_green"


@dataclass
class ModelVariant:
    """Configuration for a model variant in A/B test."""
    name: str
    model_id: str
    endpoint_id: str
    traffic_percentage: int
    description: str = ""
    is_champion: bool = False
    deployment_config: Optional[Dict[str, Any]] = None


@dataclass
class ABTestConfig:
    """Configuration for A/B testing experiment."""
    test_name: str
    test_type: ABTestType
    variants: List[ModelVariant]
    success_metric: str = "accuracy"
    significance_level: StatisticalSignificance = StatisticalSignificance.HIGH
    min_sample_size: int = 1000
    max_duration_hours: int = 168  # 1 week
    early_stopping_enabled: bool = True
    ramp_up_duration_hours: int = 24
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetricResult(NamedTuple):
    """Result of a metric calculation."""
    value: float
    sample_count: int
    confidence_interval: Tuple[float, float]
    standard_error: float


@dataclass
class ABTestResult:
    """Results from an A/B test variant."""
    variant_name: str
    requests_count: int
    success_count: int
    error_count: int
    avg_latency_ms: float
    avg_response_time_ms: float
    success_rate: float
    error_rate: float
    throughput_rps: float
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


@dataclass
class StatisticalComparison:
    """Statistical comparison between variants."""
    variant_a: str
    variant_b: str
    metric: str
    value_a: float
    value_b: float
    difference: float
    percent_difference: float
    p_value: float
    is_significant: bool
    confidence_level: float
    confidence_interval: Tuple[float, float]
    effect_size: float


@dataclass
class ABTestReport:
    """Comprehensive A/B test report."""
    test_name: str
    test_status: ABTestStatus
    start_time: datetime
    end_time: Optional[datetime]
    duration_hours: float
    total_requests: int
    results_by_variant: Dict[str, ABTestResult]
    statistical_comparisons: List[StatisticalComparison]
    winner: Optional[str] = None
    confidence_level: float = 0.0
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ABTestAnalyzer:
    """
    Statistical analyzer for A/B test results.
    
    Provides statistical significance testing, confidence intervals,
    and effect size calculations for A/B test comparisons.
    """
    
    def __init__(self, significance_level: StatisticalSignificance = StatisticalSignificance.HIGH):
        """
        Initialize A/B test analyzer.
        
        Args:
            significance_level: Required significance level for tests
        """
        self.significance_level = significance_level.value
        self.z_score_table = {
            0.99: 2.576,
            0.95: 1.960,
            0.90: 1.645,
            0.80: 1.282
        }
    
    def calculate_conversion_rate(self, successes: int, total: int) -> MetricResult:
        """
        Calculate conversion rate with confidence interval.
        
        Args:
            successes: Number of successful outcomes
            total: Total number of trials
            
        Returns:
            MetricResult with conversion rate and confidence interval
        """
        if total == 0:
            return MetricResult(0.0, 0, (0.0, 0.0), 0.0)
        
        rate = successes / total
        z_score = self.z_score_table.get(self.significance_level, 1.960)
        
        # Standard error for proportion
        se = math.sqrt(rate * (1 - rate) / total) if total > 0 else 0.0
        
        # Confidence interval
        margin_error = z_score * se
        ci_lower = max(0.0, rate - margin_error)
        ci_upper = min(1.0, rate + margin_error)
        
        return MetricResult(rate, total, (ci_lower, ci_upper), se)
    
    def calculate_mean_metric(self, values: List[float]) -> MetricResult:
        """
        Calculate mean metric with confidence interval.
        
        Args:
            values: List of metric values
            
        Returns:
            MetricResult with mean and confidence interval
        """
        if not values:
            return MetricResult(0.0, 0, (0.0, 0.0), 0.0)
        
        mean = statistics.mean(values)
        sample_count = len(values)
        
        if sample_count < 2:
            return MetricResult(mean, sample_count, (mean, mean), 0.0)
        
        stdev = statistics.stdev(values)
        se = stdev / math.sqrt(sample_count)
        
        z_score = self.z_score_table.get(self.significance_level, 1.960)
        margin_error = z_score * se
        
        ci_lower = mean - margin_error
        ci_upper = mean + margin_error
        
        return MetricResult(mean, sample_count, (ci_lower, ci_upper), se)
    
    def compare_conversion_rates(self, 
                               successes_a: int, total_a: int,
                               successes_b: int, total_b: int,
                               variant_a: str = "A", variant_b: str = "B") -> StatisticalComparison:
        """
        Compare conversion rates between two variants.
        
        Args:
            successes_a: Successes for variant A
            total_a: Total trials for variant A
            successes_b: Successes for variant B  
            total_b: Total trials for variant B
            variant_a: Name of variant A
            variant_b: Name of variant B
            
        Returns:
            StatisticalComparison object
        """
        # Calculate conversion rates
        rate_a = successes_a / total_a if total_a > 0 else 0.0
        rate_b = successes_b / total_b if total_b > 0 else 0.0
        
        # Calculate pooled proportion for test statistic
        pooled_successes = successes_a + successes_b
        pooled_total = total_a + total_b
        pooled_rate = pooled_successes / pooled_total if pooled_total > 0 else 0.0
        
        # Standard error for difference
        if pooled_total > 0 and pooled_rate > 0 and pooled_rate < 1:
            se_diff = math.sqrt(pooled_rate * (1 - pooled_rate) * (1/total_a + 1/total_b))
        else:
            se_diff = 0.0
        
        # Test statistic (z-score)
        difference = rate_b - rate_a
        z_stat = difference / se_diff if se_diff > 0 else 0.0
        
        # P-value (two-tailed test)
        p_value = 2 * (1 - self._normal_cdf(abs(z_stat))) if se_diff > 0 else 1.0
        
        # Significance test
        is_significant = p_value < (1 - self.significance_level)
        
        # Percent difference
        percent_diff = (difference / rate_a * 100) if rate_a > 0 else 0.0
        
        # Effect size (Cohen's h for proportions)
        effect_size = self._cohens_h(rate_a, rate_b)
        
        # Confidence interval for difference
        z_score = self.z_score_table.get(self.significance_level, 1.960)
        margin_error = z_score * se_diff
        ci_lower = difference - margin_error
        ci_upper = difference + margin_error
        
        return StatisticalComparison(
            variant_a=variant_a,
            variant_b=variant_b,
            metric="conversion_rate",
            value_a=rate_a,
            value_b=rate_b,
            difference=difference,
            percent_difference=percent_diff,
            p_value=p_value,
            is_significant=is_significant,
            confidence_level=self.significance_level,
            confidence_interval=(ci_lower, ci_upper),
            effect_size=effect_size
        )
    
    def compare_means(self, values_a: List[float], values_b: List[float],
                     variant_a: str = "A", variant_b: str = "B",
                     metric_name: str = "metric") -> StatisticalComparison:
        """
        Compare means between two variants using t-test.
        
        Args:
            values_a: Values for variant A
            values_b: Values for variant B
            variant_a: Name of variant A
            variant_b: Name of variant B
            metric_name: Name of the metric being compared
            
        Returns:
            StatisticalComparison object
        """
        if not values_a or not values_b:
            return StatisticalComparison(
                variant_a=variant_a,
                variant_b=variant_b,
                metric=metric_name,
                value_a=0.0,
                value_b=0.0,
                difference=0.0,
                percent_difference=0.0,
                p_value=1.0,
                is_significant=False,
                confidence_level=self.significance_level,
                confidence_interval=(0.0, 0.0),
                effect_size=0.0
            )
        
        mean_a = statistics.mean(values_a)
        mean_b = statistics.mean(values_b)
        difference = mean_b - mean_a
        
        # Welch's t-test (unequal variances)
        var_a = statistics.variance(values_a) if len(values_a) > 1 else 0.0
        var_b = statistics.variance(values_b) if len(values_b) > 1 else 0.0
        
        n_a = len(values_a)
        n_b = len(values_b)
        
        # Standard error of difference
        se_diff = math.sqrt(var_a/n_a + var_b/n_b) if (var_a + var_b) > 0 else 0.0
        
        # T-statistic
        t_stat = difference / se_diff if se_diff > 0 else 0.0
        
        # Degrees of freedom (Welch's formula)
        if se_diff > 0:
            df = (var_a/n_a + var_b/n_b)**2 / ((var_a/n_a)**2/(n_a-1) + (var_b/n_b)**2/(n_b-1))
        else:
            df = 1
        
        # Approximate p-value using normal distribution for large samples
        p_value = 2 * (1 - self._normal_cdf(abs(t_stat))) if se_diff > 0 else 1.0
        
        # Significance test
        is_significant = p_value < (1 - self.significance_level)
        
        # Percent difference
        percent_diff = (difference / mean_a * 100) if mean_a != 0 else 0.0
        
        # Effect size (Cohen's d)
        pooled_std = math.sqrt((var_a + var_b) / 2) if (var_a + var_b) > 0 else 0.0
        effect_size = difference / pooled_std if pooled_std > 0 else 0.0
        
        # Confidence interval for difference
        z_score = self.z_score_table.get(self.significance_level, 1.960)
        margin_error = z_score * se_diff
        ci_lower = difference - margin_error
        ci_upper = difference + margin_error
        
        return StatisticalComparison(
            variant_a=variant_a,
            variant_b=variant_b,
            metric=metric_name,
            value_a=mean_a,
            value_b=mean_b,
            difference=difference,
            percent_difference=percent_diff,
            p_value=p_value,
            is_significant=is_significant,
            confidence_level=self.significance_level,
            confidence_interval=(ci_lower, ci_upper),
            effect_size=effect_size
        )
    
    def calculate_minimum_sample_size(self, baseline_rate: float, 
                                    minimum_detectable_effect: float,
                                    power: float = 0.8) -> int:
        """
        Calculate minimum sample size for A/B test.
        
        Args:
            baseline_rate: Expected baseline conversion rate
            minimum_detectable_effect: Minimum effect size to detect (as percentage)
            power: Statistical power (1 - beta)
            
        Returns:
            Minimum sample size per variant
        """
        alpha = 1 - self.significance_level
        beta = 1 - power
        
        z_alpha = self.z_score_table.get(self.significance_level, 1.960)
        z_beta = self.z_score_table.get(power, 0.842)  # Approximate
        
        p1 = baseline_rate
        p2 = baseline_rate * (1 + minimum_detectable_effect / 100)
        
        # Ensure p2 is valid probability
        p2 = min(p2, 1.0)
        
        p_avg = (p1 + p2) / 2
        
        # Sample size formula
        numerator = (z_alpha * math.sqrt(2 * p_avg * (1 - p_avg)) + 
                    z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2)))**2
        denominator = (p2 - p1)**2
        
        if denominator > 0:
            n = math.ceil(numerator / denominator)
        else:
            n = 1000  # Default minimum
        
        return max(n, 100)  # Minimum 100 samples
    
    def _normal_cdf(self, z: float) -> float:
        """Approximate normal CDF using error function."""
        # Using approximation for normal CDF
        return 0.5 * (1 + math.erf(z / math.sqrt(2)))
    
    def _cohens_h(self, p1: float, p2: float) -> float:
        """Calculate Cohen's h effect size for proportions."""
        if p1 < 0 or p1 > 1 or p2 < 0 or p2 > 1:
            return 0.0
        
        # Arcsine transformation
        phi1 = 2 * math.asin(math.sqrt(p1)) if p1 <= 1 else 0
        phi2 = 2 * math.asin(math.sqrt(p2)) if p2 <= 1 else 0
        
        return phi2 - phi1


class ABTestManager:
    """
    Manager for A/B testing experiments.
    
    Handles test configuration, execution, monitoring, and analysis
    for model deployment A/B tests.
    """
    
    def __init__(self, project_id: str, location: str = "us-central1"):
        """
        Initialize A/B test manager.
        
        Args:
            project_id: Google Cloud project ID
            location: Vertex AI location/region
        """
        self.project_id = project_id
        self.location = location
        self.endpoint_manager = EndpointManager(project_id, location)
        self.serving_manager = ModelServingManager(project_id, location)
        self.analyzer = ABTestAnalyzer()
        
        # Active tests tracking
        self.active_tests: Dict[str, ABTestConfig] = {}
        self.test_results: Dict[str, Dict[str, ABTestResult]] = {}
    
    def create_ab_test(self, config: ABTestConfig) -> bool:
        """
        Create and start an A/B test.
        
        Args:
            config: ABTestConfig object
            
        Returns:
            True if test created successfully
        """
        try:
            # Validate configuration
            if not self._validate_config(config):
                logger.error("Invalid A/B test configuration")
                return False
            
            # Check if test already exists
            if config.test_name in self.active_tests:
                logger.error(f"Test {config.test_name} already exists")
                return False
            
            # Validate traffic split
            total_traffic = sum(v.traffic_percentage for v in config.variants)
            if total_traffic != 100:
                logger.error(f"Traffic split must total 100%, got {total_traffic}%")
                return False
            
            # Configure traffic split on endpoints
            for variant in config.variants:
                traffic_split = {variant.model_id: variant.traffic_percentage}
                success = self.endpoint_manager.update_traffic_split(
                    variant.endpoint_id, traffic_split
                )
                if not success:
                    logger.error(f"Failed to configure traffic for variant {variant.name}")
                    return False
            
            # Store test configuration
            self.active_tests[config.test_name] = config
            
            # Initialize results tracking
            self.test_results[config.test_name] = {}
            for variant in config.variants:
                self.test_results[config.test_name][variant.name] = ABTestResult(
                    variant_name=variant.name,
                    requests_count=0,
                    success_count=0,
                    error_count=0,
                    avg_latency_ms=0.0,
                    avg_response_time_ms=0.0,
                    success_rate=0.0,
                    error_rate=0.0,
                    throughput_rps=0.0,
                    start_time=datetime.now()
                )
            
            logger.info(f"Created A/B test: {config.test_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create A/B test: {e}")
            return False
    
    def get_test_results(self, test_name: str, 
                        end_time: Optional[datetime] = None) -> Optional[ABTestReport]:
        """
        Get results and analysis for an A/B test.
        
        Args:
            test_name: Name of the test
            end_time: Optional end time for analysis
            
        Returns:
            ABTestReport object or None
        """
        try:
            if test_name not in self.active_tests:
                logger.error(f"Test {test_name} not found")
                return None
            
            config = self.active_tests[test_name]
            results = self.test_results.get(test_name, {})
            
            if not results:
                logger.warning(f"No results found for test {test_name}")
                return None
            
            # Update results from monitoring data
            updated_results = self._collect_latest_metrics(test_name)
            
            # Statistical analysis
            comparisons = self._perform_statistical_analysis(config, updated_results)
            
            # Determine winner
            winner, confidence = self._determine_winner(comparisons, config.success_metric)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(config, updated_results, comparisons)
            
            # Calculate test duration
            start_time = min(r.start_time for r in updated_results.values() if r.start_time)
            end_time = end_time or datetime.now()
            duration = (end_time - start_time).total_seconds() / 3600
            
            # Total requests
            total_requests = sum(r.requests_count for r in updated_results.values())
            
            report = ABTestReport(
                test_name=test_name,
                test_status=ABTestStatus.RUNNING,  # Simplified
                start_time=start_time,
                end_time=end_time,
                duration_hours=duration,
                total_requests=total_requests,
                results_by_variant=updated_results,
                statistical_comparisons=comparisons,
                winner=winner,
                confidence_level=confidence,
                recommendations=recommendations,
                metadata=config.metadata
            )
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to get test results: {e}")
            return None
    
    def stop_test(self, test_name: str) -> bool:
        """
        Stop an A/B test and revert traffic to champion.
        
        Args:
            test_name: Name of the test to stop
            
        Returns:
            True if test stopped successfully
        """
        try:
            if test_name not in self.active_tests:
                logger.error(f"Test {test_name} not found")
                return False
            
            config = self.active_tests[test_name]
            
            # Find champion variant
            champion_variant = None
            for variant in config.variants:
                if variant.is_champion:
                    champion_variant = variant
                    break
            
            if not champion_variant:
                logger.error("No champion variant found")
                return False
            
            # Revert traffic to champion
            traffic_split = {champion_variant.model_id: 100}
            success = self.endpoint_manager.update_traffic_split(
                champion_variant.endpoint_id, traffic_split
            )
            
            if success:
                # Mark test as stopped
                del self.active_tests[test_name]
                logger.info(f"Stopped A/B test: {test_name}")
                return True
            else:
                logger.error(f"Failed to revert traffic for test: {test_name}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to stop test: {e}")
            return False
    
    def promote_winner(self, test_name: str, winner_variant: str) -> bool:
        """
        Promote winning variant to 100% traffic.
        
        Args:
            test_name: Name of the test
            winner_variant: Name of winning variant
            
        Returns:
            True if promotion successful
        """
        try:
            if test_name not in self.active_tests:
                logger.error(f"Test {test_name} not found")
                return False
            
            config = self.active_tests[test_name]
            
            # Find winner variant
            winner = None
            for variant in config.variants:
                if variant.name == winner_variant:
                    winner = variant
                    break
            
            if not winner:
                logger.error(f"Winner variant {winner_variant} not found")
                return False
            
            # Promote winner to 100% traffic
            traffic_split = {winner.model_id: 100}
            success = self.endpoint_manager.update_traffic_split(
                winner.endpoint_id, traffic_split
            )
            
            if success:
                logger.info(f"Promoted {winner_variant} to 100% traffic")
                return True
            else:
                logger.error(f"Failed to promote winner {winner_variant}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to promote winner: {e}")
            return False
    
    def _validate_config(self, config: ABTestConfig) -> bool:
        """Validate A/B test configuration."""
        if not config.test_name:
            return False
        
        if not config.variants:
            return False
        
        if len(config.variants) < 2:
            return False
        
        # Check for champion variant
        champions = [v for v in config.variants if v.is_champion]
        if len(champions) != 1:
            logger.error("Exactly one champion variant required")
            return False
        
        return True
    
    def _collect_latest_metrics(self, test_name: str) -> Dict[str, ABTestResult]:
        """Collect latest metrics from monitoring systems."""
        # Simplified implementation - would integrate with monitoring
        results = {}
        config = self.active_tests[test_name]
        
        for variant in config.variants:
            # Get serving stats
            stats = self.serving_manager.get_serving_stats(variant.endpoint_id)
            
            # Create updated result
            result = ABTestResult(
                variant_name=variant.name,
                requests_count=stats.get('request_count', 0),
                success_count=stats.get('request_count', 0) - stats.get('error_count', 0),
                error_count=stats.get('error_count', 0),
                avg_latency_ms=stats.get('average_latency_ms', 0.0),
                avg_response_time_ms=stats.get('average_latency_ms', 0.0),
                success_rate=1.0 - (stats.get('error_count', 0) / max(stats.get('request_count', 1), 1)),
                error_rate=stats.get('error_count', 0) / max(stats.get('request_count', 1), 1),
                throughput_rps=stats.get('request_count', 0) / 3600,  # Simplified
                start_time=datetime.now() - timedelta(hours=1)  # Simplified
            )
            
            results[variant.name] = result
        
        return results
    
    def _perform_statistical_analysis(self, config: ABTestConfig, 
                                    results: Dict[str, ABTestResult]) -> List[StatisticalComparison]:
        """Perform statistical analysis between variants."""
        comparisons = []
        
        # Find champion
        champion = None
        for variant in config.variants:
            if variant.is_champion:
                champion = variant.name
                break
        
        if not champion:
            return comparisons
        
        champion_result = results.get(champion)
        if not champion_result:
            return comparisons
        
        # Compare each challenger to champion
        for variant in config.variants:
            if variant.name != champion and variant.name in results:
                challenger_result = results[variant.name]
                
                # Compare success rates
                comparison = self.analyzer.compare_conversion_rates(
                    champion_result.success_count, champion_result.requests_count,
                    challenger_result.success_count, challenger_result.requests_count,
                    champion, variant.name
                )
                
                comparisons.append(comparison)
        
        return comparisons
    
    def _determine_winner(self, comparisons: List[StatisticalComparison], 
                         success_metric: str) -> Tuple[Optional[str], float]:
        """Determine test winner based on statistical comparisons."""
        if not comparisons:
            return None, 0.0
        
        # Find best performing variant
        best_variant = None
        best_improvement = 0.0
        best_confidence = 0.0
        
        for comparison in comparisons:
            if comparison.is_significant and comparison.difference > best_improvement:
                best_variant = comparison.variant_b
                best_improvement = comparison.difference
                best_confidence = comparison.confidence_level
        
        return best_variant, best_confidence
    
    def _generate_recommendations(self, config: ABTestConfig, 
                                results: Dict[str, ABTestResult],
                                comparisons: List[StatisticalComparison]) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        # Check sample sizes
        min_samples = config.min_sample_size
        for variant_name, result in results.items():
            if result.requests_count < min_samples:
                recommendations.append(
                    f"Variant {variant_name} needs more data: "
                    f"{result.requests_count}/{min_samples} samples"
                )
        
        # Check for significant results
        significant_results = [c for c in comparisons if c.is_significant]
        if significant_results:
            best = max(significant_results, key=lambda x: x.difference)
            recommendations.append(
                f"Statistically significant improvement found: "
                f"{best.variant_b} beats {best.variant_a} by "
                f"{best.percent_difference:.1f}%"
            )
        else:
            recommendations.append("No statistically significant differences found yet")
        
        # Check test duration
        total_requests = sum(r.requests_count for r in results.values())
        if total_requests < config.min_sample_size * len(config.variants):
            recommendations.append("Continue test to reach minimum sample size")
        
        return recommendations


def create_ab_test_manager(project_id: str, location: str = "us-central1") -> ABTestManager:
    """
    Create an ABTestManager instance.
    
    Args:
        project_id: Google Cloud project ID
        location: Vertex AI location
        
    Returns:
        ABTestManager instance
    """
    return ABTestManager(project_id, location)


def validate_ab_test_setup(config: ABTestConfig) -> Dict[str, Any]:
    """
    Validate A/B test setup and provide recommendations.
    
    Args:
        config: A/B test configuration
        
    Returns:
        Dictionary with validation results and recommendations
    """
    validation_result = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'recommendations': []
    }
    
    # Check traffic split
    total_traffic = sum(v.traffic_percentage for v in config.variants)
    if total_traffic != 100:
        validation_result['is_valid'] = False
        validation_result['errors'].append(f"Traffic split totals {total_traffic}%, must be 100%")
    
    # Check for champion
    champions = [v for v in config.variants if v.is_champion]
    if len(champions) != 1:
        validation_result['is_valid'] = False
        validation_result['errors'].append("Exactly one champion variant required")
    
    # Check minimum sample size
    if config.min_sample_size < 100:
        validation_result['warnings'].append("Minimum sample size is very low, consider increasing")
    
    # Check test duration
    if config.max_duration_hours > 720:  # 30 days
        validation_result['warnings'].append("Test duration is very long, consider shorter duration")
    
    # Traffic allocation recommendations
    challenger_variants = [v for v in config.variants if not v.is_champion]
    if challenger_variants:
        challenger_traffic = sum(v.traffic_percentage for v in challenger_variants)
        if challenger_traffic > 50:
            validation_result['recommendations'].append(
                "Consider reducing challenger traffic to minimize risk"
            )
    
    return validation_result
