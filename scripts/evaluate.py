"""
Evaluation script for the semantic search engine
"""
import json
import time
import statistics
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
from loguru import logger
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

from src.searcher import SemanticSearcher

@dataclass
class EvaluationResult:
    """Evaluation results container"""
    query: str
    expected_results: List[str]
    actual_results: List[Dict[str, Any]]
    metrics: Dict[str, float]
    execution_time: float

class SearchEvaluator:
    """Evaluator for semantic search engine"""
    
    def __init__(self, test_data_path: str = "data/test_queries.json"):
        self.searcher = SemanticSearcher()
        self.test_data_path = Path(test_data_path)
        self.results = []
    
    def load_test_data(self) -> List[Dict[str, Any]]:
        """Load test data from file"""
        if not self.test_data_path.exists():
            logger.warning(f"Test data file not found: {self.test_data_path}")
            logger.info("Creating sample test data...")
            return self._create_sample_test_data()
        
        with open(self.test_data_path, 'r') as f:
            return json.load(f)
    
    def _create_sample_test_data(self) -> List[Dict[str, Any]]:
        """Create sample test data"""
        test_data = [
            {
                "query": "function for user authentication",
                "expected_files": ["auth.py", "security.py"],
                "description": "Find authentication-related functions"
            },
            {
                "query": "database connection handler",
                "expected_files": ["database.py", "db.py"],
                "description": "Find database connection code"
            },
            {
                "query": "error handling middleware",
                "expected_files": ["middleware.py", "error_handler.py"],
                "description": "Find error handling code"
            },
            {
                "query": "API endpoint registration",
                "expected_files": ["routes.py", "api.py"],
                "description": "Find API endpoint definitions"
            },
            {
                "query": "configuration file parsing",
                "expected_files": ["config.py", "settings.py"],
                "description": "Find configuration parsing code"
            },
            {
                "query": "unit test for login",
                "expected_files": ["test_auth.py", "test_login.py"],
                "description": "Find login-related tests"
            },
            {
                "query": "password hashing implementation",
                "expected_files": ["security.py", "hash_utils.py"],
                "description": "Find password hashing functions"
            },
            {
                "query": "file upload handling",
                "expected_files": ["upload.py", "file_handler.py"],
                "description": "Find file upload code"
            },
            {
                "query": "rate limiting logic",
                "expected_files": ["rate_limiter.py", "throttle.py"],
                "description": "Find rate limiting implementation"
            },
            {
                "query": "JSON response formatting",
                "expected_files": ["serializers.py", "response.py"],
                "description": "Find JSON response code"
            }
        ]
        
        # Save sample data
        self.test_data_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.test_data_path, 'w') as f:
            json.dump(test_data, f, indent=2)
        
        logger.info(f"Created sample test data at {self.test_data_path}")
        return test_data
    
    def evaluate_query(self, test_case: Dict[str, Any], 
                      top_k: int = 10) -> EvaluationResult:
        """Evaluate a single query"""
        query = test_case['query']
        expected_files = test_case.get('expected_files', [])
        
        logger.info(f"Evaluating query: '{query}'")
        
        # Execute search
        start_time = time.time()
        results = self.searcher.search(
            query=query,
            top_k=top_k,
            use_hybrid=True,
            include_explanation=False
        )
        execution_time = time.time() - start_time
        
        # Extract actual file names from results
        actual_files = []
        for result in results:
            file_name = Path(result.chunk.file_path).name
            actual_files.append({
                'file': file_name,
                'score': result.similarity_score,
                'rank': results.index(result) + 1
            })
        
        # Calculate metrics
        metrics = self._calculate_metrics(
            expected_files, 
            [item['file'] for item in actual_files[:5]]  # Use top 5 for evaluation
        )
        
        # Create evaluation result
        eval_result = EvaluationResult(
            query=query,
            expected_results=expected_files,
            actual_results=actual_files,
            metrics=metrics,
            execution_time=execution_time
        )
        
        return eval_result
    
    def _calculate_metrics(self, expected: List[str], actual: List[str]) -> Dict[str, float]:
        """Calculate evaluation metrics"""
        if not expected:
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'mrr': 0.0,
                'success_at_1': 0.0,
                'success_at_3': 0.0,
                'success_at_5': 0.0
            }
        
        # Calculate binary relevance for each actual result
        y_true = [1 if file in expected else 0 for file in actual]
        y_pred = [1] * len(actual)  # All results are predicted as relevant
        
        # Calculate precision, recall, f1
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )
        
        # Calculate MRR (Mean Reciprocal Rank)
        reciprocal_ranks = []
        for i, file in enumerate(actual, 1):
            if file in expected:
                reciprocal_ranks.append(1.0 / i)
        
        mrr = statistics.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
        
        # Calculate success@k
        success_at_1 = 1.0 if any(file in expected for file in actual[:1]) else 0.0
        success_at_3 = 1.0 if any(file in expected for file in actual[:3]) else 0.0
        success_at_5 = 1.0 if any(file in expected for file in actual[:5]) else 0.0
        
        return {
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'mrr': float(mrr),
            'success_at_1': success_at_1,
            'success_at_3': success_at_3,
            'success_at_5': success_at_5
        }
    
    def run_evaluation(self, num_queries: int = None) -> pd.DataFrame:
        """Run full evaluation"""
        test_cases = self.load_test_data()
        
        if num_queries:
            test_cases = test_cases[:num_queries]
        
        logger.info(f"Running evaluation on {len(test_cases)} test cases")
        
        self.results = []
        for test_case in test_cases:
            result = self.evaluate_query(test_case)
            self.results.append(result)
            
            # Log individual results
            logger.info(
                f"Query: '{test_case['query'][:50]}...' | "
                f"Time: {result.execution_time:.3f}s | "
                f"Precision: {result.metrics['precision']:.3f} | "
                f"MRR: {result.metrics['mrr']:.3f}"
            )
        
        # Calculate overall statistics
        overall_stats = self._calculate_overall_statistics()
        
        logger.info("\n" + "="*60)
        logger.info("EVALUATION SUMMARY")
        logger.info("="*60)
        for metric, value in overall_stats.items():
            logger.info(f"{metric:25}: {value:.3f}")
        
        # Create DataFrame
        df = self._create_results_dataframe()
        
        return df, overall_stats
    
    def _calculate_overall_statistics(self) -> Dict[str, float]:
        """Calculate overall evaluation statistics"""
        if not self.results:
            return {}
        
        metrics_list = [result.metrics for result in self.results]
        execution_times = [result.execution_time for result in self.results]
        
        stats = {
            'avg_precision': statistics.mean([m['precision'] for m in metrics_list]),
            'avg_recall': statistics.mean([m['recall'] for m in metrics_list]),
            'avg_f1_score': statistics.mean([m['f1_score'] for m in metrics_list]),
            'avg_mrr': statistics.mean([m['mrr'] for m in metrics_list]),
            'avg_success_at_1': statistics.mean([m['success_at_1'] for m in metrics_list]),
            'avg_success_at_3': statistics.mean([m['success_at_3'] for m in metrics_list]),
            'avg_success_at_5': statistics.mean([m['success_at_5'] for m in metrics_list]),
            'avg_execution_time': statistics.mean(execution_times),
            'median_execution_time': statistics.median(execution_times),
            'p95_execution_time': np.percentile(execution_times, 95),
            'total_queries': len(self.results),
            'queries_with_results': sum(1 for r in self.results if r.actual_results)
        }
        
        return stats
    
    def _create_results_dataframe(self) -> pd.DataFrame:
        """Create DataFrame from evaluation results"""
        data = []
        for result in self.results:
            row = {
                'query': result.query,
                'execution_time': result.execution_time,
                'num_expected': len(result.expected_results),
                'num_actual': len(result.actual_results),
                'has_results': len(result.actual_results) > 0
            }
            
            # Add metrics
            row.update(result.metrics)
            
            # Add top actual results
            for i, actual in enumerate(result.actual_results[:3], 1):
                row[f'top_{i}_file'] = actual['file']
                row[f'top_{i}_score'] = actual['score']
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def save_results(self, df: pd.DataFrame, stats: Dict[str, float], 
                    output_dir: str = "data/evaluation"):
        """Save evaluation results"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = output_dir / f"detailed_results_{timestamp}.csv"
        df.to_csv(results_file, index=False)
        
        # Save summary statistics
        stats_file = output_dir / f"summary_stats_{timestamp}.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Generate visualizations
        self._generate_visualizations(df, stats, output_dir, timestamp)
        
        # Generate report
        self._generate_report(df, stats, output_dir, timestamp)
        
        logger.info(f"Evaluation results saved to {output_dir}")
    
    def _generate_visualizations(self, df: pd.DataFrame, stats: Dict[str, float], 
                               output_dir: Path, timestamp: str):
        """Generate visualization plots"""
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 1. Metrics Comparison
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        metrics_to_plot = ['precision', 'recall', 'f1_score', 'mrr', 'success_at_1', 'success_at_3']
        
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx // 3, idx % 3]
            values = df[metric].dropna()
            
            ax.hist(values, bins=10, edgecolor='black', alpha=0.7)
            ax.axvline(values.mean(), color='red', linestyle='--', 
                      label=f'Mean: {values.mean():.3f}')
            ax.set_xlabel(metric.replace('_', ' ').title())
            ax.set_ylabel('Frequency')
            ax.set_title(f'{metric.replace("_", " ").title()} Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / f"metrics_distribution_{timestamp}.png", dpi=300)
        plt.close()
        
        # 2. Execution Time Analysis
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Histogram
        ax1.hist(df['execution_time'], bins=20, edgecolor='black', alpha=0.7)
        ax1.axvline(df['execution_time'].mean(), color='red', linestyle='--',
                   label=f'Mean: {df["execution_time"].mean():.3f}s')
        ax1.set_xlabel('Execution Time (s)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Query Execution Time Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Scatter plot: Time vs Precision
        ax2.scatter(df['execution_time'], df['precision'], alpha=0.7)
        ax2.set_xlabel('Execution Time (s)')
        ax2.set_ylabel('Precision')
        ax2.set_title('Execution Time vs Precision')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / f"performance_analysis_{timestamp}.png", dpi=300)
        plt.close()
        
        # 3. Summary Metrics Bar Chart
        summary_metrics = ['avg_precision', 'avg_recall', 'avg_f1_score', 'avg_mrr']
        summary_values = [stats[m] for m in summary_metrics]
        summary_labels = ['Precision', 'Recall', 'F1-Score', 'MRR']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(summary_labels, summary_values, color=['blue', 'green', 'orange', 'purple'])
        
        # Add value labels
        for bar, value in zip(bars, summary_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
        
        ax.set_ylabel('Score')
        ax.set_title('Overall Evaluation Metrics')
        ax.set_ylim(0, 1.1)
        ax.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / f"summary_metrics_{timestamp}.png", dpi=300)
        plt.close()
        
        # 4. Query Length Analysis
        df['query_length'] = df['query'].apply(len)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        axes = axes.flatten()
        
        # Query Length vs Execution Time
        axes[0].scatter(df['query_length'], df['execution_time'], alpha=0.7)
        axes[0].set_xlabel('Query Length (chars)')
        axes[0].set_ylabel('Execution Time (s)')
        axes[0].set_title('Query Length vs Execution Time')
        axes[0].grid(True, alpha=0.3)
        
        # Query Length vs Precision
        axes[1].scatter(df['query_length'], df['precision'], alpha=0.7)
        axes[1].set_xlabel('Query Length (chars)')
        axes[1].set_ylabel('Precision')
        axes[1].set_title('Query Length vs Precision')
        axes[1].grid(True, alpha=0.3)
        
        # Query Length Distribution
        axes[2].hist(df['query_length'], bins=20, edgecolor='black', alpha=0.7)
        axes[2].set_xlabel('Query Length (chars)')
        axes[2].set_ylabel('Frequency')
        axes[2].set_title('Query Length Distribution')
        axes[2].grid(True, alpha=0.3)
        
        # Correlation Heatmap
        corr_matrix = df[['query_length', 'execution_time', 'precision', 'mrr']].corr()
        im = axes[3].imshow(corr_matrix, cmap='coolwarm', aspect='auto')
        axes[3].set_xticks(range(len(corr_matrix.columns)))
        axes[3].set_yticks(range(len(corr_matrix.columns)))
        axes[3].set_xticklabels(corr_matrix.columns, rotation=45)
        axes[3].set_yticklabels(corr_matrix.columns)
        axes[3].set_title('Feature Correlation Heatmap')
        
        # Add correlation values
        for i in range(len(corr_matrix.columns)):
            for j in range(len(corr_matrix.columns)):
                text = axes[3].text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                  ha="center", va="center", color="w")
        
        plt.colorbar(im, ax=axes[3])
        plt.tight_layout()
        plt.savefig(output_dir / f"query_analysis_{timestamp}.png", dpi=300)
        plt.close()
    
    def _generate_report(self, df: pd.DataFrame, stats: Dict[str, float], 
                        output_dir: Path, timestamp: str):
        """Generate HTML evaluation report"""
        report_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Semantic Search Engine Evaluation Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                         color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; }
                .metrics-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin-bottom: 30px; }
                .metric-card { background: white; border-radius: 10px; padding: 20px; 
                             box-shadow: 0 5px 15px rgba(0,0,0,0.1); text-align: center; }
                .metric-value { font-size: 32px; font-weight: bold; margin: 10px 0; }
                .metric-label { color: #666; font-size: 14px; }
                .good { color: #4CAF50; }
                .average { color: #FF9800; }
                .poor { color: #F44336; }
                .section { margin-bottom: 40px; }
                table { width: 100%; border-collapse: collapse; }
                th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
                th { background-color: #f8f9fa; }
                tr:hover { background-color: #f5f5f5; }
                .images { display: flex; flex-wrap: wrap; gap: 20px; }
                .image-container { flex: 1; min-width: 300px; }
                img { width: 100%; border-radius: 8px; box-shadow: 0 3px 10px rgba(0,0,0,0.1); }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🔍 Semantic Search Engine Evaluation Report</h1>
                <p>Generated on {timestamp}</p>
            </div>
            
            <div class="section">
                <h2>📊 Overall Metrics</h2>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-label">Average Precision</div>
                        <div class="metric-value {precision_class}">{avg_precision:.3f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Average Recall</div>
                        <div class="metric-value {recall_class}">{avg_recall:.3f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">F1-Score</div>
                        <div class="metric-value {f1_class}">{avg_f1_score:.3f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">MRR</div>
                        <div class="metric-value {mrr_class}">{avg_mrr:.3f}</div>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>⚡ Performance Metrics</h2>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-label">Avg Query Time</div>
                        <div class="metric-value">{avg_execution_time:.3f}s</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Success@1</div>
                        <div class="metric-value {success1_class}">{avg_success_at_1:.3f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Success@3</div>
                        <div class="metric-value {success3_class}">{avg_success_at_3:.3f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Success@5</div>
                        <div class="metric-value {success5_class}">{avg_success_at_5:.3f}</div>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>📈 Detailed Results</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Query</th>
                            <th>Time (s)</th>
                            <th>Precision</th>
                            <th>Recall</th>
                            <th>F1</th>
                            <th>MRR</th>
                        </tr>
                    </thead>
                    <tbody>
                        {table_rows}
                    </tbody>
                </table>
            </div>
            
            <div class="section">
                <h2>📸 Visualizations</h2>
                <div class="images">
                    <div class="image-container">
                        <h3>Metrics Distribution</h3>
                        <img src="metrics_distribution_{timestamp}.png" alt="Metrics Distribution">
                    </div>
                    <div class="image-container">
                        <h3>Performance Analysis</h3>
                        <img src="performance_analysis_{timestamp}.png" alt="Performance Analysis">
                    </div>
                    <div class="image-container">
                        <h3>Summary Metrics</h3>
                        <img src="summary_metrics_{timestamp}.png" alt="Summary Metrics">
                    </div>
                    <div class="image-container">
                        <h3>Query Analysis</h3>
                        <img src="query_analysis_{timestamp}.png" alt="Query Analysis">
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>📋 Test Details</h2>
                <p><strong>Total Queries:</strong> {total_queries}</p>
                <p><strong>Queries with Results:</strong> {queries_with_results}</p>
                <p><strong>Success Rate:</strong> {success_rate:.1%}</p>
                <p><strong>Evaluation Date:</strong> {timestamp}</p>
            </div>
        </body>
        </html>
        """
        
        # Determine CSS classes based on metric values
        def get_class(value, good_threshold=0.7, average_threshold=0.5):
            if value >= good_threshold:
                return "good"
            elif value >= average_threshold:
                return "average"
            else:
                return "poor"
        
        # Generate table rows
        table_rows = ""
        for _, row in df.iterrows():
            table_rows += f"""
                <tr>
                    <td>{row['query'][:50]}...</td>
                    <td>{row['execution_time']:.3f}</td>
                    <td class="{get_class(row['precision'])}">{row['precision']:.3f}</td>
                    <td class="{get_class(row['recall'])}">{row['recall']:.3f}</td>
                    <td class="{get_class(row['f1_score'])}">{row['f1_score']:.3f}</td>
                    <td class="{get_class(row['mrr'])}">{row['mrr']:.3f}</td>
                </tr>
            """
        
        # Calculate success rate
        success_rate = stats['queries_with_results'] / stats['total_queries']
        
        # Fill template
        report_html = report_template.format(
            timestamp=timestamp,
            avg_precision=stats['avg_precision'],
            avg_recall=stats['avg_recall'],
            avg_f1_score=stats['avg_f1_score'],
            avg_mrr=stats['avg_mrr'],
            avg_execution_time=stats['avg_execution_time'],
            avg_success_at_1=stats['avg_success_at_1'],
            avg_success_at_3=stats['avg_success_at_3'],
            avg_success_at_5=stats['avg_success_at_5'],
            total_queries=stats['total_queries'],
            queries_with_results=stats['queries_with_results'],
            success_rate=success_rate,
            table_rows=table_rows,
            precision_class=get_class(stats['avg_precision']),
            recall_class=get_class(stats['avg_recall']),
            f1_class=get_class(stats['avg_f1_score']),
            mrr_class=get_class(stats['avg_mrr']),
            success1_class=get_class(stats['avg_success_at_1']),
            success3_class=get_class(stats['avg_success_at_3']),
            success5_class=get_class(stats['avg_success_at_5'])
        )
        
        # Save report
        report_file = output_dir / f"evaluation_report_{timestamp}.html"
        with open(report_file, 'w') as f:
            f.write(report_html)
        
        logger.info(f"Evaluation report saved to {report_file}")

def main():
    """Run evaluation"""
    evaluator = SearchEvaluator()
    
    # Run evaluation
    df, stats = evaluator.run_evaluation()
    
    # Save results
    evaluator.save_results(df, stats)
    
    logger.info("✅ Evaluation completed successfully!")

if __name__ == "__main__":
    main()
