"""
Benchmarking script for the semantic search engine
"""
import time
import json
import statistics
from typing import List, Dict, Any
from pathlib import Path
from loguru import logger
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.searcher import SemanticSearcher

class Benchmark:
    """Benchmark the semantic search engine"""
    
    def __init__(self):
        self.searcher = SemanticSearcher()
        self.results = []
    
    def load_test_queries(self, queries_path: str = "data/test_queries.json") -> List[Dict[str, Any]]:
        """Load test queries from file"""
        if not Path(queries_path).exists():
            # Default test queries
            return [
                {"query": "function for user authentication", "expected_files": ["auth.py"]},
                {"query": "database connection handler", "expected_files": ["database.py"]},
                {"query": "email validation function", "expected_files": ["utils.py", "validation.py"]},
                {"query": "API endpoint for user registration", "expected_files": ["api.py", "routes.py"]},
                {"query": "error handling middleware", "expected_files": ["middleware.py", "error_handler.py"]},
                {"query": "configuration file parser", "expected_files": ["config.py"]},
                {"query": "unit test for authentication", "expected_files": ["test_auth.py"]},
                {"query": "class for handling file uploads", "expected_files": ["upload.py"]},
                {"query": "function to hash passwords", "expected_files": ["security.py", "auth.py"]},
                {"query": "rate limiting implementation", "expected_files": ["rate_limiter.py"]},
            ]
        
        with open(queries_path, 'r') as f:
            return json.load(f)
    
    def run_benchmark(self, queries: List[Dict[str, Any]], 
                     iterations: int = 3) -> pd.DataFrame:
        """Run benchmark tests"""
        logger.info(f"Running benchmark with {len(queries)} queries, {iterations} iterations")
        
        all_results = []
        
        for query_data in queries:
            query = query_data['query']
            expected_files = query_data.get('expected_files', [])
            
            iteration_times = []
            iteration_results = []
            
            for i in range(iterations):
                start_time = time.time()
                
                results = self.searcher.search(
                    query=query,
                    top_k=10,
                    use_hybrid=True,
                    include_explanation=False
                )
                
                elapsed_time = time.time() - start_time
                iteration_times.append(elapsed_time)
                iteration_results.append(results)
            
            # Calculate metrics
            avg_time = statistics.mean(iteration_times)
            std_time = statistics.stdev(iteration_times) if len(iteration_times) > 1 else 0
            
            # Calculate precision@k
            precision_scores = []
            for results in iteration_results:
                if results:
                    # Check if expected files are in results
                    found_files = set()
                    for result in results[:5]:  # Precision@5
                        file_name = Path(result.chunk.file_path).name
                        found_files.add(file_name)
                    
                    # Calculate precision
                    relevant_found = len([f for f in expected_files if f in found_files])
                    precision = relevant_found / min(5, len(expected_files)) if expected_files else 0
                    precision_scores.append(precision)
            
            avg_precision = statistics.mean(precision_scores) if precision_scores else 0
            
            # Calculate MRR (Mean Reciprocal Rank)
            mrr_scores = []
            for results in iteration_results:
                if results and expected_files:
                    for rank, result in enumerate(results, 1):
                        file_name = Path(result.chunk.file_path).name
                        if file_name in expected_files:
                            mrr_scores.append(1.0 / rank)
                            break
                    else:
                        mrr_scores.append(0.0)
            
            avg_mrr = statistics.mean(mrr_scores) if mrr_scores else 0
            
            # Store results
            result = {
                'query': query,
                'expected_files': expected_files,
                'avg_time_ms': avg_time * 1000,
                'std_time_ms': std_time * 1000,
                'avg_precision': avg_precision,
                'avg_mrr': avg_mrr,
                'num_results': len(iteration_results[0]) if iteration_results else 0,
                'iterations': iterations
            }
            
            all_results.append(result)
            logger.info(f"Query: '{query[:50]}...' | Time: {avg_time*1000:.1f}ms | Precision: {avg_precision:.3f} | MRR: {avg_mrr:.3f}")
        
        # Create DataFrame
        df = pd.DataFrame(all_results)
        
        # Calculate overall statistics
        overall_stats = {
            'avg_query_time_ms': df['avg_time_ms'].mean(),
            'median_query_time_ms': df['avg_time_ms'].median(),
            'p95_query_time_ms': df['avg_time_ms'].quantile(0.95),
            'avg_precision': df['avg_precision'].mean(),
            'avg_mrr': df['avg_mrr'].mean(),
            'total_queries': len(df),
            'success_rate': (df['num_results'] > 0).mean()
        }
        
        logger.info("\n" + "="*50)
        logger.info("BENCHMARK RESULTS SUMMARY")
        logger.info("="*50)
        for key, value in overall_stats.items():
            logger.info(f"{key:30}: {value:.3f}")
        
        return df, overall_stats
    
    def save_results(self, df: pd.DataFrame, stats: Dict[str, float], 
                    output_dir: str = "data/benchmark"):
        """Save benchmark results"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save DataFrame
        df.to_csv(output_dir / "detailed_results.csv", index=False)
        
        # Save summary statistics
        with open(output_dir / "summary_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Generate plots
        self._generate_plots(df, stats, output_dir)
        
        logger.info(f"Results saved to {output_dir}")
    
    def _generate_plots(self, df: pd.DataFrame, stats: Dict[str, float], output_dir: Path):
        """Generate visualization plots"""
        plt.style.use('seaborn-v0_8')
        
        # 1. Query time distribution
        plt.figure(figsize=(10, 6))
        plt.hist(df['avg_time_ms'], bins=20, edgecolor='black', alpha=0.7)
        plt.axvline(stats['avg_query_time_ms'], color='red', linestyle='--', label=f"Mean: {stats['avg_query_time_ms']:.1f}ms")
        plt.axvline(stats['p95_query_time_ms'], color='orange', linestyle='--', label=f"P95: {stats['p95_query_time_ms']:.1f}ms")
        plt.xlabel('Query Time (ms)')
        plt.ylabel('Frequency')
        plt.title('Query Time Distribution')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "query_time_distribution.png", dpi=300)
        plt.close()
        
        # 2. Precision vs MRR scatter plot
        plt.figure(figsize=(10, 6))
        plt.scatter(df['avg_precision'], df['avg_mrr'], alpha=0.7, s=100)
        
        # Add labels for outliers
        for i, row in df.iterrows():
            if row['avg_precision'] > 0.8 or row['avg_mrr'] > 0.8:
                plt.annotate(row['query'][:20] + '...', 
                           (row['avg_precision'], row['avg_mrr']),
                           fontsize=8)
        
        plt.xlabel('Average Precision@5')
        plt.ylabel('Mean Reciprocal Rank (MRR)')
        plt.title('Precision vs MRR')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "precision_vs_mrr.png", dpi=300)
        plt.close()
        
        # 3. Performance by query length
        df['query_length'] = df['query'].apply(len)
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.scatter(df['query_length'], df['avg_time_ms'], alpha=0.7)
        plt.xlabel('Query Length (chars)')
        plt.ylabel('Query Time (ms)')
        plt.title('Query Time vs Length')
        
        plt.subplot(1, 2, 2)
        plt.scatter(df['query_length'], df['avg_precision'], alpha=0.7)
        plt.xlabel('Query Length (chars)')
        plt.ylabel('Precision@5')
        plt.title('Precision vs Query Length')
        
        plt.tight_layout()
        plt.savefig(output_dir / "performance_by_query_length.png", dpi=300)
        plt.close()
        
        # 4. Summary bar chart
        summary_metrics = ['avg_query_time_ms', 'avg_precision', 'avg_mrr', 'success_rate']
        summary_values = [stats[m] for m in summary_metrics]
        summary_labels = ['Avg Time (ms)', 'Precision@5', 'MRR', 'Success Rate']
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(summary_labels, summary_values, color=['blue', 'green', 'orange', 'red'])
        
        # Add value labels
        for bar, value in zip(bars, summary_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.ylabel('Score')
        plt.title('Overall Benchmark Results')
        plt.ylim(0, 1.1)
        plt.tight_layout()
        plt.savefig(output_dir / "summary_results.png", dpi=300)
        plt.close()

def main():
    """Run benchmark"""
    benchmark = Benchmark()
    
    # Load test queries
    queries = benchmark.load_test_queries()
    
    # Run benchmark
    df, stats = benchmark.run_benchmark(queries, iterations=5)
    
    # Save results
    benchmark.save_results(df, stats)
    
    logger.info("✅ Benchmark completed successfully!")

if __name__ == "__main__":
    main()
