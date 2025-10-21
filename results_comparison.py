"""
Results Comparison and Analysis Script

This script compares benchmark results across different models,
generates comparison tables, and creates statistical reports.

Usage:
    python scripts/compare_results.py --results_dir results/
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List
import numpy as np
from scipy import stats
import pandas as pd
from tabulate import tabulate


class ResultsComparator:
    """Compare and analyze benchmark results across multiple models."""
    
    def __init__(self, results_dir: str):
        """
        Initialize comparator.
        
        Args:
            results_dir: Directory containing result JSON files
        """
        self.results_dir = Path(results_dir)
        self.results = self._load_all_results()
    
    def _load_all_results(self) -> Dict:
        """Load all result files from directory."""
        results = {}
        
        for file in self.results_dir.glob('*.json'):
            try:
                with open(file) as f:
                    data = json.load(f)
                    model_name = data.get('model', file.stem)
                    results[model_name] = data
            except Exception as e:
                print(f"Error loading {file}: {e}")
        
        print(f"Loaded results for {len(results)} models")
        return results
    
    def generate_comparison_table(self, metrics: List[str] = None) -> pd.DataFrame:
        """
        Generate comparison table for specified metrics.
        
        Args:
            metrics: List of metric names to compare
            
        Returns:
            DataFrame with comparison results
        """
        if metrics is None:
            metrics = [
                'hallucination_rate', 'accuracy', 'precision', 
                'recall', 'f1_score', 'bertscore'
            ]
        
        data = []
        for model_name, result in self.results.items():
            row = {'Model': model_name}
            for metric in metrics:
                value = result['metrics'].get(metric, 'N/A')
                if isinstance(value, float):
                    row[metric] = f"{value:.4f}"
                else:
                    row[metric] = value
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Sort by F1 score if available
        if 'f1_score' in df.columns:
            df = df.sort_values('f1_score', ascending=False)
        
        return df
    
    def statistical_comparison(self) -> Dict:
        """
        Perform statistical comparison between models.
        
        Returns:
            Dictionary with statistical test results
        """
        stats_results = {
            'pairwise_ttests': {},
            'effect_sizes': {},
            'rankings': {}
        }
        
        models = list(self.results.keys())
        
        # Pairwise t-tests
        for i, model1 in enumerate(models):
            for model2 in models[i+1:]:
                # Get prediction accuracies
                acc1 = [
                    1 if p['predicted_hallucination'] == p['actual_hallucination'] 
                    else 0
                    for p in self.results[model1].get('predictions', [])
                ]
                acc2 = [
                    1 if p['predicted_hallucination'] == p['actual_hallucination'] 
                    else 0
                    for p in self.results[model2].get('predictions', [])
                ]
                
                if acc1 and acc2:
                    # Perform t-test
                    t_stat, p_value = stats.ttest_ind(acc1, acc2)
                    
                    # Calculate effect size (Cohen's d)
                    mean_diff = np.mean(acc1) - np.mean(acc2)
                    pooled_std = np.sqrt(
                        (np.std(acc1)**2 + np.std(acc2)**2) / 2
                    )
                    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
                    
                    pair = f"{model1} vs {model2}"
                    stats_results['pairwise_ttests'][pair] = {
                        't_statistic': float(t_stat),
                        'p_value': float(p_value),
                        'significant': p_value < 0.05
                    }
                    stats_results['effect_sizes'][pair] = float(cohens_d)
        
        # Rankings by different metrics
        for metric in ['accuracy', 'f1_score', 'bertscore']:
            values = []
            for model in models:
                val = self.results[model]['metrics'].get(metric, 0)
                values.append((model, val))
            
            # Sort descending
            values.sort(key=lambda x: x[1], reverse=True)
            stats_results['rankings'][metric] = [
                {'rank': i+1, 'model': m, 'value': v} 
                for i, (m, v) in enumerate(values)
            ]
        
        return stats_results
    
    def error_analysis_comparison(self) -> pd.DataFrame:
        """Compare error patterns across models."""
        error_data = []
        
        for model_name, result in self.results.items():
            # Count error types
            errors = result.get('error_analysis', {}).get('error_types', {})
            
            row = {'Model': model_name}
            row.update(errors)
            error_data.append(row)
        
        df = pd.DataFrame(error_data)
        return df
    
    def generate_latex_table(self, df: pd.DataFrame) -> str:
        """
        Generate LaTeX table from DataFrame.
        
        Args:
            df: DataFrame to convert
            
        Returns:
            LaTeX table string
        """
        # Format floats
        def format_cell(x):
            if isinstance(x, str) and x.replace('.', '').replace('-', '').isdigit():
                try:
                    return f"{float(x):.3f}"
                except:
                    return x
            return x
        
        df_formatted = df.applymap(format_cell)
        
        latex = df_formatted.to_latex(
            index=False,
            column_format='l' + 'c' * (len(df.columns) - 1),
            escape=False,
            caption='Benchmark Results Comparison',
            label='tab:results'
        )
        
        return latex
    
    def generate_report(self, output_file: str = 'comparison_report.md'):
        """
        Generate comprehensive comparison report.
        
        Args:
            output_file: Path to output markdown file
        """
        report = []
        report.append("# Benchmark Results Comparison Report\n")
        report.append(f"Generated: {pd.Timestamp.now()}\n")
        report.append(f"Models Compared: {len(self.results)}\n\n")
        
        # Overall comparison
        report.append("## Overall Performance Comparison\n")
        df = self.generate_comparison_table()
        report.append(tabulate(df, headers='keys', tablefmt='github', showindex=False))
        report.append("\n\n")
        
        # Statistical tests
        report.append("## Statistical Analysis\n")
        stats_results = self.statistical_comparison()
        
        report.append("### Pairwise Significance Tests\n")
        for pair, result in stats_results['pairwise_ttests'].items():
            sig = "✓ Significant" if result['significant'] else "✗ Not significant"
            report.append(
                f"- **{pair}**: p={result['p_value']:.4f} ({sig})\n"
            )
        report.append("\n")
        
        report.append("### Model Rankings\n")
        for metric, rankings in stats_results['rankings'].items():
            report.append(f"\n**{metric.upper()}:**\n")
            for r in rankings:
                report.append(
                    f"{r['rank']}. {r['model']}: {r['value']:.4f}\n"
                )
        report.append("\n")
        
        # Error analysis
        report.append("## Error Type Comparison\n")
        error_df = self.error_analysis_comparison()
        if not error_df.empty:
            report.append(
                tabulate(error_df, headers='keys', tablefmt='github', showindex=False)
            )
        report.append("\n\n")
        
        # Best model recommendations
        report.append("## Recommendations\n")
        best_f1 = stats_results['rankings'].get('f1_score', [{}])[0]
        best_acc = stats_results['rankings'].get('accuracy', [{}])[0]
        
        report.append(f"- **Best F1 Score**: {best_f1.get('model', 'N/A')}\n")
        report.append(f"- **Best Accuracy**: {best_acc.get('model', 'N/A')}\n")
        
        # Save report
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(''.join(report))
        
        print(f"Report saved to: {output_path}")
        return ''.join(report)
    
    def export_to_csv(self, output_dir: str = 'results/csv/'):
        """
        Export all results to CSV format.
        
        Args:
            output_dir: Directory to save CSV files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Overall metrics
        df = self.generate_comparison_table()
        df.to_csv(output_path / 'overall_comparison.csv', index=False)
        
        # Detailed predictions for each model
        for model_name, result in self.results.items():
            if 'predictions' in result:
                predictions_df = pd.DataFrame(result['predictions'])
                safe_name = model_name.replace('/', '_').replace(' ', '_')
                predictions_df.to_csv(
                    output_path / f'{safe_name}_predictions.csv', 
                    index=False
                )
        
        print(f"CSV files exported to: {output_path}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Compare benchmark results across models'
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        default='results/',
        help='Directory containing result JSON files'
    )
    parser.add_argument(
        '--output_report',
        type=str,
        default='results/comparison_report.md',
        help='Output path for comparison report'
    )
    parser.add_argument(
        '--export_csv',
        action='store_true',
        help='Export results to CSV format'
    )
    parser.add_argument(
        '--latex',
        action='store_true',
        help='Generate LaTeX tables'
    )
    
    args = parser.parse_args()
    
    # Initialize comparator
    comparator = ResultsComparator(args.results_dir)
    
    if not comparator.results:
        print("No results found! Please run benchmarks first.")
        return
    
    print("\n" + "="*60)
    print("BENCHMARK RESULTS COMPARISON")
    print("="*60 + "\n")
    
    # Generate main comparison table
    print("Generating comparison table...")
    df = comparator.generate_comparison_table()
    print("\n" + tabulate(df, headers='keys', tablefmt='github', showindex=False))
    print("\n")
    
    # Generate full report
    print("Generating comprehensive report...")
    comparator.generate_report(args.output_report)
    
    # Export to CSV if requested
    if args.export_csv:
        print("\nExporting to CSV...")
        comparator.export_to_csv()
    
    # Generate LaTeX if requested
    if args.latex:
        print("\nGenerating LaTeX tables...")
        latex_table = comparator.generate_latex_table(df)
        latex_path = Path(args.output_report).parent / 'comparison_table.tex'
        with open(latex_path, 'w') as f:
            f.write(latex_table)
        print(f"LaTeX table saved to: {latex_path}")
    
    print("\n" + "="*60)
    print("COMPARISON COMPLETE")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
