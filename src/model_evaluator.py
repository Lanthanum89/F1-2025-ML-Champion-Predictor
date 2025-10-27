"""
Model Evaluation and Visualization Module for F1 Championship Prediction
Provides comprehensive evaluation metrics, visualizations, and analysis tools
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

# ML evaluation metrics
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score
)
from sklearn.calibration import calibration_curve

# Statistical tests
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class F1ModelEvaluator:
    """Comprehensive evaluation and visualization for F1 championship prediction models"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        self.evaluation_results = {}
        
        # Set plotting style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def evaluate_model_performance(self, y_true: np.array, y_pred: np.array, 
                                 y_proba: np.array = None, model_name: str = "Model") -> Dict:
        """Comprehensive model performance evaluation"""
        
        logger.info(f"Evaluating performance for {model_name}")
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'support_positive': np.sum(y_true == 1),
            'support_negative': np.sum(y_true == 0),
            'total_predictions': len(y_true)
        }
        
        # Add probability-based metrics if available
        if y_proba is not None:
            try:
                auc_score = roc_auc_score(y_true, y_proba)
                results['auc_roc'] = auc_score
            except ValueError as e:
                logger.warning(f"Could not calculate AUC-ROC: {e}")
                results['auc_roc'] = None
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        results['confusion_matrix'] = cm
        
        # Store results
        self.evaluation_results[model_name] = results
        
        return results
    
    def compare_models(self, results_dict: Dict[str, Dict]) -> pd.DataFrame:
        """Compare multiple models performance"""
        
        comparison_data = []
        
        for model_name, results in results_dict.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': results.get('accuracy', 0),
                'Precision': results.get('precision', 0),
                'Recall': results.get('recall', 0),
                'F1-Score': results.get('f1_score', 0),
                'AUC-ROC': results.get('auc_roc', 0)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('F1-Score', ascending=False)
        
        return comparison_df
    
    def plot_confusion_matrix(self, y_true: np.array, y_pred: np.array, 
                            model_name: str = "Model", normalize: bool = False):
        """Plot confusion matrix"""
        
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2%'
            title = f'Normalized Confusion Matrix - {model_name}'
        else:
            fmt = 'd'
            title = f'Confusion Matrix - {model_name}'
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                   xticklabels=['Not Champion', 'Champion'],
                   yticklabels=['Not Champion', 'Champion'])
        
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_roc_curves(self, models_data: Dict[str, Tuple[np.array, np.array]]):
        """Plot ROC curves for multiple models"""
        
        plt.figure(figsize=self.figsize)
        
        for model_name, (y_true, y_proba) in models_data.items():
            try:
                fpr, tpr, _ = roc_curve(y_true, y_proba)
                auc_score = auc(fpr, tpr)
                
                plt.plot(fpr, tpr, linewidth=2, 
                        label=f'{model_name} (AUC = {auc_score:.3f})')
            except Exception as e:
                logger.warning(f"Could not plot ROC curve for {model_name}: {e}")
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_precision_recall_curves(self, models_data: Dict[str, Tuple[np.array, np.array]]):
        """Plot Precision-Recall curves for multiple models"""
        
        plt.figure(figsize=self.figsize)
        
        for model_name, (y_true, y_proba) in models_data.items():
            try:
                precision, recall, _ = precision_recall_curve(y_true, y_proba)
                avg_precision = np.trapz(precision, recall)
                
                plt.plot(recall, precision, linewidth=2,
                        label=f'{model_name} (AP = {avg_precision:.3f})')
            except Exception as e:
                logger.warning(f"Could not plot PR curve for {model_name}: {e}")
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves Comparison')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_model_comparison(self, comparison_df: pd.DataFrame):
        """Plot model performance comparison"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        for i, metric in enumerate(metrics):
            ax = axes[i//2, i%2]
            
            bars = ax.bar(comparison_df['Model'], comparison_df[metric])
            ax.set_title(f'{metric} Comparison')
            ax.set_ylabel(metric)
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom')
            
            ax.set_ylim(0, 1.1)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_feature_importance_comparison(self, feature_importance_dict: Dict[str, pd.DataFrame], 
                                         top_n: int = 10):
        """Compare feature importance across different models"""
        
        fig, axes = plt.subplots(1, len(feature_importance_dict), 
                               figsize=(6*len(feature_importance_dict), 8))
        
        if len(feature_importance_dict) == 1:
            axes = [axes]
        
        for i, (model_name, importance_df) in enumerate(feature_importance_dict.items()):
            ax = axes[i] if len(feature_importance_dict) > 1 else axes[0]
            
            top_features = importance_df.head(top_n)
            
            bars = ax.barh(range(len(top_features)), top_features['importance'])
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features['feature'])
            ax.set_xlabel('Importance')
            ax.set_title(f'{model_name} - Top {top_n} Features')
            ax.invert_yaxis()
            
            # Add value labels
            for j, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width + 0.001, bar.get_y() + bar.get_height()/2.,
                       f'{width:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        return fig
    
    def analyze_predictions_by_driver(self, predictions_df: pd.DataFrame, 
                                    actual_champions: List[str] = None):
        """Analyze prediction accuracy by driver characteristics"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Championship probability distribution
        ax1 = axes[0, 0]
        ax1.hist(predictions_df['championship_probability'], bins=20, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Championship Probability')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Championship Probabilities')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Top predicted vs actual (if actual champions provided)
        ax2 = axes[0, 1]
        top_predictions = predictions_df.nlargest(10, 'championship_probability')
        bars = ax2.bar(range(len(top_predictions)), top_predictions['championship_probability'])
        ax2.set_xticks(range(len(top_predictions)))
        ax2.set_xticklabels(top_predictions['driver_name'], rotation=45, ha='right')
        ax2.set_ylabel('Championship Probability')
        ax2.set_title('Top 10 Predicted Championship Contenders')
        
        # Highlight actual champions if provided
        if actual_champions:
            for i, driver in enumerate(top_predictions['driver_name']):
                if driver in actual_champions:
                    bars[i].set_color('gold')
        
        # Plot 3: Constructor analysis
        ax3 = axes[1, 0]
        constructor_avg = predictions_df.groupby('constructor_name')['championship_probability'].mean().sort_values(ascending=False)
        constructor_avg.head(10).plot(kind='bar', ax=ax3)
        ax3.set_title('Average Championship Probability by Constructor')
        ax3.set_ylabel('Average Probability')
        ax3.tick_params(axis='x', rotation=45)
        
        # Plot 4: Confidence analysis
        ax4 = axes[1, 1]
        predictions_df['confidence_level'] = pd.cut(
            predictions_df['championship_probability'],
            bins=[0, 0.1, 0.3, 0.6, 1.0],
            labels=['Low', 'Medium', 'High', 'Very High']
        )
        confidence_counts = predictions_df['confidence_level'].value_counts()
        ax4.pie(confidence_counts.values, labels=confidence_counts.index, autopct='%1.1f%%')
        ax4.set_title('Prediction Confidence Distribution')
        
        plt.tight_layout()
        return fig
    
    def calculate_prediction_calibration(self, y_true: np.array, y_proba: np.array, 
                                       n_bins: int = 10) -> Tuple[np.array, np.array]:
        """Calculate calibration curve for probability predictions"""
        
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_proba, n_bins=n_bins
        )
        
        return fraction_of_positives, mean_predicted_value
    
    def plot_calibration_curve(self, y_true: np.array, y_proba: np.array, 
                              model_name: str = "Model", n_bins: int = 10):
        """Plot calibration curve to assess probability calibration"""
        
        fraction_of_positives, mean_predicted_value = self.calculate_prediction_calibration(
            y_true, y_proba, n_bins
        )
        
        plt.figure(figsize=(8, 6))
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", 
                linewidth=2, label=f'{model_name}')
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title(f'Calibration Curve - {model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return plt.gcf()
    
    def analyze_historical_accuracy(self, historical_predictions: Dict[int, pd.DataFrame], 
                                  historical_champions: Dict[int, str]):
        """Analyze historical prediction accuracy across seasons"""
        
        accuracy_data = []
        
        for year, predictions_df in historical_predictions.items():
            actual_champion = historical_champions.get(year)
            
            if actual_champion:
                # Find the predicted rank of the actual champion
                champion_prediction = predictions_df[
                    predictions_df['driver_name'] == actual_champion
                ]
                
                if not champion_prediction.empty:
                    champion_rank = (predictions_df['championship_probability'] > 
                                   champion_prediction['championship_probability'].iloc[0]).sum() + 1
                    champion_prob = champion_prediction['championship_probability'].iloc[0]
                    
                    # Check if model predicted this driver as champion
                    predicted_correctly = predictions_df.iloc[0]['driver_name'] == actual_champion
                    
                    accuracy_data.append({
                        'year': year,
                        'actual_champion': actual_champion,
                        'predicted_champion': predictions_df.iloc[0]['driver_name'],
                        'champion_predicted_rank': champion_rank,
                        'champion_probability': champion_prob,
                        'correctly_predicted': predicted_correctly,
                        'top_3_prediction': champion_rank <= 3,
                        'top_5_prediction': champion_rank <= 5
                    })
        
        accuracy_df = pd.DataFrame(accuracy_data)
        
        if not accuracy_df.empty:
            # Calculate summary statistics
            summary_stats = {
                'exact_accuracy': accuracy_df['correctly_predicted'].mean(),
                'top_3_accuracy': accuracy_df['top_3_prediction'].mean(),
                'top_5_accuracy': accuracy_df['top_5_prediction'].mean(),
                'average_champion_rank': accuracy_df['champion_predicted_rank'].mean(),
                'average_champion_probability': accuracy_df['champion_probability'].mean()
            }
            
            return accuracy_df, summary_stats
        
        return pd.DataFrame(), {}
    
    def plot_historical_accuracy(self, accuracy_df: pd.DataFrame):
        """Plot historical prediction accuracy analysis"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Accuracy by year
        ax1 = axes[0, 0]
        years = accuracy_df['year']
        correct_predictions = accuracy_df['correctly_predicted']
        
        colors = ['green' if correct else 'red' for correct in correct_predictions]
        bars = ax1.bar(years, [1] * len(years), color=colors, alpha=0.7)
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Prediction Result')
        ax1.set_title('Prediction Accuracy by Year')
        ax1.set_yticks([0, 1])
        ax1.set_yticklabels(['Incorrect', 'Correct'])
        
        # Plot 2: Champion probability distribution
        ax2 = axes[0, 1]
        ax2.hist(accuracy_df['champion_probability'], bins=10, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Assigned Probability to Actual Champion')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Model Confidence in Actual Champions')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Predicted rank of actual champions
        ax3 = axes[1, 0]
        rank_counts = accuracy_df['champion_predicted_rank'].value_counts().sort_index()
        ax3.bar(rank_counts.index, rank_counts.values, alpha=0.7)
        ax3.set_xlabel('Predicted Rank of Actual Champion')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of Actual Champion Rankings')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Cumulative accuracy levels
        ax4 = axes[1, 1]
        accuracy_levels = ['Exact', 'Top 3', 'Top 5']
        accuracy_values = [
            accuracy_df['correctly_predicted'].mean(),
            accuracy_df['top_3_prediction'].mean(),
            accuracy_df['top_5_prediction'].mean()
        ]
        
        bars = ax4.bar(accuracy_levels, accuracy_values, alpha=0.7)
        ax4.set_ylabel('Accuracy Rate')
        ax4.set_title('Prediction Accuracy at Different Levels')
        ax4.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, accuracy_values):
            ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                    f'{value:.2%}', ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
    
    def generate_evaluation_report(self, results_dict: Dict[str, Dict], 
                                 save_path: Optional[str] = None) -> str:
        """Generate comprehensive evaluation report"""
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("F1 CHAMPIONSHIP PREDICTION MODEL EVALUATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Model comparison
        comparison_df = self.compare_models(results_dict)
        report_lines.append("MODEL PERFORMANCE COMPARISON")
        report_lines.append("-" * 40)
        report_lines.append(comparison_df.to_string(index=False))
        report_lines.append("")
        
        # Best model analysis
        best_model = comparison_df.iloc[0]['Model']
        best_results = results_dict[best_model]
        
        report_lines.append(f"BEST PERFORMING MODEL: {best_model}")
        report_lines.append("-" * 40)
        report_lines.append(f"Accuracy:  {best_results['accuracy']:.4f}")
        report_lines.append(f"Precision: {best_results['precision']:.4f}")
        report_lines.append(f"Recall:    {best_results['recall']:.4f}")
        report_lines.append(f"F1-Score:  {best_results['f1_score']:.4f}")
        if best_results.get('auc_roc'):
            report_lines.append(f"AUC-ROC:   {best_results['auc_roc']:.4f}")
        report_lines.append("")
        
        # Confusion matrix
        if 'confusion_matrix' in best_results:
            cm = best_results['confusion_matrix']
            report_lines.append("CONFUSION MATRIX (Best Model)")
            report_lines.append("-" * 30)
            report_lines.append(f"True Negatives:  {cm[0][0]}")
            report_lines.append(f"False Positives: {cm[0][1]}")
            report_lines.append(f"False Negatives: {cm[1][0]}")
            report_lines.append(f"True Positives:  {cm[1][1]}")
            report_lines.append("")
        
        # Recommendations
        report_lines.append("RECOMMENDATIONS")
        report_lines.append("-" * 20)
        
        if best_results['precision'] < 0.5:
            report_lines.append("• Low precision suggests many false positive predictions")
            report_lines.append("  Consider adjusting prediction threshold or improving features")
        
        if best_results['recall'] < 0.5:
            report_lines.append("• Low recall suggests missing true champions")
            report_lines.append("  Consider collecting more historical data or feature engineering")
        
        if best_results.get('auc_roc', 1) < 0.7:
            report_lines.append("• Low AUC-ROC suggests poor discrimination ability")
            report_lines.append("  Consider trying different algorithms or feature selection")
        
        report_lines.append("• Regular model retraining recommended as new season data becomes available")
        report_lines.append("")
        
        report_text = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Evaluation report saved to {save_path}")
        
        return report_text

if __name__ == "__main__":
    # Example usage
    print("F1 Model Evaluator - Example Usage")
    print("This module provides comprehensive evaluation tools for F1 championship prediction models.")
    print("Use in conjunction with ml_model.py to evaluate trained models.")