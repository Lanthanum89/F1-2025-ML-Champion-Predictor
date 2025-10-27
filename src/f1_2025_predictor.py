"""
F1 2025 Championship Prediction Interface
Main interface for predicting the 2025 F1 World Championship winner
"""

import pandas as pd
import numpy as np
import sys
import os
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, date
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import custom modules
from data_collector import F1DataCollector
from feature_engineering_simple import F1FeatureEngineer
from ml_model import F1ChampionshipPredictor
from model_evaluator import F1ModelEvaluator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class F12025Predictor:
    """Main interface for F1 2025 championship prediction"""
    
    def __init__(self):
        self.data_collector = F1DataCollector()
        self.predictor = F1ChampionshipPredictor()
        self.evaluator = F1ModelEvaluator()
        
        # 2025 F1 season data (estimated/current as of 2024)
        self.f1_2025_drivers = self.get_2025_season_data()
        
    def get_2025_season_data(self) -> pd.DataFrame:
        """Get estimated 2025 F1 season driver and team data"""
        
        # Load 2025 grid data from CSV file
        grid_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'f1_2025_grid.csv')
        df_2025 = pd.read_csv(grid_file)
        
        logger.info(f"Loaded 2025 F1 grid with {len(df_2025)} drivers")
        
        return df_2025
    
    def collect_and_prepare_training_data(self) -> pd.DataFrame:
        """Collect historical F1 data and prepare for training"""
        
        logger.info("Collecting historical F1 data for training...")
        
        # Try to load existing data first
        data_file = '../data/f1_championship_data.csv'
        if os.path.exists(data_file):
            logger.info("Loading existing F1 data...")
            df = pd.read_csv(data_file)
        else:
            logger.info("Collecting new F1 data from API...")
            df = self.data_collector.get_seasons_data(2010, 2024)
            
            if not df.empty:
                # Save the data
                os.makedirs('../data', exist_ok=True)
                self.data_collector.save_data(df, "f1_championship_data.csv")
            else:
                raise ValueError("Failed to collect F1 data")
        
        logger.info(f"Loaded data for {df['year'].nunique()} seasons with {len(df)} driver records")
        
        return df
    
    def train_championship_model(self, df: pd.DataFrame) -> Dict:
        """Train the F1 championship prediction model"""
        
        logger.info("Training F1 championship prediction model...")
        
        # Train the model
        results = self.predictor.train(df, optimize_hyperparameters=True)
        
        # Save the trained model
        os.makedirs('../models', exist_ok=True)
        model_path = '../models/f1_championship_predictor_2025.joblib'
        self.predictor.save_model(model_path)
        
        logger.info(f"Model trained and saved to {model_path}")
        
        return results
    
    def predict_2025_champion(self, training_data: pd.DataFrame) -> pd.DataFrame:
        """Predict the 2025 F1 World Championship winner"""
        
        logger.info("Making 2025 F1 championship predictions...")
        
        if not self.predictor.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Create a combined dataset with historical data + 2025 data for feature engineering
        combined_data = pd.concat([training_data, self.f1_2025_drivers], ignore_index=True)
        
        # Make predictions using the combined dataset
        predictions_df = self.predictor.predict_champion_probabilities_with_history(combined_data)
        
        # Add additional analysis
        predictions_df['odds'] = 1 / predictions_df['championship_probability']
        predictions_df['confidence_level'] = pd.cut(
            predictions_df['championship_probability'],
            bins=[0, 0.05, 0.15, 0.35, 1.0],
            labels=['Low', 'Medium', 'High', 'Very High']
        )
        
        return predictions_df
    
    def generate_prediction_report(self, predictions_df: pd.DataFrame, 
                                 training_results: Dict) -> str:
        """Generate comprehensive 2025 championship prediction report"""
        
        report_lines = []
        report_lines.append("=" * 100)
        report_lines.append("F1 2025 WORLD CHAMPIONSHIP PREDICTION REPORT")
        report_lines.append("=" * 100)
        report_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Prediction Model: F1ChampionshipPredictor (Ensemble)")
        report_lines.append("")
        
        # Top predictions
        report_lines.append("TOP 10 CHAMPIONSHIP CONTENDERS")
        report_lines.append("-" * 50)
        top_10 = predictions_df.head(10)
        
        for i, (_, row) in enumerate(top_10.iterrows(), 1):
            prob_pct = row['championship_probability'] * 100
            report_lines.append(
                f"{i:2d}. {row['driver_name']:<20} ({row['constructor_name']:<12}) - "
                f"{prob_pct:6.2f}% (Odds: {row['odds']:5.1f}/1)"
            )
        
        report_lines.append("")
        
        # Championship favorite analysis
        favorite = predictions_df.iloc[0]
        report_lines.append("CHAMPIONSHIP FAVORITE ANALYSIS")
        report_lines.append("-" * 40)
        report_lines.append(f"Predicted Champion: {favorite['driver_name']}")
        report_lines.append(f"Team: {favorite['constructor_name']}")
        report_lines.append(f"Championship Probability: {favorite['championship_probability']:.1%}")
        report_lines.append(f"Betting Odds Equivalent: {favorite['odds']:.1f}/1")
        report_lines.append(f"Confidence Level: {favorite['confidence_level']}")
        report_lines.append("")
        
        # Constructor analysis
        report_lines.append("CONSTRUCTOR CHAMPIONSHIP OUTLOOK")
        report_lines.append("-" * 45)
        constructor_probs = predictions_df.groupby('constructor_name')['championship_probability'].sum().sort_values(ascending=False)
        
        for constructor, total_prob in constructor_probs.head(8).items():
            report_lines.append(f"{constructor:<15}: {total_prob:.1%}")
        
        report_lines.append("")
        
        # Model performance summary
        report_lines.append("MODEL PERFORMANCE SUMMARY")
        report_lines.append("-" * 35)
        
        if 'test_results' in training_results and 'ensemble' in training_results['test_results']:
            ensemble_metrics = training_results['test_results']['ensemble']
            report_lines.append(f"Model Accuracy: {ensemble_metrics.get('accuracy', 0):.1%}")
            report_lines.append(f"Precision: {ensemble_metrics.get('precision', 0):.1%}")
            report_lines.append(f"Recall: {ensemble_metrics.get('recall', 0):.1%}")
            report_lines.append(f"F1-Score: {ensemble_metrics.get('f1', 0):.1%}")
            if ensemble_metrics.get('auc'):
                report_lines.append(f"AUC-ROC: {ensemble_metrics.get('auc'):.3f}")
        
        report_lines.append(f"Training Data: {training_results.get('training_data_shape', 'N/A')}")
        report_lines.append("")
        
        # Key insights
        report_lines.append("KEY INSIGHTS & FACTORS")
        report_lines.append("-" * 30)
        
        # Analyze prediction spread
        top_3_prob = predictions_df.head(3)['championship_probability'].sum()
        
        if favorite['championship_probability'] > 0.4:
            report_lines.append("• Strong favorite detected - championship race may be less competitive")
        elif top_3_prob < 0.6:
            report_lines.append("• Very competitive season expected - multiple strong contenders")
        else:
            report_lines.append("• Moderately competitive season with a few key contenders")
        
        # Constructor dominance
        top_constructor_prob = constructor_probs.iloc[0]
        if top_constructor_prob > 0.6:
            report_lines.append("• One team shows clear dominance in championship odds")
        elif constructor_probs.head(3).sum() > 0.8:
            report_lines.append("• Championship likely to be contested between top 3 teams")
        
        # Uncertainty analysis
        prediction_entropy = -np.sum(predictions_df['championship_probability'] * 
                                   np.log(predictions_df['championship_probability'] + 1e-10))
        
        if prediction_entropy > 2.5:
            report_lines.append("• High uncertainty detected - season outcome difficult to predict")
        elif prediction_entropy < 1.5:
            report_lines.append("• Low uncertainty - model shows strong confidence in predictions")
        
        report_lines.append("")
        
        # Disclaimers
        report_lines.append("IMPORTANT DISCLAIMERS")
        report_lines.append("-" * 25)
        report_lines.append("• Predictions based on historical data and estimated 2025 performance")
        report_lines.append("• Actual results may vary due to:")
        report_lines.append("  - Car development and regulation changes")
        report_lines.append("  - Driver transfers and team dynamics")
        report_lines.append("  - Injuries, penalties, and unforeseen circumstances")
        report_lines.append("  - Weather conditions and race incidents")
        report_lines.append("• Model will be updated as 2025 season data becomes available")
        report_lines.append("")
        
        report_lines.append("For entertainment purposes only - not investment advice!")
        report_lines.append("=" * 100)
        
        return "\n".join(report_lines)
    
    def run_full_prediction_pipeline(self, retrain_model: bool = True) -> Tuple[pd.DataFrame, str]:
        """Run the complete 2025 championship prediction pipeline"""
        
        logger.info("Starting F1 2025 championship prediction pipeline...")
        
        try:
            # Step 1: Collect training data
            training_df = self.collect_and_prepare_training_data()
            
            # Step 2: Train model (or load existing)
            model_path = '../models/f1_championship_predictor_2025.joblib'
            
            if retrain_model or not os.path.exists(model_path):
                training_results = self.train_championship_model(training_df)
            else:
                logger.info("Loading existing trained model...")
                self.predictor.load_model(model_path)
                training_results = {"message": "Loaded pre-trained model"}
            
            # Step 3: Make 2025 predictions
            predictions_df = self.predict_2025_champion(training_df)
            
            # Step 4: Generate report
            report = self.generate_prediction_report(predictions_df, training_results)
            
            # Step 5: Save results
            os.makedirs('../results', exist_ok=True)
            
            predictions_path = '../results/f1_2025_championship_predictions.csv'
            predictions_df.to_csv(predictions_path, index=False)
            
            report_path = '../results/f1_2025_prediction_report.txt'
            with open(report_path, 'w') as f:
                f.write(report)
            
            logger.info(f"Predictions saved to {predictions_path}")
            logger.info(f"Report saved to {report_path}")
            
            return predictions_df, report
            
        except Exception as e:
            logger.error(f"Error in prediction pipeline: {e}")
            raise

def main():
    """Main function to run F1 2025 championship prediction"""
    
    print("🏎️  F1 2025 World Championship Prediction System")
    print("=" * 60)
    
    try:
        # Initialize predictor
        f1_predictor = F12025Predictor()
        
        # Run prediction pipeline
        predictions_df, report = f1_predictor.run_full_prediction_pipeline(retrain_model=True)
        
        # Display results
        print("\n" + report)
        
        print("\n🏆 Prediction Summary:")
        print("-" * 30)
        top_5 = predictions_df.head(5)
        for i, (_, row) in enumerate(top_5.iterrows(), 1):
            print(f"{i}. {row['driver_name']} ({row['constructor_name']}) - {row['championship_probability']:.1%}")
        
        print("\n✅ Prediction complete! Check the 'results' folder for detailed outputs.")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Please ensure you have internet connection for data collection.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())