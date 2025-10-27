"""
Feature Engineering Module for F1 Championship Prediction
Creates meaningful features from raw F1 data for machine learning
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class F1FeatureEngineer:
    """Engineer features for F1 championship prediction"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive feature set for ML model"""
        
        logger.info("Creating features for F1 championship prediction...")
        
        # Make a copy to avoid modifying original data
        features_df = df.copy()
        
        # 1. Performance Features
        features_df = self.create_performance_features(features_df)
        
        # 2. Historical Features
        features_df = self.create_historical_features(features_df)
        
        # 3. Relative Performance Features
        features_df = self.create_relative_features(features_df)
        
        # 4. Consistency Features
        features_df = self.create_consistency_features(features_df)
        
        # 5. Team/Constructor Features
        features_df = self.create_team_features(features_df)
        
        # 6. Target Variable
        features_df = self.create_target_variable(features_df)
        
        # 7. Categorical Encoding
        features_df = self.encode_categorical_features(features_df)
        
        logger.info(f"Created {len(features_df.columns)} total features")
        
        return features_df
    
    def create_performance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create performance-based features"""
        
        # Points per race
        df['points_per_race'] = df['final_points'] / df['races_started']
        
        # Win rate
        df['win_rate'] = df['wins'] / df['races_started']
        
        # Podium efficiency (podiums per points finish)
        df['podium_efficiency'] = np.where(df['points_finishes'] > 0, 
                                          df['podiums'] / df['points_finishes'], 0)
        
        # Grid to finish improvement
        df['grid_finish_improvement'] = df['avg_grid_position'] - df['avg_finish_position']
        
        # Race craft score (combination of consistency and speed)
        df['race_craft_score'] = (df['completion_rate'] * 0.3 + 
                                 df['points_rate'] * 0.4 + 
                                 df['podium_rate'] * 0.3)
        
        # Speed score (based on qualifying and fastest laps)
        df['speed_score'] = (1 / (df['avg_grid_position'] + 1)) * 0.7 + df['fastest_laps'] / df['races_started'] * 0.3
        
        return df
    
    def create_historical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features based on historical performance"""
        
        # Sort by year to enable time-based features
        df_sorted = df.sort_values(['driver_id', 'year'])
        
        # Previous year performance
        df_sorted['prev_year_points'] = df_sorted.groupby('driver_id')['final_points'].shift(1)
        df_sorted['prev_year_position'] = df_sorted.groupby('driver_id')['final_position'].shift(1)
        df_sorted['prev_year_wins'] = df_sorted.groupby('driver_id')['wins'].shift(1)
        
        # Career progression
        df_sorted['years_in_f1'] = df_sorted.groupby('driver_id').cumcount() + 1
        
        # Rolling averages (3-year window)
        df_sorted['rolling_avg_points'] = (df_sorted.groupby('driver_id')['final_points']
                                          .rolling(window=3, min_periods=1).mean().values)
        df_sorted['rolling_avg_position'] = (df_sorted.groupby('driver_id')['final_position']
                                           .rolling(window=3, min_periods=1).mean().values)
        
        # Improvement trend
        df_sorted['points_trend'] = (df_sorted.groupby('driver_id')['final_points']
                                   .pct_change().fillna(0))
        
        # Peak performance (best position ever achieved)
        df_sorted['career_best_position'] = (df_sorted.groupby('driver_id')['final_position']
                                           .expanding().min())
        df_sorted['career_total_wins'] = (df_sorted.groupby('driver_id')['wins']
                                        .expanding().sum())
        
        return df_sorted
    
    def create_relative_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features relative to season competitors"""
        
        # Rank features within each season
        df['points_rank'] = df.groupby('year')['final_points'].rank(ascending=False)
        df['wins_rank'] = df.groupby('year')['wins'].rank(ascending=False)
        df['podium_rate_rank'] = df.groupby('year')['podium_rate'].rank(ascending=False)
        
        # Percentile rankings
        df['points_percentile'] = df.groupby('year')['final_points'].rank(pct=True)
        df['completion_percentile'] = df.groupby('year')['completion_rate'].rank(pct=True)
        
        # Gap to leader
        season_leaders = df.groupby('year')['final_points'].max()
        df['points_gap_to_leader'] = df.apply(
            lambda row: season_leaders[row['year']] - row['final_points'], axis=1
        )
        
        # Points concentration (what % of season points did driver earn)
        season_totals = df.groupby('year')['final_points'].sum()
        df['points_share'] = df.apply(
            lambda row: row['final_points'] / season_totals[row['year']] if season_totals[row['year']] > 0 else 0, 
            axis=1
        )
        
        return df
    
    def create_consistency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features measuring driver consistency"""
        
        # Reliability score
        df['reliability_score'] = df['completion_rate'] * (1 - df['dnf_rate'])
        
        # Performance consistency (lower variance = more consistent)
        # We'll approximate this using available data
        df['performance_consistency'] = np.where(
            df['avg_finish_position'] > 0,
            1 / (df['avg_finish_position'] * (df['dnf_rate'] + 0.1)),  # Lower is better for position
            0
        )
        
        # Points consistency (ability to regularly score points)
        df['points_consistency'] = df['points_rate'] * df['completion_rate']
        
        # Championship contention factor (how often in top positions)
        df['championship_factor'] = (df['podium_rate'] * 0.4 + 
                                   df['points_rate'] * 0.3 + 
                                   df['win_rate'] * 0.3)
        
        return df
    
    def create_team_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create team/constructor-based features"""
        
        # Team performance metrics
        team_stats = df.groupby(['year', 'constructor_id']).agg({
            'final_points': 'sum',
            'wins': 'sum',
            'podiums': 'sum',
            'constructor_final_points': 'first',
            'constructor_final_position': 'first'
        }).reset_index()
        
        team_stats.columns = [
            'year', 'constructor_id', 'team_total_points', 'team_total_wins',
            'team_total_podiums', 'constructor_final_points', 'constructor_final_position'
        ]
        
        # Merge team stats back
        df = df.merge(team_stats, on=['year', 'constructor_id'], how='left')
        
        # Driver's contribution to team
        df['driver_team_points_share'] = np.where(
            df['team_total_points'] > 0,
            df['final_points'] / df['team_total_points'],
            0
        )
        
        # Team competitiveness
        df['team_competitiveness'] = 1 / (df['constructor_final_position'] + 1)
        
        # Historical team performance
        df_sorted = df.sort_values(['constructor_id', 'year'])
        df['team_prev_year_position'] = df_sorted.groupby('constructor_id')['constructor_final_position'].shift(1)
        df['team_trend'] = np.where(
            df['team_prev_year_position'].notna(),
            df['team_prev_year_position'] - df['constructor_final_position'],  # Positive = improvement
            0
        )
        
        return df
    
    def create_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target variables for different prediction tasks"""
        
        # Main target: Championship winner (binary)
        df['is_champion'] = (df['final_position'] == 1).astype(int)
        
        # Additional targets for multi-class prediction
        df['championship_category'] = pd.cut(
            df['final_position'], 
            bins=[0, 1, 3, 10, float('inf')], 
            labels=['Champion', 'Podium', 'Points', 'No_Points']
        )
        
        # Points-based target (normalized)
        df['points_target'] = df.groupby('year')['final_points'].transform(
            lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0
        )
        
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables for ML algorithms"""
        
        categorical_columns = ['nationality', 'constructor_name', 'championship_category']
        
        for col in categorical_columns:
            if col in df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                
                # Handle missing values
                df[col] = df[col].fillna('unknown')
                df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col])
        
        return df
    
    def select_features(self, df: pd.DataFrame) -> List[str]:
        """Select the most relevant features for the model"""
        
        # Define feature categories
        performance_features = [
            'points_per_race', 'win_rate', 'podium_efficiency', 
            'grid_finish_improvement', 'race_craft_score', 'speed_score'
        ]
        
        historical_features = [
            'prev_year_points', 'prev_year_position', 'years_in_f1',
            'rolling_avg_points', 'rolling_avg_position', 'points_trend',
            'career_best_position', 'career_total_wins'
        ]
        
        relative_features = [
            'points_rank', 'wins_rank', 'points_percentile', 
            'points_gap_to_leader', 'points_share'
        ]
        
        consistency_features = [
            'reliability_score', 'performance_consistency', 
            'points_consistency', 'championship_factor'
        ]
        
        team_features = [
            'driver_team_points_share', 'team_competitiveness', 
            'team_trend', 'constructor_final_position'
        ]
        
        basic_features = [
            'completion_rate', 'podium_rate', 'points_rate', 
            'avg_finish_position', 'avg_grid_position'
        ]
        
        categorical_features = [
            'nationality_encoded', 'constructor_name_encoded'
        ]
        
        # Combine all features
        all_features = (performance_features + historical_features + 
                       relative_features + consistency_features + 
                       team_features + basic_features + categorical_features)
        
        # Filter features that actually exist in the dataframe
        available_features = [f for f in all_features if f in df.columns]
        
        self.feature_columns = available_features
        logger.info(f"Selected {len(available_features)} features for modeling")
        
        return available_features
    
    def prepare_training_data(self, df: pd.DataFrame, target_col: str = 'is_champion', 
                            test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Prepare data for training ML models"""
        
        # Remove rows with missing target
        clean_df = df.dropna(subset=[target_col])
        
        # Select features
        feature_cols = self.select_features(clean_df)
        
        # Handle missing values in features
        X = clean_df[feature_cols].fillna(clean_df[feature_cols].median())
        y = clean_df[target_col]
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale the features
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        logger.info(f"Training data shape: {X_train_scaled.shape}")
        logger.info(f"Test data shape: {X_test_scaled.shape}")
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def prepare_prediction_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare new data for making predictions"""
        
        if not self.feature_columns:
            raise ValueError("Model not trained yet. Call prepare_training_data first.")
        
        # Handle missing values
        X = df[self.feature_columns].fillna(df[self.feature_columns].median())
        
        # Scale features using fitted scaler
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=X.columns,
            index=X.index
        )
        
        return X_scaled
    
    def get_feature_importance_names(self) -> List[str]:
        """Get list of feature names for importance analysis"""
        return self.feature_columns

if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append('../src')
    
    # Load sample data (you would load your actual collected data here)
    try:
        df = pd.read_csv('../data/f1_championship_data.csv')
        
        # Initialize feature engineer
        engineer = F1FeatureEngineer()
        
        # Create features
        features_df = engineer.create_features(df)
        
        # Prepare training data
        X_train, X_test, y_train, y_test = engineer.prepare_training_data(features_df)
        
        print("Feature engineering completed successfully!")
        print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        print(f"\nFeatures created: {engineer.get_feature_importance_names()[:10]}...")  # Show first 10
        
    except FileNotFoundError:
        print("No data file found. Run data_collector.py first to collect F1 data.")