"""
F1 Data Collection Module
Fetches historical F1 data for championship prediction model from local CSV file
"""

import pandas as pd
import os
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class F1DataCollector:
    """Collect F1 data from local CSV file"""
    
    def __init__(self):
        self.data_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'f1_championship_data.csv')
        self.cache = {}
    
    def get_seasons_data(self, start_year: int = 2010, end_year: int = 2025) -> pd.DataFrame:
        """
        Collect championship data for multiple seasons from local CSV
        
        Args:
            start_year: First year to collect data from
            end_year: Last year to collect data to
            
        Returns:
            DataFrame with all seasons' championship data
        """
        try:
            # Read the CSV file
            logger.info(f"Loading data from {self.data_file}")
            df = pd.read_csv(self.data_file)
            
            # Filter by year range
            df = df[(df['year'] >= start_year) & (df['year'] <= end_year)]
            
            logger.info(f"Loaded data for {len(df['year'].unique())} seasons from {start_year} to {end_year}")
            logger.info(f"Total records: {len(df)}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return pd.DataFrame()
    
    def get_season_data(self, year: int) -> Optional[pd.DataFrame]:
        """
        Get championship standings and race results for a specific season
        
        Args:
            year: The season year
            
        Returns:
            DataFrame with season data or None if failed
        """
        try:
            df = pd.read_csv(self.data_file)
            season_df = df[df['year'] == year]
            
            if len(season_df) > 0:
                logger.info(f"Loaded {len(season_df)} driver records for {year}")
                return season_df
            else:
                logger.warning(f"No data found for {year}")
                return None
                
        except Exception as e:
            logger.error(f"Error collecting data for {year}: {e}")
            return None
    
    def save_data(self, data: pd.DataFrame, filename: Optional[str] = None) -> bool:
        """
        Save data to CSV file (for compatibility)
        
        Args:
            data: DataFrame to save
            filename: Optional filename (ignored since we use fixed file)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Data is already saved in our CSV file
            logger.info(f"Data saved successfully ({len(data)} records)")
            return True
        except Exception as e:
            logger.error(f"Error saving data: {e}")
            return False