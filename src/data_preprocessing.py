
"""
XFoil Data Preprocessing for Airfoil ML Optimization
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
import glob

class XFoilDataProcessor:
    """
    Preprocessor for XFoil airfoil datasets
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def load_xfoil_data(self, data_path):
        """
        Load XFoil datasets from directory
        """
        print(f"📂 Loading XFoil data from: {data_path}")
        
        # Find all CSV files
        csv_files = glob.glob(os.path.join(data_path, "*.csv"))
        
        if not csv_files:
            print("⚠️  No CSV files found. Creating sample data...")
            return self.create_sample_data()
        
        dataframes = []
        for file in csv_files:
            df = pd.read_csv(file)
            dataframes.append(df)
            
        combined_df = pd.concat(dataframes, ignore_index=True)
        print(f"✅ Loaded {len(combined_df)} samples from {len(csv_files)} files")
        
        return self.preprocess_data(combined_df)
    
    def create_sample_data(self):
        """
        Create sample XFoil-like data for testing
        """
        print("🧪 Creating synthetic XFoil dataset...")
        
        np.random.seed(42)
        n_samples = 2000
        
        # Airfoil geometric parameters
        data = {
            'angle_of_attack': np.random.uniform(-10, 15, n_samples),
            'reynolds_number': np.random.uniform(50000, 500000, n_samples),
            'max_thickness': np.random.uniform(0.08, 0.18, n_samples),
            'max_camber': np.random.uniform(0.0, 0.08, n_samples),
            'camber_position': np.random.uniform(0.3, 0.6, n_samples),
            'leading_edge_radius': np.random.uniform(0.005, 0.025, n_samples),
            'trailing_edge_angle': np.random.uniform(8, 20, n_samples),
            'chord_length': np.random.uniform(0.8, 1.2, n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Calculate synthetic L/D ratio with realistic relationships
        df['lift_coefficient'] = (
            1.2 * df['angle_of_attack'] * np.pi/180 +
            2.5 * df['max_camber'] +
            np.random.normal(0, 0.1, n_samples)
        )
        
        df['drag_coefficient'] = (
            0.008 + 
            0.02 * df['max_thickness'] +
            0.001 * (df['angle_of_attack']**2) +
            np.random.normal(0, 0.002, n_samples)
        )
        
        # L/D ratio (target variable)
        df['lift_to_drag_ratio'] = df['lift_coefficient'] / np.maximum(df['drag_coefficient'], 0.001)
        
        print(f"✅ Created synthetic dataset with {n_samples} samples")
        return self.preprocess_data(df)
    
    def preprocess_data(self, df):
        """
        Preprocess the airfoil data
        """
        print("🔧 Preprocessing airfoil data...")
        
        # Remove any rows with missing values
        df = df.dropna()
        
        # Define features and target
        target_col = 'lift_to_drag_ratio'
        feature_cols = [col for col in df.columns if col != target_col and 
                       col not in ['lift_coefficient', 'drag_coefficient']]
        
        X = df[feature_cols].values
        y = df[target_col].values
        
        self.feature_names = feature_cols
        
        # Remove outliers (L/D ratio should be reasonable)
        valid_indices = (y > 0) & (y < 200)
        X = X[valid_indices]
        y = y[valid_indices]
        
        print(f"📊 Preprocessed data shape: {X.shape}")
        print(f"🎯 Target range: {y.min():.2f} to {y.max():.2f}")
        print(f"📋 Features: {self.feature_names}")
        
        return X, y
    
    def add_feature_engineering(self, X):
        """
        Add engineered features
        """
        X_df = pd.DataFrame(X, columns=self.feature_names)
        
        # Polynomial features
        X_df['aoa_squared'] = X_df['angle_of_attack'] ** 2
        X_df['thickness_camber'] = X_df['max_thickness'] * X_df['max_camber']
        X_df['reynolds_log'] = np.log(X_df['reynolds_number'])
        
        return X_df.values

def load_xfoil_data(data_path='data/xfoil_datasets/'):
    """
    Convenience function to load and preprocess XFoil data
    """
    processor = XFoilDataProcessor()
    return processor.load_xfoil_data(data_path)

if __name__ == "__main__":
    # Test the data processor
    X, y = load_xfoil_data()
    print(f"\n✅ Data loading test completed!")
    print(f"   Features shape: {X.shape}")
    print(f"   Target shape: {y.shape}")
