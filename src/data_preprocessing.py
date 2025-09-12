"""
XFoil Data Preprocessing for NACA Airfoil Data
Updated for your specific CSV format
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import glob

class XFoilDataProcessor:
    """
    Preprocessor for XFoil airfoil datasets - Updated for NACA polar data
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def load_xfoil_data(self, data_path):
        """
        Load XFoil datasets from directory - handles your polar format
        """
        print(f"📂 Loading XFoil data from: {data_path}")
        
        # Find all CSV files
        csv_files = glob.glob(os.path.join(data_path, "*.csv"))
        
        if not csv_files:
            print("⚠️  No CSV files found. Please add your NACA polar CSV files.")
            return self.create_sample_data()
        
        dataframes = []
        for file in csv_files:
            print(f"   📄 Processing: {os.path.basename(file)}")
            df = pd.read_csv(file)
            
            # Extract metadata from filename if possible
            filename = os.path.basename(file)
            if 'Re_' in filename:
                # Extract Reynolds number from filename like "naca_6616_Re_1000000_polar.csv"
                try:
                    re_value = int(filename.split('Re_')[1].split('_')[0])
                    df['reynolds_number'] = re_value
                except:
                    df['reynolds_number'] = 1000000  # Default
            
            # Extract airfoil name
            if 'naca' in filename.lower():
                airfoil_code = filename.split('naca_')[1].split('_')[0] if 'naca_' in filename else '6616'
                df['airfoil_code'] = airfoil_code
            
            dataframes.append(df)
            
        combined_df = pd.concat(dataframes, ignore_index=True)
        print(f"✅ Loaded {len(combined_df)} samples from {len(csv_files)} files")
        
        return self.preprocess_naca_data(combined_df)
    
    def preprocess_naca_data(self, df):
        """
        Preprocess the NACA airfoil polar data
        """
        print("🔧 Preprocessing NACA airfoil data...")
        
        # Clean column names (remove any extra spaces)
        df.columns = df.columns.str.strip()
        
        # Calculate L/D ratio (our target variable)
        df['lift_to_drag_ratio'] = df['CL'] / np.maximum(df['CD'], 0.0001)  # Avoid division by zero
        
        # Remove any rows with missing or invalid values
        df = df.dropna()
        df = df[df['CD'] > 0]  # Remove negative drag (unphysical)
        df = df[df['lift_to_drag_ratio'] > 0]  # Remove negative L/D
        df = df[df['lift_to_drag_ratio'] < 500]  # Remove unrealistic high L/D
        
        # Define features for neural network
        feature_cols = [
            'alpha',           # Angle of attack
            'CL',             # Lift coefficient  
            'CD',             # Drag coefficient
            'CDp',            # Pressure drag coefficient
            'CM',             # Moment coefficient
            'Top_Xtr',        # Top transition location
            'Bot_Xtr',        # Bottom transition location
        ]
        
        # Add Reynolds number if available
        if 'reynolds_number' in df.columns:
            feature_cols.append('reynolds_number')
        
        # Optionally include iteration counts (convergence indicators)
        if 'Top_Itr' in df.columns and 'Bot_Itr' in df.columns:
            feature_cols.extend(['Top_Itr', 'Bot_Itr'])
        
        # Extract features and target
        available_features = [col for col in feature_cols if col in df.columns]
        X = df[available_features].values
        y = df['lift_to_drag_ratio'].values
        
        self.feature_names = available_features
        
        print(f"📊 Preprocessed data shape: {X.shape}")
        print(f"🎯 L/D ratio range: {y.min():.2f} to {y.max():.2f}")
        print(f"📋 Features: {self.feature_names}")
        print(f"📈 Alpha range: {df['alpha'].min():.1f}° to {df['alpha'].max():.1f}°")
        
        return X, y
    
    def add_aerodynamic_features(self, df):
        """
        Add engineered aerodynamic features
        """
        print("⚙️ Adding aerodynamic feature engineering...")
        
        # Efficiency metrics
        df['efficiency'] = df['CL'] / df['CD']  # Same as L/D ratio
        df['cl_cd_ratio'] = df['CL'] / np.maximum(df['CD'], 0.0001)
        
        # Stall indicators
        df['near_stall'] = (df['alpha'] > 12).astype(int)  # High AoA indicator
        df['cl_alpha'] = df.groupby('airfoil_code')['CL'].diff() / df.groupby('airfoil_code')['alpha'].diff()
        
        # Transition behavior
        df['transition_diff'] = df['Top_Xtr'] - df['Bot_Xtr']
        df['avg_transition'] = (df['Top_Xtr'] + df['Bot_Xtr']) / 2
        
        # Pressure characteristics
        df['total_drag_ratio'] = df['CDp'] / df['CD']
        
        # Polynomial features for non-linear relationships
        df['alpha_squared'] = df['alpha'] ** 2
        df['cl_squared'] = df['CL'] ** 2
        
        return df
    
    def create_sample_data(self):
        """
        Create sample data based on your NACA format for testing
        """
        print("🧪 Creating sample NACA polar data...")
        
        # Generate realistic NACA polar data
        alpha_range = np.linspace(-2, 16, 19)
        data_list = []
        
        for alpha in alpha_range:
            # Realistic CL curve
            cl = 0.1 + 0.11 * alpha - 0.002 * alpha**2 + np.random.normal(0, 0.01)
            
            # Realistic CD polar
            cd = 0.008 + 0.05 * cl**2 + np.random.normal(0, 0.0005)
            cd = max(cd, 0.005)  # Minimum drag
            
            # Other coefficients
            cdp = -0.003 + 0.001 * alpha + np.random.normal(0, 0.0002)
            cm = -0.25 + 0.01 * alpha + np.random.normal(0, 0.005)
            
            # Transition locations
            top_xtr = 0.8 - 0.05 * abs(alpha) + np.random.normal(0, 0.02)
            bot_xtr = 0.3 + 0.03 * alpha + np.random.normal(0, 0.02)
            
            # Clip transition locations
            top_xtr = np.clip(top_xtr, 0.1, 1.0)
            bot_xtr = np.clip(bot_xtr, 0.1, 1.0)
            
            data_list.append({
                'alpha': alpha,
                'CL': cl,
                'CD': cd,
                'CDp': cdp,
                'CM': cm,
                'Top_Xtr': top_xtr,
                'Bot_Xtr': bot_xtr,
                'Top_Itr': np.random.randint(30, 100),
                'Bot_Itr': np.random.randint(150, 250),
                'reynolds_number': 1000000
            })
        
        df = pd.DataFrame(data_list)
        print(f"✅ Created sample dataset with {len(df)} samples")
        return self.preprocess_naca_data(df)

def load_xfoil_data(data_path='data/xfoil_datasets/'):
    """
    Convenience function to load and preprocess your NACA XFoil data
    """
    processor = XFoilDataProcessor()
    return processor.load_xfoil_data(data_path)

if __name__ == "__main__":
    # Test with your data format
    print("🧪 Testing NACA data processor...")
    X, y = load_xfoil_data()
    print(f"\n✅ Data loading test completed!")
    print(f"   Features shape: {X.shape}")
    print(f"   Target shape: {y.shape}")
    print(f"   L/D range: {y.min():.2f} to {y.max():.2f}")
