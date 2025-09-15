"""
FIXED Individual Prediction Plots Generator
Ensures all algorithms generate their plots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import os
import glob
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib backend to prevent display issues
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving
import matplotlib.pyplot as plt

class FixedAirfoilMLPredictor:
    def __init__(self, data_path=r"D:\NAME 400\dipta\airfoil-ml-optimization\data"):
        """
        Initialize with your specific data path and 8 algorithms
        """
        self.data_path = data_path
        self.algorithms = {
            'Linear Regression': LinearRegression(),
            'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=10),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
            'AdaBoost': AdaBoostRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
            'LightGBM': lgb.LGBMRegressor(n_estimators=100, random_state=42, verbosity=-1),
            'CatBoost': cb.CatBoostRegressor(n_estimators=100, random_state=42, verbose=False)
        }
        
        self.trained_models = {}
        self.predictions = {}
        self.metrics = {}
        
        # Create output directory for individual plots
        self.output_dir = "individual_prediction_plots"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
    def load_xfoil_data(self):
        """
        Load XFoil NACA airfoil data from your specific directory
        """
        print(f"📂 Loading XFoil data from: {self.data_path}")
        
        # Find all CSV files in your data directory
        csv_files = glob.glob(os.path.join(self.data_path, "*.csv"))
        
        if not csv_files:
            print("⚠️  No CSV files found in the data directory!")
            print(f"Please check if CSV files exist in: {self.data_path}")
            return False
        
        dataframes = []
        for file in csv_files:
            try:
                df = pd.read_csv(file)
                
                # Extract Reynolds number from filename if possible
                filename = os.path.basename(file)
                if 'Re_' in filename:
                    try:
                        re_value = int(filename.split('Re_')[1].split('_')[0])
                        df['reynolds_number'] = re_value
                    except:
                        df['reynolds_number'] = 1000000  # Default
                else:
                    df['reynolds_number'] = 1000000
                
                # Extract airfoil code
                if 'naca' in filename.lower():
                    airfoil_code = filename.split('naca_')[1].split('_')[0] if 'naca_' in filename else '0012'
                    df['airfoil_code'] = airfoil_code
                else:
                    df['airfoil_code'] = '0012'
                
                dataframes.append(df)
                
            except Exception as e:
                print(f"Error reading {file}: {e}")
                continue
        
        if not dataframes:
            print("❌ No valid data files could be loaded!")
            return False
            
        self.data = pd.concat(dataframes, ignore_index=True)
        print(f"✅ Loaded {len(self.data)} samples from {len(csv_files)} files")
        
        return self.preprocess_data()
    
    def preprocess_data(self):
        """
        Preprocess the NACA airfoil data for ML algorithms
        """
        print("🔧 Preprocessing airfoil data...")
        
        # Clean column names
        self.data.columns = self.data.columns.str.strip()
        
        # Calculate L/D ratio as target variable (if CL and CD exist)
        if 'CL' in self.data.columns and 'CD' in self.data.columns:
            self.data['lift_to_drag_ratio'] = self.data['CL'] / np.maximum(self.data['CD'], 0.0001)
            target_column = 'lift_to_drag_ratio'
        elif 'CL' in self.data.columns:
            target_column = 'CL'  # Use lift coefficient as target
        else:
            print("❌ No suitable target variable found (CL or CD columns)")
            return False
        
        # Remove invalid data
        self.data = self.data.dropna()
        if 'CD' in self.data.columns:
            self.data = self.data[self.data['CD'] > 0]
        if 'lift_to_drag_ratio' in self.data.columns:
            self.data = self.data[self.data['lift_to_drag_ratio'] > 0]
            self.data = self.data[self.data['lift_to_drag_ratio'] < 500]
        
        # Define feature columns based on available data
        possible_features = ['alpha', 'CL', 'CD', 'CDp', 'CM', 'Top_Xtr', 'Bot_Xtr', 'reynolds_number']
        available_features = [col for col in possible_features if col in self.data.columns and col != target_column]
        
        if len(available_features) < 2:
            print(f"❌ Not enough features available. Found: {available_features}")
            return False
        
        print(f"Using features: {available_features}")
        print(f"Target variable: {target_column}")
        
        # Prepare features and target
        self.X = self.data[available_features]
        self.y = self.data[target_column]
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        # Scale the features for algorithms that need it
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"✅ Data preprocessed successfully!")
        print(f"   Training samples: {len(self.X_train)}")
        print(f"   Testing samples: {len(self.X_test)}")
        print(f"   Features: {len(available_features)}")
        
        return True
        
    def train_all_models(self):
        """
        Train all 8 ML algorithms
        """
        print("\n🚀 Training all 8 ML algorithms...")
        
        for name, model in self.algorithms.items():
            print(f"   Training {name}...")
            
            try:
                # Use scaled data for Linear Regression, original for tree-based models
                if name in ['Linear Regression']:
                    model.fit(self.X_train_scaled, self.y_train)
                    y_pred = model.predict(self.X_test_scaled)
                else:
                    model.fit(self.X_train, self.y_train)
                    y_pred = model.predict(self.X_test)
                
                self.trained_models[name] = model
                self.predictions[name] = y_pred
                
                # Calculate metrics
                mse = mean_squared_error(self.y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(self.y_test, y_pred)
                r2 = r2_score(self.y_test, y_pred)
                
                self.metrics[name] = {
                    'MSE': mse,
                    'RMSE': rmse,
                    'MAE': mae,
                    'R2': r2
                }
                
                print(f"      ✅ {name} - R²: {r2:.4f}")
                
            except Exception as e:
                print(f"      ❌ Error training {name}: {e}")
                continue
                
        print("🎉 All models trained successfully!")
        
    def create_single_plot(self, algorithm_name, y_pred, save_only=True):
        """
        Create a single prediction plot for one algorithm
        """
        print(f"   📊 Creating plot for {algorithm_name}...")
        
        # Define colors for each algorithm
        algorithm_colors = {
            'Linear Regression': '#FF6B6B',      # Red
            'Decision Tree': '#4ECDC4',          # Teal
            'Random Forest': '#45B7D1',          # Blue
            'AdaBoost': '#96CEB4',               # Green
            'Gradient Boosting': '#FECA57',      # Yellow
            'XGBoost': '#FF9FF3',                # Pink
            'LightGBM': '#54A0FF',               # Light Blue
            'CatBoost': '#5F27CD'                # Purple
        }
        
        try:
            # Create figure with explicit figure number to avoid conflicts
            fig = plt.figure(figsize=(10, 8), dpi=150)
            
            # Clear any existing plots
            plt.clf()
            
            # Scatter plot of actual vs predicted
            plt.scatter(self.y_test, y_pred, alpha=0.7, s=60, 
                       color=algorithm_colors.get(algorithm_name, '#333333'), 
                       edgecolors='black', linewidth=0.5, 
                       label=f'{algorithm_name} Predictions')
            
            # Perfect prediction line
            min_val = min(min(self.y_test), min(y_pred))
            max_val = max(max(self.y_test), max(y_pred))
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', 
                    linewidth=3, alpha=0.8, label='Perfect Prediction')
            
            # Labels and title
            plt.xlabel('Actual L/D Ratio', fontsize=14, fontweight='bold')
            plt.ylabel('Predicted L/D Ratio', fontsize=14, fontweight='bold')
            plt.title(f'{algorithm_name}\nPrediction Performance', fontsize=16, fontweight='bold', pad=20)
            plt.grid(True, alpha=0.3)
            
            # Add metrics text box
            correlation = np.corrcoef(self.y_test, y_pred)[0, 1]
            metrics_text = (f'R² = {self.metrics[algorithm_name]["R2"]:.3f}\n'
                          f'RMSE = {self.metrics[algorithm_name]["RMSE"]:.2f}\n'
                          f'Correlation = {correlation:.3f}')
            
            plt.text(0.05, 0.95, metrics_text, 
                    transform=plt.gca().transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8),
                    fontsize=12, fontweight='bold')
            
            # Add legend
            plt.legend(loc='lower right', fontsize=11)
            
            # Improve layout
            plt.tight_layout()
            
            # Save the plot
            filename = f"{algorithm_name.replace(' ', '_').lower()}_prediction_plot.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"      ✅ Saved: {filename}")
            
            # Only show if not save_only mode
            if not save_only:
                plt.show()
            
            # Always close the figure to free memory
            plt.close(fig)
            
            return True
            
        except Exception as e:
            print(f"      ❌ Error creating plot for {algorithm_name}: {e}")
            plt.close('all')  # Close any open figures
            return False
    
    def plot_all_individual_predictions(self):
        """
        Create individual prediction plots for each algorithm using separate function calls
        """
        print(f"📊 Generating individual prediction plots...")
        print(f"💾 Saving plots to: {self.output_dir}/")
        print(f"🤖 Processing {len(self.predictions)} algorithms...")
        
        successful_plots = 0
        failed_plots = []
        
        # Process each algorithm separately
        for algorithm_name, y_pred in self.predictions.items():
            try:
                success = self.create_single_plot(algorithm_name, y_pred, save_only=True)
                if success:
                    successful_plots += 1
                else:
                    failed_plots.append(algorithm_name)
                    
            except Exception as e:
                print(f"   ❌ Failed to create plot for {algorithm_name}: {e}")
                failed_plots.append(algorithm_name)
                continue
        
        print(f"\n🎉 Plot Generation Summary:")
        print(f"   ✅ Successful: {successful_plots}/{len(self.predictions)}")
        
        if failed_plots:
            print(f"   ❌ Failed: {failed_plots}")
        
        print(f"📁 All plots saved in: {self.output_dir}/")
        
        # List saved files
        saved_files = []
        for file in os.listdir(self.output_dir):
            if file.endswith('.png'):
                saved_files.append(file)
        
        print(f"💾 Saved files ({len(saved_files)}):")
        for file in saved_files:
            print(f"   • {file}")
            
        return successful_plots

def main():
    """
    Main function to run the fixed individual prediction plots generation
    """
    print("🚁 FIXED AIRFOIL ML ALGORITHMS - INDIVIDUAL PREDICTION PLOTS")
    print("🔬 Generating separate plots for each of 8 algorithms")
    print("=" * 80)
    
    # Initialize predictor
    predictor = FixedAirfoilMLPredictor()
    
    # Load and preprocess data
    if not predictor.load_xfoil_data():
        print("❌ Failed to load data. Please check your data directory and files.")
        return
    
    # Train all models
    predictor.train_all_models()
    
    if not predictor.predictions:
        print("❌ No models were trained successfully.")
        return
    
    print(f"\n✅ Successfully trained {len(predictor.predictions)} models:")
    for name, metrics in predictor.metrics.items():
        print(f"   • {name}: R² = {metrics['R2']:.4f}")
    
    # Generate individual prediction plots
    successful_plots = predictor.plot_all_individual_predictions()
    
    if successful_plots == len(predictor.predictions):
        print(f"\n🎉 SUCCESS! All {successful_plots} individual plots created!")
    else:
        print(f"\n⚠️  Partial success: {successful_plots}/{len(predictor.predictions)} plots created")
    
    print(f"📁 Check the 'individual_prediction_plots' directory for all PNG files")

if __name__ == "__main__":
    main()
