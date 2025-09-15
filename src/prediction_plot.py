"""
Debug version - Individual Prediction Plots with detailed error tracking
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
import os
import glob
import warnings
warnings.filterwarnings('ignore')

# Try to import optional libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    print("✅ XGBoost available")
except ImportError:
    XGBOOST_AVAILABLE = False
    print("❌ XGBoost not available - install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
    print("✅ LightGBM available")
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("❌ LightGBM not available - install with: pip install lightgbm")

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
    print("✅ CatBoost available")
except ImportError:
    CATBOOST_AVAILABLE = False
    print("❌ CatBoost not available - install with: pip install catboost")

class DebugAirfoilMLPredictor:
    def __init__(self, data_path=r"D:\NAME 400\dipta\airfoil-ml-optimization\data"):
        """
        Initialize with debugging capabilities
        """
        self.data_path = data_path
        
        # Base algorithms that should always work
        self.algorithms = {
            'Linear Regression': LinearRegression(),
            'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=10),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
            'AdaBoost': AdaBoostRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        # Add optional algorithms if available
        if XGBOOST_AVAILABLE:
            self.algorithms['XGBoost'] = xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
        
        if LIGHTGBM_AVAILABLE:
            self.algorithms['LightGBM'] = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbosity=-1)
            
        if CATBOOST_AVAILABLE:
            self.algorithms['CatBoost'] = cb.CatBoostRegressor(n_estimators=100, random_state=42, verbose=False)
        
        print(f"🤖 Initialized {len(self.algorithms)} algorithms: {list(self.algorithms.keys())}")
        
        self.trained_models = {}
        self.predictions = {}
        self.metrics = {}
        self.training_errors = {}
        
        # Create output directory
        self.output_dir = "individual_prediction_plots"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"📁 Created directory: {self.output_dir}")
        
    def load_xfoil_data(self):
        """Load data with better error handling"""
        print(f"📂 Loading XFoil data from: {self.data_path}")
        
        csv_files = glob.glob(os.path.join(self.data_path, "*.csv"))
        
        if not csv_files:
            print("⚠️  No CSV files found!")
            # Try alternative paths
            alternative_paths = [
                "data",
                "../data", 
                "../../data",
                os.path.join(os.getcwd(), "data")
            ]
            
            for alt_path in alternative_paths:
                if os.path.exists(alt_path):
                    csv_files = glob.glob(os.path.join(alt_path, "*.csv"))
                    if csv_files:
                        print(f"✅ Found data in alternative path: {alt_path}")
                        self.data_path = alt_path
                        break
            
            if not csv_files:
                print("❌ No CSV files found in any location!")
                return False
        
        print(f"📄 Found {len(csv_files)} CSV files")
        
        dataframes = []
        for file in csv_files:
            print(f"   Processing: {os.path.basename(file)}")
            try:
                df = pd.read_csv(file)
                print(f"      Shape: {df.shape}, Columns: {list(df.columns)}")
                
                # Add metadata
                filename = os.path.basename(file)
                df['reynolds_number'] = 1000000  # Default
                df['airfoil_code'] = '0012'  # Default
                
                dataframes.append(df)
                
            except Exception as e:
                print(f"      ❌ Error reading {file}: {e}")
                continue
        
        if not dataframes:
            print("❌ No valid data files loaded!")
            return False
            
        self.data = pd.concat(dataframes, ignore_index=True)
        print(f"✅ Loaded {len(self.data)} samples total")
        
        return self.preprocess_data()
    
    def preprocess_data(self):
        """Preprocess with detailed logging"""
        print("🔧 Preprocessing data...")
        
        # Clean column names
        self.data.columns = self.data.columns.str.strip()
        print(f"Available columns: {list(self.data.columns)}")
        
        # Calculate target variable
        if 'CL' in self.data.columns and 'CD' in self.data.columns:
            self.data['lift_to_drag_ratio'] = self.data['CL'] / np.maximum(self.data['CD'], 0.0001)
            target_column = 'lift_to_drag_ratio'
            print("✅ Using L/D ratio as target")
        elif 'CL' in self.data.columns:
            target_column = 'CL'
            print("✅ Using CL as target")
        else:
            print("❌ No suitable target variable found!")
            return False
        
        # Clean data
        initial_size = len(self.data)
        self.data = self.data.dropna()
        print(f"Removed {initial_size - len(self.data)} rows with NaN values")
        
        if 'CD' in self.data.columns:
            self.data = self.data[self.data['CD'] > 0]
        if 'lift_to_drag_ratio' in self.data.columns:
            self.data = self.data[self.data['lift_to_drag_ratio'] > 0]
            self.data = self.data[self.data['lift_to_drag_ratio'] < 500]
        
        print(f"Final dataset size: {len(self.data)} samples")
        
        # Define features
        possible_features = ['alpha', 'CL', 'CD', 'CDp', 'CM', 'Top_Xtr', 'Bot_Xtr', 'reynolds_number']
        available_features = [col for col in possible_features if col in self.data.columns and col != target_column]
        
        print(f"Available features: {available_features}")
        print(f"Target: {target_column}")
        
        if len(available_features) < 1:
            print("❌ Not enough features!")
            return False
        
        # Prepare data
        self.X = self.data[available_features]
        self.y = self.data[target_column]
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        # Scale features
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"✅ Train: {len(self.X_train)}, Test: {len(self.X_test)}")
        return True
        
    def train_all_models(self):
        """Train models with detailed error tracking"""
        print(f"\n🚀 Training {len(self.algorithms)} algorithms...")
        
        successful_models = 0
        
        for name, model in self.algorithms.items():
            print(f"\n   🔄 Training {name}...")
            
            try:
                # Choose data type
                if name in ['Linear Regression']:
                    X_train, X_test = self.X_train_scaled, self.X_test_scaled
                    print(f"      Using scaled data")
                else:
                    X_train, X_test = self.X_train, self.X_test
                    print(f"      Using original data")
                
                # Train model
                model.fit(X_train, self.y_train)
                y_pred = model.predict(X_test)
                
                # Store results
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
                
                successful_models += 1
                print(f"      ✅ SUCCESS - R²: {r2:.4f}, RMSE: {rmse:.4f}")
                
            except Exception as e:
                print(f"      ❌ FAILED - Error: {str(e)}")
                self.training_errors[name] = str(e)
                continue
        
        print(f"\n🎯 Training Summary: {successful_models}/{len(self.algorithms)} models successful")
        
        if self.training_errors:
            print(f"\n❌ Failed models:")
            for name, error in self.training_errors.items():
                print(f"   {name}: {error}")
        
        return successful_models > 0
        
    def plot_individual_predictions_one_by_one(self, figsize=(10, 8)):
        """Create plots one by one with pause between each"""
        print(f"\n📊 Creating individual plots for {len(self.predictions)} models...")
        
        # Colors for each algorithm
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
        
        saved_plots = []
        
        for i, (name, y_pred) in enumerate(self.predictions.items()):
            print(f"\n   🎨 Creating plot {i+1}/{len(self.predictions)}: {name}")
            
            try:
                # Create new figure
                plt.figure(figsize=figsize, dpi=150)
                
                # Get color
                color = algorithm_colors.get(name, '#333333')
                
                # Scatter plot
                plt.scatter(self.y_test, y_pred, alpha=0.7, s=60, 
                           color=color, edgecolors='black', linewidth=0.5)
                
                # Perfect prediction line
                min_val = min(min(self.y_test), min(y_pred))
                max_val = max(max(self.y_test), max(y_pred))
                plt.plot([min_val, max_val], [min_val, max_val], 'r--', 
                        linewidth=3, alpha=0.8, label='Perfect Prediction')
                
                # Labels and title
                plt.xlabel('Actual L/D Ratio', fontsize=14, fontweight='bold')
                plt.ylabel('Predicted L/D Ratio', fontsize=14, fontweight='bold')
                plt.title(f'{name}\nPrediction Performance', fontsize=16, fontweight='bold')
                plt.grid(True, alpha=0.3)
                
                # Metrics
                metrics = self.metrics[name]
                correlation = np.corrcoef(self.y_test, y_pred)[0, 1]
                metrics_text = (f'R² = {metrics["R2"]:.3f}\n'
                              f'RMSE = {metrics["RMSE"]:.2f}\n'
                              f'Correlation = {correlation:.3f}')
                
                plt.text(0.05, 0.95, metrics_text, 
                        transform=plt.gca().transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8),
                        fontsize=12, fontweight='bold')
                
                plt.legend(loc='lower right')
                plt.tight_layout()
                
                # Save plot
                filename = f"{name.replace(' ', '_').lower()}_prediction_plot.png"
                filepath = os.path.join(self.output_dir, filename)
                plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
                saved_plots.append(filename)
                
                print(f"      ✅ Saved: {filename}")
                
                # Show plot
                plt.show()
                
                # Clear figure
                plt.close()
                
            except Exception as e:
                print(f"      ❌ Error creating plot for {name}: {e}")
                continue
        
        print(f"\n🎉 Successfully created {len(saved_plots)} individual plots!")
        print(f"📁 Saved in: {self.output_dir}/")
        for plot in saved_plots:
            print(f"   • {plot}")
        
        return saved_plots

def main():
    """Main execution with comprehensive debugging"""
    print("🚁 DEBUG VERSION - INDIVIDUAL PREDICTION PLOTS")
    print("=" * 80)
    
    # Initialize
    predictor = DebugAirfoilMLPredictor()
    
    # Load data
    print(f"\n📂 STEP 1: Loading data...")
    if not predictor.load_xfoil_data():
        print("❌ Data loading failed!")
        return
    
    # Train models
    print(f"\n🤖 STEP 2: Training models...")
    if not predictor.train_all_models():
        print("❌ No models trained successfully!")
        return
    
    # Create plots
    print(f"\n🎨 STEP 3: Creating individual plots...")
    saved_plots = predictor.plot_individual_predictions_one_by_one()
    
    if saved_plots:
        print(f"\n✅ SUCCESS! Created {len(saved_plots)} plots")
        print(f"📁 Check folder: {predictor.output_dir}")
    else:
        print("❌ No plots were created!")

if __name__ == "__main__":
    main()

