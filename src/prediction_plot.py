"""
Airfoil ML Prediction Plots Generator
Using 8 Specific Algorithms: Linear Regression, Decision Tree, Random Forest, 
AdaBoost, Gradient Boosting, XGBoost, LightGBM, CatBoost
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

class AirfoilMLPredictor:
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
            print(f"   📄 Processing: {os.path.basename(file)}")
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
        
        # Print available columns for debugging
        print(f"Available columns: {list(self.data.columns)}")
        
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
        
    def plot_individual_predictions(self, figsize=(20, 10)):
        """
        Create individual prediction plots for each algorithm in 2x4 grid
        """
        print("📊 Generating individual prediction plots...")
        
        fig, axes = plt.subplots(2, 4, figsize=figsize)
        axes = axes.ravel()
        
        # Define colors for each algorithm
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        
        for idx, (name, y_pred) in enumerate(self.predictions.items()):
            ax = axes[idx]
            
            # Scatter plot of actual vs predicted
            ax.scatter(self.y_test, y_pred, alpha=0.6, s=40, color=colors[idx], edgecolors='black', linewidth=0.5)
            
            # Perfect prediction line
            min_val = min(min(self.y_test), min(y_pred))
            max_val = max(max(self.y_test), max(y_pred))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.8)
            
            # Labels and title
            ax.set_xlabel('Actual Values', fontsize=11)
            ax.set_ylabel('Predicted Values', fontsize=11)
            ax.set_title(f'{name}\nR² = {self.metrics[name]["R2"]:.4f}', fontsize=12, pad=15)
            ax.grid(True, alpha=0.3)
            
            # Add metrics text box
            correlation = np.corrcoef(self.y_test, y_pred)[0, 1]
            metrics_text = f'R²: {self.metrics[name]["R2"]:.4f}\nRMSE: {self.metrics[name]["RMSE"]:.4f}\nCorr: {correlation:.4f}'
            ax.text(0.05, 0.95, metrics_text, 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                   fontsize=9)
        
        plt.tight_layout()
        plt.suptitle('Airfoil ML Algorithms - Individual Prediction Plots (8 Algorithms)', fontsize=18, y=0.98)
        plt.show()
        
    def plot_combined_predictions(self, figsize=(16, 12)):
        """
        Create a combined plot showing all algorithms
        """
        print("📊 Generating combined prediction plot...")
        
        plt.figure(figsize=figsize)
        
        # Define distinct colors for each algorithm
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3', '#54A0FF', '#5F27CD']
        
        for idx, (name, y_pred) in enumerate(self.predictions.items()):
            plt.scatter(self.y_test, y_pred, alpha=0.7, s=50, 
                       color=colors[idx], label=f'{name} (R²={self.metrics[name]["R2"]:.3f})',
                       edgecolors='black', linewidth=0.5)
        
        # Perfect prediction line
        min_val = min([min(pred) for pred in self.predictions.values()] + [min(self.y_test)])
        max_val = max([max(pred) for pred in self.predictions.values()] + [max(self.y_test)])
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=3, 
                label='Perfect Prediction', alpha=0.8)
        
        plt.xlabel('Actual Values', fontsize=14)
        plt.ylabel('Predicted Values', fontsize=14)
        plt.title('Combined Prediction Plot - All 8 ML Algorithms\nAirfoil Aerodynamic Performance Prediction', fontsize=16)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
    def plot_algorithm_ranking(self, figsize=(14, 8)):
        """
        Create algorithm ranking visualization
        """
        print("📊 Generating algorithm ranking plot...")
        
        metrics_df = pd.DataFrame(self.metrics).T
        ranked_algos = metrics_df.sort_values('R2', ascending=True)  # Ascending for horizontal bar plot
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # R² Score ranking
        colors = plt.cm.viridis(np.linspace(0, 1, len(ranked_algos)))
        bars = ax1.barh(range(len(ranked_algos)), ranked_algos['R2'], color=colors, alpha=0.8)
        ax1.set_title('Algorithm Ranking by R² Score', fontsize=14, pad=20)
        ax1.set_xlabel('R² Score', fontsize=12)
        ax1.set_yticks(range(len(ranked_algos)))
        ax1.set_yticklabels(ranked_algos.index, fontsize=11)
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, ranked_algos['R2'])):
            ax1.text(value + 0.005, bar.get_y() + bar.get_height()/2, 
                    f'{value:.4f}', va='center', fontsize=10)
        
        # RMSE ranking (lower is better, so reverse order)
        ranked_rmse = metrics_df.sort_values('RMSE', ascending=False)
        colors_rmse = plt.cm.plasma(np.linspace(0, 1, len(ranked_rmse)))
        bars2 = ax2.barh(range(len(ranked_rmse)), ranked_rmse['RMSE'], color=colors_rmse, alpha=0.8)
        ax2.set_title('Algorithm Ranking by RMSE\n(Lower is Better)', fontsize=14, pad=20)
        ax2.set_xlabel('RMSE', fontsize=12)
        ax2.set_yticks(range(len(ranked_rmse)))
        ax2.set_yticklabels(ranked_rmse.index, fontsize=11)
        ax2.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.show()
        
    def plot_metrics_comparison(self, figsize=(16, 12)):
        """
        Create comprehensive metrics comparison
        """
        print("📊 Generating comprehensive metrics comparison...")
        
        metrics_df = pd.DataFrame(self.metrics).T
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Color scheme
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3', '#54A0FF', '#5F27CD']
        
        # R² Score
        ax1 = axes[0, 0]
        bars1 = ax1.bar(range(len(metrics_df)), metrics_df['R2'], color=colors, alpha=0.8)
        ax1.set_title('R² Score Comparison\n(Higher = Better)', fontsize=13, pad=15)
        ax1.set_ylabel('R² Score', fontsize=11)
        ax1.set_xticks(range(len(metrics_df)))
        ax1.set_xticklabels(metrics_df.index, rotation=45, ha='right', fontsize=10)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, value in zip(bars1, metrics_df['R2']):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        # RMSE
        ax2 = axes[0, 1]
        bars2 = ax2.bar(range(len(metrics_df)), metrics_df['RMSE'], color=colors, alpha=0.8)
        ax2.set_title('RMSE Comparison\n(Lower = Better)', fontsize=13, pad=15)
        ax2.set_ylabel('RMSE', fontsize=11)
        ax2.set_xticks(range(len(metrics_df)))
        ax2.set_xticklabels(metrics_df.index, rotation=45, ha='right', fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # MAE
        ax3 = axes[1, 0]
        bars3 = ax3.bar(range(len(metrics_df)), metrics_df['MAE'], color=colors, alpha=0.8)
        ax3.set_title('MAE Comparison\n(Lower = Better)', fontsize=13, pad=15)
        ax3.set_ylabel('MAE', fontsize=11)
        ax3.set_xticks(range(len(metrics_df)))
        ax3.set_xticklabels(metrics_df.index, rotation=45, ha='right', fontsize=10)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # MSE
        ax4 = axes[1, 1]
        bars4 = ax4.bar(range(len(metrics_df)), metrics_df['MSE'], color=colors, alpha=0.8)
        ax4.set_title('MSE Comparison\n(Lower = Better)', fontsize=13, pad=15)
        ax4.set_ylabel('MSE', fontsize=11)
        ax4.set_xticks(range(len(metrics_df)))
        ax4.set_xticklabels(metrics_df.index, rotation=45, ha='right', fontsize=10)
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()
        
    def print_detailed_summary(self):
        """
        Print comprehensive analysis summary
        """
        print("\n" + "="*100)
        print("🎯 AIRFOIL ML ALGORITHMS - PERFORMANCE ANALYSIS")
        print("   Linear Regression | Decision Tree | Random Forest | AdaBoost")
        print("   Gradient Boosting | XGBoost | LightGBM | CatBoost")
        print("="*100)
        
        metrics_df = pd.DataFrame(self.metrics).T
        metrics_df = metrics_df.round(4)
        
        # Sort by R² score (descending)
        metrics_df_sorted = metrics_df.sort_values('R2', ascending=False)
        
        print(f"\n📊 Dataset Information:")
        print(f"   • Training samples: {len(self.X_train)}")
        print(f"   • Testing samples: {len(self.X_test)}")
        print(f"   • Features: {len(self.X.columns)}")
        print(f"   • Target variable range: {self.y.min():.4f} to {self.y.max():.4f}")
        
        print(f"\n📈 Performance Metrics Summary:")
        print(metrics_df_sorted.to_string())
        
        print(f"\n🏆 ALGORITHM RANKINGS:")
        print("-" * 70)
        print(f"🥇 Best Overall (R²): {metrics_df_sorted.index[0]} ({metrics_df_sorted.iloc[0]['R2']:.4f})")
        print(f"🥈 Second Best: {metrics_df_sorted.index[1]} ({metrics_df_sorted.iloc[1]['R2']:.4f})")
        print(f"🥉 Third Best: {metrics_df_sorted.index[2]} ({metrics_df_sorted.iloc[2]['R2']:.4f})")
        print(f"🎯 Lowest RMSE: {metrics_df['RMSE'].idxmin()} ({metrics_df.loc[metrics_df['RMSE'].idxmin(), 'RMSE']:.4f})")
        print(f"📉 Lowest MAE: {metrics_df['MAE'].idxmin()} ({metrics_df.loc[metrics_df['MAE'].idxmin(), 'MAE']:.4f})")
        
        # Algorithm categories
        print(f"\n📊 Algorithm Performance Categories:")
        excellent = metrics_df_sorted[metrics_df_sorted['R2'] >= 0.9]
        good = metrics_df_sorted[(metrics_df_sorted['R2'] >= 0.8) & (metrics_df_sorted['R2'] < 0.9)]
        fair = metrics_df_sorted[(metrics_df_sorted['R2'] >= 0.6) & (metrics_df_sorted['R2'] < 0.8)]
        poor = metrics_df_sorted[metrics_df_sorted['R2'] < 0.6]
        
        if len(excellent) > 0:
            print(f"   🌟 Excellent (R² ≥ 0.9): {', '.join(excellent.index)}")
        if len(good) > 0:
            print(f"   ✅ Good (0.8 ≤ R² < 0.9): {', '.join(good.index)}")
        if len(fair) > 0:
            print(f"   ⚠️  Fair (0.6 ≤ R² < 0.8): {', '.join(fair.index)}")
        if len(poor) > 0:
            print(f"   ❌ Needs Improvement (R² < 0.6): {', '.join(poor.index)}")
        
        # Algorithm insights
        print(f"\n🔍 Algorithm Insights:")
        tree_based = ['Decision Tree', 'Random Forest', 'AdaBoost', 'Gradient Boosting', 'XGBoost', 'LightGBM', 'CatBoost']
        tree_performance = metrics_df_sorted[metrics_df_sorted.index.isin(tree_based)]
        best_tree = tree_performance.index[0] if len(tree_performance) > 0 else None
        
        if best_tree:
            print(f"   🌳 Best Tree-based Algorithm: {best_tree}")
        if 'Linear Regression' in metrics_df_sorted.index:
            lr_rank = list(metrics_df_sorted.index).index('Linear Regression') + 1
            print(f"   📊 Linear Regression Rank: #{lr_rank} out of 8")

def main():
    """
    Main function to run the complete airfoil ML analysis
    """
    print("🚁 AIRFOIL ML ALGORITHMS - PREDICTION ANALYSIS")
    print("🔬 Using 8 Algorithms: Linear Regression, Decision Tree, Random Forest,")
    print("   AdaBoost, Gradient Boosting, XGBoost, LightGBM, CatBoost")
    print("=" * 80)
    
    # Initialize predictor with your data path
    predictor = AirfoilMLPredictor()
    
    # Load and preprocess data
    if not predictor.load_xfoil_data():
        print("❌ Failed to load data. Please check your data directory and files.")
        return
    
    # Train all models
    predictor.train_all_models()
    
    if not predictor.predictions:
        print("❌ No models were trained successfully.")
        return
    
    # Generate all visualization plots
    print("\n📊 Generating comprehensive visualization plots...")
    predictor.plot_individual_predictions()
    predictor.plot_combined_predictions()
    predictor.plot_algorithm_ranking()
    predictor.plot_metrics_comparison()
    
    # Print detailed analysis
    predictor.print_detailed_summary()
    
    print("\n🎉 Analysis complete! All prediction plots have been generated.")
    print("📁 Check the generated plots to compare algorithm performance.")

if __name__ == "__main__":
    main()