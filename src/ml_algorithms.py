"""
Airfoil ML Optimization - 8 Algorithm Comparison
Author: Sharif Nirjon
Date: September 2025
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import time
import warnings
from data_preprocessing import XFoilDataProcessor
warnings.filterwarnings('ignore')

class AirfoilMLComparison:
    """
    Comprehensive comparison of 8 ML algorithms for airfoil performance prediction
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.results = []
        
        # Initialize all 8 algorithms
        self.algorithms = {
            'Linear_Regression': LinearRegression(),
            'Decision_Tree': DecisionTreeRegressor(random_state=random_state),
            'Random_Forest': RandomForestRegressor(n_estimators=100, random_state=random_state),
            'AdaBoost': AdaBoostRegressor(random_state=random_state),
            'Gradient_Boosting': GradientBoostingRegressor(random_state=random_state),
            'XGBoost': xgb.XGBRegressor(random_state=random_state, verbosity=0),
            'LightGBM': lgb.LGBMRegressor(random_state=random_state, verbosity=-1),
            'CatBoost': CatBoostRegressor(random_state=random_state, verbose=False)
        }
    
    def evaluate_algorithm(self, model, X_train, X_test, y_train, y_test, name):
        """
        Evaluate a single algorithm with comprehensive metrics
        """
        # Training time
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Prediction time
        start_time = time.time()
        y_pred = model.predict(X_test)
        prediction_time = time.time() - start_time
        
        # Performance metrics
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        return {
            'Algorithm': name,
            'R2_Score': r2,
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'CV_R2_Mean': cv_mean,
            'CV_R2_Std': cv_std,
            'Training_Time': training_time,
            'Prediction_Time': prediction_time
        }
    
    def compare_algorithms(self, X, y, test_size=0.2):
        """
        Compare all 8 algorithms with comprehensive evaluation
        """
        print("🚀 Starting Airfoil ML Algorithm Comparison...")
        print(f"📊 Dataset shape: {X.shape}")
        print(f"🎯 Target variable (L/D ratio) range: {y.min():.2f} to {y.max():.2f}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        results = []
        
        print(f"\n📈 Training and evaluating {len(self.algorithms)} algorithms...")
        
        for name, model in self.algorithms.items():
            print(f"   ⚡ Training {name}...")
            
            # Use scaled data for algorithms that benefit from it
            if name in ['Linear_Regression', 'AdaBoost']:
                result = self.evaluate_algorithm(
                    model, X_train_scaled, X_test_scaled, y_train, y_test, name
                )
            else:
                result = self.evaluate_algorithm(
                    model, X_train, X_test, y_train, y_test, name
                )
            
            results.append(result)
            print(f"      ✅ {name}: R² = {result['R2_Score']:.4f}, "
                  f"RMSE = {result['RMSE']:.4f}, "
                  f"Time = {result['Training_Time']:.3f}s")
        
        # Convert to DataFrame and sort by R2 score
        self.results_df = pd.DataFrame(results)
        self.results_df = self.results_df.sort_values('R2_Score', ascending=False)
        
        print(f"\n🏆 Top performing algorithm: {self.results_df.iloc[0]['Algorithm']} "
              f"(R² = {self.results_df.iloc[0]['R2_Score']:.4f})")
        
        return self.results_df
    
    def get_best_algorithms(self, top_n=3):
        """
        Get top N performing algorithms
        """
        return self.results_df.head(top_n)
    
    def save_results(self, filename='results/model_performance.csv'):
        """
        Save results to CSV file
        """
        self.results_df.to_csv(filename, index=False)
        print(f"📁 Results saved to {filename}")

 if __name__ == "__main__":
    # Load real XFoil airfoil data
    print("🧪 Testing AirfoilMLComparison with real XFoil data...")
    
    # Load data using the same method as main.py
    processor = XFoilDataProcessor()
    X, y = processor.load_xfoil_data(r'D:\NAME 400\dipta\airfoil-ml-optimization\data')
    
    print(f"📊 Loaded real dataset with shape: {X.shape}")
    print(f"🎯 Target range: {y.min():.3f} to {y.max():.3f}")
    
    # Initialize and run comparison
    ml_comp = AirfoilMLComparison()
    results = ml_comp.compare_algorithms(X, y)
    
    print("\n📊 Final Results Summary:")
    print(results[['Algorithm', 'R2_Score', 'RMSE', 'Training_Time']].round(4))
 
