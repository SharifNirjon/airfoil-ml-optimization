"""
Main execution script for Airfoil ML Optimization
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_preprocessing import load_xfoil_data
from ml_algorithms import AirfoilMLComparison
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def main():
    """
    Main execution function
    """
    print("🚀 Airfoil ML Optimization - Starting Analysis")
    print("=" * 60)
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/plots', exist_ok=True)
    
    # Load and preprocess data
    print("\n1️⃣ Loading XFoil Data...")
    X, y = load_xfoil_data('data/xfoil_datasets/')
    
    # Initialize ML comparison
    print("\n2️⃣ Initializing ML Algorithm Comparison...")
    ml_comp = AirfoilMLComparison()
    
    # Run comparison
    print("\n3️⃣ Running Algorithm Comparison...")
    results = ml_comp.compare_algorithms(X, y, test_size=0.2)
    
    # Save results
    print("\n4️⃣ Saving Results...")
    ml_comp.save_results('results/model_performance.csv')
    
    # Display top performers
    print("\n5️⃣ Top 3 Performing Algorithms:")
    top_3 = ml_comp.get_best_algorithms(3)
    print(top_3[['Algorithm', 'R2_Score', 'RMSE', 'Training_Time']].to_string(index=False))
    
    # Create visualization
    print("\n6️⃣ Creating Visualizations...")
    create_performance_plots(results)
    
    print("\n✅ Analysis Complete!")
    print(f"📁 Results saved in 'results/' directory")
    print(f"📊 Performance plots saved in 'results/plots/' directory")

def create_performance_plots(results):
    """
    Create performance visualization plots
    """
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # R2 Score comparison
    axes[0,0].barh(results['Algorithm'], results['R2_Score'])
    axes[0,0].set_title('R² Score Comparison')
    axes[0,0].set_xlabel('R² Score')
    
    # RMSE comparison
    axes[0,1].barh(results['Algorithm'], results['RMSE'])
    axes[0,1].set_title('RMSE Comparison')
    axes[0,1].set_xlabel('RMSE')
    
    # Training time comparison
    axes[1,0].barh(results['Algorithm'], results['Training_Time'])
    axes[1,0].set_title('Training Time Comparison')
    axes[1,0].set_xlabel('Time (seconds)')
    
    # Performance vs Speed scatter
    axes[1,1].scatter(results['Training_Time'], results['R2_Score'])
    for i, txt in enumerate(results['Algorithm']):
        axes[1,1].annotate(txt, (results['Training_Time'].iloc[i], results['R2_Score'].iloc[i]), 
                          rotation=45, fontsize=8)
    axes[1,1].set_xlabel('Training Time (seconds)')
    axes[1,1].set_ylabel('R² Score')
    axes[1,1].set_title('Performance vs Speed')
    
    plt.tight_layout()
    plt.savefig('results/plots/algorithm_comparison.png', dpi=300, bbox_inches='tight')
    print("📊 Performance plots saved to 'results/plots/algorithm_comparison.png'")

if __name__ == "__main__":
    main()
