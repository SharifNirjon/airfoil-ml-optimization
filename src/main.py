"""
Main execution script for TensorFlow-based Airfoil Neural Network Optimization
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_preprocessing import load_xfoil_data
from neural_networks import AirfoilNeuralNetworks
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tensorflow as tf

def main():
    """
    Main execution function for neural network comparison
    """
    print("🧠 TensorFlow Airfoil Neural Network Optimization")
    print("=" * 70)
    
    # Check TensorFlow version and GPU availability
    print(f"📦 TensorFlow version: {tf.__version__}")
    if tf.config.list_physical_devices('GPU'):
        print("🚀 GPU acceleration available!")
    else:
        print("💻 Using CPU (consider GPU for faster training)")
    
    # Create directories
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/plots', exist_ok=True)
    os.makedirs('models/saved_models', exist_ok=True)
    os.makedirs('logs/tensorboard', exist_ok=True)
    
    # Load data
    print("\n1️⃣ Loading XFoil Data...")
    from data_preprocessing import XFoilDataProcessor
    processor = XFoilDataProcessor()
    data = processor.load_xfoil_data(r'D:\NAME 400\dipta\airfoil-ml-optimization\data')
    # Extract features and target from the processed data
    X = data[processor.feature_names]
    y = data['lift_to_drag_ratio']
    # Initialize neural network comparison
    print("\n2️⃣ Initializing Neural Network Architectures...")
    nn_comp = AirfoilNeuralNetworks()
    
    # Run comparison
    print("\n3️⃣ Training 8 Neural Network Architectures...")
    print("   ⏱️  This may take 10-30 minutes depending on your hardware...")
    results = nn_comp.compare_architectures(X, y)
    
    # Save results
    print("\n4️⃣ Saving Results...")
    nn_comp.save_results('results/architecture_performance.csv')
    
    # Display results
    print("\n5️⃣ Top 3 Performing Architectures:")
    top_3 = results.head(3)
    print(top_3[['Architecture', 'R2_Score', 'RMSE', 'Total_Parameters', 'Training_Time']].to_string(index=False))
    
    # Create visualizations
    print("\n6️⃣ Creating Visualizations...")
    create_architecture_plots(results, nn_comp.histories)
    
    print("\n7️⃣ TensorBoard Visualization:")
    print("   Run: tensorboard --logdir=logs/tensorboard")
    print("   Then open: http://localhost:6006")
    
    print("\n✅ Neural Network Analysis Complete!")
    print(f"📁 Results saved in 'results/' directory")
    print(f"🧠 Trained models saved in 'models/saved_models/' directory")
    print(f"📊 TensorBoard logs in 'logs/tensorboard/' directory")

def create_architecture_plots(results, histories):
    """
    Create comprehensive visualization plots for neural architectures
    """
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # R2 Score comparison
    results_sorted = results.sort_values('R2_Score', ascending=True)
    axes[0,0].barh(results_sorted['Architecture'], results_sorted['R2_Score'], color='skyblue')
    axes[0,0].set_title('R² Score Comparison', fontsize=14, fontweight='bold')
    axes[0,0].set_xlabel('R² Score')
    axes[0,0].grid(axis='x', alpha=0.3)
    
    # RMSE comparison
    axes[0,1].barh(results_sorted['Architecture'], results_sorted['RMSE'], color='lightcoral')
    axes[0,1].set_title('RMSE Comparison', fontsize=14, fontweight='bold')
    axes[0,1].set_xlabel('RMSE')
    axes[0,1].grid(axis='x', alpha=0.3)
    
    # Model complexity (parameters)
    axes[0,2].barh(results_sorted['Architecture'], results_sorted['Total_Parameters'], color='lightgreen')
    axes[0,2].set_title('Model Complexity (Parameters)', fontsize=14, fontweight='bold')
    axes[0,2].set_xlabel('Number of Parameters')
    axes[0,2].grid(axis='x', alpha=0.3)
    
    # Training time comparison
    axes[1,0].barh(results_sorted['Architecture'], results_sorted['Training_Time'], color='orange')
    axes[1,0].set_title('Training Time Comparison', fontsize=14, fontweight='bold')
    axes[1,0].set_xlabel('Time (seconds)')
    axes[1,0].grid(axis='x', alpha=0.3)
    
    # Performance vs Complexity scatter
    axes[1,1].scatter(results['Total_Parameters'], results['R2_Score'], 
                     s=100, alpha=0.7, c='purple')
    for i, txt in enumerate(results['Architecture']):
        axes[1,1].annotate(txt, (results['Total_Parameters'].iloc[i], results['R2_Score'].iloc[i]), 
                          rotation=45, fontsize=8, ha='left')
    axes[1,1].set_xlabel('Model Parameters')
    axes[1,1].set_ylabel('R² Score')
    axes[1,1].set_title('Performance vs Complexity', fontsize=14, fontweight='bold')
    axes[1,1].grid(alpha=0.3)
    
    # Training convergence for top 3 models
    top_3_names = results.head(3)['Architecture'].tolist()
    colors = ['blue', 'red', 'green']
    
    for i, name in enumerate(top_3_names):
        if name in histories and 'val_loss' in histories[name].history:
            epochs = range(1, len(histories[name].history['val_loss']) + 1)
            axes[1,2].plot(epochs, histories[name].history['val_loss'], 
                          color=colors[i], label=f'{name}', linewidth=2)
    
    axes[1,2].set_xlabel('Epoch')
    axes[1,2].set_ylabel('Validation Loss')
    axes[1,2].set_title('Training Convergence (Top 3)', fontsize=14, fontweight='bold')
    axes[1,2].legend()
    axes[1,2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/plots/neural_architecture_comparison.png', dpi=300, bbox_inches='tight')
    print("📊 Architecture comparison plots saved to 'results/plots/neural_architecture_comparison.png'")

if __name__ == "__main__":
    main()
