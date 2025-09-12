# Airfoil ML Optimization

Machine Learning Algorithms for Airfoil Aerodynamic Performance Prediction - Comparative Study of 8 ML Algorithms using XFoil Datasets

## 🎯 Project Overview

This research project compares 8 machine learning algorithms for predicting airfoil aerodynamic performance (Lift-to-Drag ratio) using XFoil datasets. The study aims to identify the most effective algorithms for airfoil optimization tasks.

## 🚀 Algorithms Compared

### Traditional ML Algorithms:
1. **Linear Regression** - Baseline linear model
2. **Decision Tree Regressor** - Tree-based decision making
3. **Random Forest** - Ensemble of decision trees
4. **AdaBoost Algorithm** - Adaptive boosting ensemble

### Modern Gradient Boosting:
5. **Gradient Boosting Regression** - Traditional gradient boosting
6. **XGBoost** - Extreme gradient boosting (2016)
7. **LightGBM** - Microsoft's gradient boosting (2017)
8. **CatBoost** - Yandex's categorical boosting (2017)

## 📊 Evaluation Metrics

- **R² Score** - Coefficient of determination
- **Mean Squared Error (MSE)** - Prediction accuracy
- **Mean Absolute Error (MAE)** - Average prediction error
- **Training Time** - Algorithm efficiency
- **Prediction Time** - Inference speed

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/SharifNirjon/airfoil-ml-optimization.git
cd airfoil-ml-optimization

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
