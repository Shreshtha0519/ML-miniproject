# 🎯 Machine Learning Algorithms Dashboard

> **🎉 NEW: Dashboard now features universal dataset compatibility!**  
> **Just change the dataset name - all algorithms automatically adapt!**
>
> 📖 **Quick Links:**
> - 🚀 [Quick Start Guide](QUICK_START.md) - Get started in 5 minutes
> - 📚 [Complete Usage Guide](README_USAGE.md) - Comprehensive documentation
> - 💡 [Examples & Tutorials](EXAMPLES.md) - Step-by-step examples
> - 🔧 [Modification Summary](MODIFICATION_SUMMARY.md) - Technical details
> - ✅ [Project Status](PROJECT_COMPLETE.md) - Completion summary

A comprehensive, interactive dashboard for visualizing and analyzing various machine learning algorithms including Linear Regression, Support Vector Machines, Multivariate Nonlinear Regression, and Clustering methods (DBSCAN & K-Means).

## 📋 Features

### 1. **Linear Regression** 📈
- **Purpose**: Model linear relationships between variables
- **Visualizations**:
  - Scatter plot with regression line
  - Residual analysis
  - Predicted vs Actual values comparison
- **Metrics**: R² Score, MSE, RMSE, Slope & Intercept
- **Applications**: Sales forecasting, price prediction, trend analysis

### 2. **Support Vector Machine (SVM)** 🎯
- **Purpose**: Classification with optimal decision boundaries
- **Visualizations**:
  - Decision boundary plot with support vectors
  - Confusion matrix heatmap
  - Detailed classification metrics
- **Metrics**: Accuracy, Precision, Recall, F1-Score
- **Applications**: Image recognition, text classification, medical diagnosis
- **Interactive Features**: Adjustable kernel type (linear, RBF, poly) and regularization parameter

### 3. **Multivariate Nonlinear Regression** 📊
- **Purpose**: Model complex non-linear relationships with multiple features
- **Visualizations**:
  - 3D scatter plot (Actual vs Predicted)
  - Residual plots and distribution
  - Feature importance analysis
- **Metrics**: R² Score, MSE, RMSE, Polynomial degree
- **Applications**: Financial modeling, weather prediction, complex system behavior
- **Interactive Features**: Adjustable polynomial degree (1-5)

### 4. **DBSCAN / K-Means Clustering** 🔍
- **Purpose**: Discover natural groupings in unlabeled data
- **Visualizations**:
  - Side-by-side cluster comparison
  - Cluster size distribution
  - Elbow method analysis for K-Means
- **Metrics**: Silhouette Score, Inertia, Cluster count
- **Applications**: Customer segmentation, anomaly detection, pattern recognition
- **Interactive Features**: 
  - K-Means: Adjustable number of clusters
  - DBSCAN: Adjustable epsilon and minimum samples

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone or download this project**
   ```bash
   cd "d:\ml miniproject"
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**
   - Windows (PowerShell):
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
   - Windows (Command Prompt):
     ```cmd
     venv\Scripts\activate.bat
     ```
   - macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

## 🎮 Usage

1. **Run the dashboard**
   ```bash
   streamlit run dashboard.py
   ```

2. **Access the dashboard**
   - The dashboard will automatically open in your default web browser
   - Default URL: http://localhost:8501

3. **Navigate the dashboard**
   - Use the sidebar to select different algorithms
   - Adjust data generation parameters:
     - Number of samples (100-1000)
     - Noise level (0.0-1.0)
     - Random state for reproducibility
   - Each algorithm has specific interactive controls

## 🎨 Dashboard Components

### Control Panel (Sidebar)
- **Algorithm Selection**: Choose which ML algorithm to analyze
- **Dataset Configuration**: 
  - Sample size slider
  - Noise level control
  - Random state setting

### Main Display Area
- **Overview Page**: Summary of all algorithms and their applications
- **Algorithm-Specific Pages**: Detailed analysis with interactive visualizations
- **Performance Metrics**: Real-time calculated metrics displayed prominently
- **Interactive Plots**: Hover for details, zoom, pan, and download capabilities

## 📊 Interactive Features

### Data Filtering & Exploration
- Adjustable data generation parameters
- Real-time model retraining
- Interactive hover information on all plots

### Zooming & Panning
- All Plotly charts support zoom and pan
- Reset view with double-click
- Box select for detailed inspection

### Detailed Information
- Hover over data points for exact values
- Expandable sections for detailed metrics
- Classification reports and feature importance tables

## 🛠️ Technical Details

### Technologies Used
- **Streamlit**: Web application framework
- **Plotly**: Interactive visualizations
- **Scikit-learn**: Machine learning algorithms
- **NumPy & Pandas**: Data manipulation
- **Matplotlib & Seaborn**: Additional plotting support

### Project Structure
```
ml miniproject/
├── dashboard.py           # Main dashboard application
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 📈 Algorithm Descriptions

### Linear Regression
Linear regression finds the best-fitting straight line through data points. The model minimizes the sum of squared residuals to find optimal parameters.

**Formula**: `y = mx + b`
- m: slope (coefficient)
- b: intercept

### Support Vector Machine
SVM finds the optimal hyperplane that maximizes the margin between different classes. Support vectors are the critical data points closest to the decision boundary.

**Key Concepts**:
- Maximum margin classification
- Kernel trick for non-linear boundaries
- C parameter balances margin size and misclassification

### Multivariate Nonlinear Regression
Extends linear regression using polynomial features to capture non-linear relationships. Creates interaction terms and powers of features.

**Example**: For features X₁, X₂ with degree 2:
- Features: [1, X₁, X₂, X₁², X₁X₂, X₂²]

### Clustering Algorithms

**K-Means**:
- Partitioning method
- Requires specifying K (number of clusters)
- Minimizes within-cluster variance
- Best for spherical clusters

**DBSCAN** (Density-Based Spatial Clustering):
- Density-based method
- Automatically determines number of clusters
- Can identify outliers
- Handles arbitrary-shaped clusters

## 🎯 Use Cases

1. **Regression Analysis**: Predict housing prices, stock trends, sales forecasts
2. **Classification**: Spam detection, disease diagnosis, customer churn prediction
3. **Clustering**: Market segmentation, document grouping, image segmentation
4. **Anomaly Detection**: Fraud detection, network intrusion, quality control

## 🔧 Customization

### Adding Custom Datasets
Modify the data generation functions in `dashboard.py`:
- `generate_regression_data()`
- `generate_classification_data()`
- `generate_multivariate_data()`
- `generate_clustering_data()`

### Adjusting Visualizations
All plots use Plotly - easily customizable through:
- Layout updates
- Color schemes
- Axis ranges
- Annotations

## 📝 Tips for Best Results

1. **Start with the Overview page** to understand all available algorithms
2. **Adjust noise levels** to see how algorithms handle real-world data
3. **Experiment with parameters** like polynomial degree, kernel types, and cluster numbers
4. **Use the Elbow Method** to find optimal K for K-Means
5. **Compare DBSCAN and K-Means** to understand their different strengths
6. **Check residual plots** to validate regression assumptions

## 🐛 Troubleshooting

### Common Issues

1. **Port already in use**
   ```bash
   streamlit run dashboard.py --server.port 8502
   ```

2. **Package installation errors**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt --no-cache-dir
   ```

3. **Dashboard not loading**
   - Check if all packages are installed
   - Verify Python version (3.8+)
   - Clear browser cache

## 📚 Additional Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Plotly Python Documentation](https://plotly.com/python/)

## 🤝 Contributing

Feel free to enhance this dashboard by:
- Adding new algorithms
- Improving visualizations
- Adding more datasets
- Enhancing documentation

## 📄 License

This project is open-source and available for educational and commercial use.

---

**Built with ❤️ using Streamlit, Scikit-learn, and Plotly**

For questions or issues, please refer to the documentation or create an issue in the repository.
