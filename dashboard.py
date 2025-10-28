import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (mean_squared_error, r2_score, accuracy_score, confusion_matrix, 
                            silhouette_score, classification_report, precision_score, recall_score, f1_score, mean_absolute_error)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="ML Algorithms Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    h1 {
        color: #1f77b4;
        padding-bottom: 20px;
    }
    h2 {
        color: #ff7f0e;
        padding-top: 20px;
    }
    .algorithm-description {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.title("🎯 Machine Learning Algorithms Dashboard")
st.markdown("*Comprehensive analysis and visualization of various ML algorithms*")

# Sidebar for navigation and data generation
st.sidebar.title("🎛️ Control Panel")
st.sidebar.markdown("---")

# Dataset Selection
st.sidebar.subheader("📁 Dataset Selection")
dataset_source = st.sidebar.radio(
    "Data Source",
    ["Upload CSV", "Use Sample Datasets", "Generate Synthetic Data"]
)

# Initialize dataset variable
uploaded_data = None
dataset_name = None

if dataset_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=['csv'])
    if uploaded_file is not None:
        uploaded_data = pd.read_csv(uploaded_file)
        dataset_name = uploaded_file.name
        st.sidebar.success(f"✅ Loaded: {dataset_name}")
        st.sidebar.write(f"Shape: {uploaded_data.shape}")
        
elif dataset_source == "Use Sample Datasets":
    sample_dataset = st.sidebar.selectbox(
        "Select Sample Dataset",
        [ "Wine" ]
    )
    dataset_name = sample_dataset
    
    # Load sample datasets
    from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_diabetes
    
    if sample_dataset == "Iris":
        data = load_iris()
        uploaded_data = pd.DataFrame(data.data, columns=data.feature_names)
        uploaded_data['target'] = data.target
    elif sample_dataset == "Wine":
        data = load_wine()
        uploaded_data = pd.DataFrame(data.data, columns=data.feature_names)
        uploaded_data['target'] = data.target
    elif sample_dataset == "Breast Cancer":
        data = load_breast_cancer()
        uploaded_data = pd.DataFrame(data.data, columns=data.feature_names)
        uploaded_data['target'] = data.target
    elif sample_dataset == "Diabetes":
        data = load_diabetes()
        uploaded_data = pd.DataFrame(data.data, columns=data.feature_names)
        uploaded_data['target'] = data.target
    
    st.sidebar.success(f"✅ Loaded: {dataset_name}")
    # st.sidebar.write(f"Shape: {uploaded_data.shape}")

st.sidebar.markdown("---")

# Algorithm selection
algorithm = st.sidebar.selectbox(
    "Select Algorithm",
    ["Overview", "Data Visualization", "Linear Regression", "Decision Tree Classification",
     "Support Vector Machine", "Multivariate Nonlinear Regression", 
     "Clustering (DBSCAN/K-Means)", "Dimensionality Reduction (PCA)", 
     "Ensemble Learning (Bagging/Boosting)"]
)

st.sidebar.markdown("---")
# st.sidebar.subheader("📊 Data Configuration")

# Data generation parameters (only for synthetic data)
if dataset_source == "Generate Synthetic Data":
    n_samples = st.sidebar.slider("Number of Samples", 100, 1000, 300, 50)
    noise_level = st.sidebar.slider("Noise Level", 0.0, 1.0, 0.1, 0.05)
    random_state = st.sidebar.number_input("Random State", 0, 100, 42)
else:
    n_samples = len(uploaded_data) if uploaded_data is not None else 300
    noise_level = 0.1
    random_state = 42

# Generate sample datasets
@st.cache_data
def generate_regression_data(n_samples, noise, random_state):
    np.random.seed(random_state)
    X = np.linspace(0, 10, n_samples)
    y = 2.5 * X + 1.5 + np.random.normal(0, noise * 5, n_samples)
    return X.reshape(-1, 1), y

@st.cache_data
def generate_classification_data(n_samples, random_state):
    np.random.seed(random_state)
    # Create two well-separated classes
    X1 = np.random.randn(n_samples//2, 2) * 0.6 + np.array([2.5, 2.5])
    X2 = np.random.randn(n_samples//2, 2) * 0.6 + np.array([-2.5, -2.5])
    X = np.vstack([X1, X2])
    y = np.hstack([np.zeros(n_samples//2), np.ones(n_samples//2)])
    return X, y

@st.cache_data
def generate_multivariate_data(n_samples, noise, random_state):
    np.random.seed(random_state)
    X = np.random.uniform(0, 10, (n_samples, 2))
    y = 2 * X[:, 0]**2 + 3 * X[:, 1] - X[:, 0] * X[:, 1] + np.random.normal(0, noise * 15, n_samples)
    return X, y

@st.cache_data
def generate_clustering_data(n_samples, random_state):
    np.random.seed(random_state)
    # Create three well-separated clusters
    X1 = np.random.randn(n_samples//3, 2) * 0.4 + np.array([0, 0])
    X2 = np.random.randn(n_samples//3, 2) * 0.4 + np.array([5, 5])
    X3 = np.random.randn(n_samples//3, 2) * 0.4 + np.array([5, 0])
    X = np.vstack([X1, X2, X3])
    return X

# Universal data preparation function
def prepare_data_for_task(data, task_type="classification"):
    """
    Prepare uploaded or sample data for different ML tasks
    task_type: 'classification', 'regression', 'clustering', 'pca'
    """
    if data is None:
        return None, None
    
    df = data.copy()
    
    # Check if 'target' column exists
    if 'target' in df.columns:
        X = df.drop('target', axis=1)
        y = df['target']
    else:
        # Assume last column is target
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
    
    # Handle non-numeric data
    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
    
    if task_type == "clustering" or task_type == "pca":
        return X.values, None
    
    # For classification/regression
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y.astype(str))
    
    # Convert to numpy arrays if they aren't already
    X_array = X.values if hasattr(X, 'values') else X
    y_array = y.values if hasattr(y, 'values') else y
    
    return X_array, y_array

# Smart data loader for each algorithm
def get_data_for_algorithm(algorithm, uploaded_data, dataset_source, n_samples, noise_level, random_state):
    """
    Returns appropriate data based on algorithm and data source
    """
    if dataset_source == "Generate Synthetic Data":
        if algorithm == "Linear Regression":
            return generate_regression_data(n_samples, noise_level, random_state)
        elif algorithm == "Decision Tree Classification" or algorithm == "Support Vector Machine":
            return generate_classification_data(n_samples, random_state)
        elif algorithm == "Multivariate Nonlinear Regression":
            return generate_multivariate_data(n_samples, noise_level, random_state)
        elif algorithm == "Clustering (DBSCAN/K-Means)":
            X = generate_clustering_data(n_samples, random_state)
            return X, None
        elif algorithm == "Dimensionality Reduction (PCA)":
            X, _ = generate_classification_data(n_samples, random_state)
            return X, None
        elif algorithm == "Ensemble Learning (Bagging/Boosting)":
            return generate_classification_data(n_samples, random_state)
    else:
        # Use uploaded or sample dataset
        if algorithm == "Linear Regression" or algorithm == "Multivariate Nonlinear Regression":
            return prepare_data_for_task(uploaded_data, task_type="regression")
        elif algorithm == "Decision Tree Classification" or algorithm == "Support Vector Machine":
            return prepare_data_for_task(uploaded_data, task_type="classification")
        elif algorithm == "Clustering (DBSCAN/K-Means)":
            return prepare_data_for_task(uploaded_data, task_type="clustering")
        elif algorithm == "Dimensionality Reduction (PCA)":
            return prepare_data_for_task(uploaded_data, task_type="pca")
        elif algorithm == "Ensemble Learning (Bagging/Boosting)":
            # Try classification first, if fails use regression
            X, y = prepare_data_for_task(uploaded_data, task_type="classification")
            if X is None or y is None:
                return prepare_data_for_task(uploaded_data, task_type="regression")
            return X, y
    
    return None, None

# Overview Page
if algorithm == "Overview":
    st.header("📚 Machine Learning Algorithms Overview")
    
    if dataset_name:
        st.info(f"📁 **Current Dataset**: {dataset_name}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="algorithm-description">
        <h3>🔵 Linear Regression</h3>
        <p><strong>Purpose:</strong> Predict continuous values based on linear relationships</p>
        <p><strong>Application:</strong> Sales forecasting, price prediction, trend analysis</p>
        <p><strong>Key Metrics:</strong> R² Score, MSE, RMSE</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="algorithm-description">
        <h3>🟢 Support Vector Machine</h3>
        <p><strong>Purpose:</strong> Classification with optimal decision boundaries</p>
        <p><strong>Application:</strong> Image recognition, text classification, medical diagnosis</p>
        <p><strong>Key Metrics:</strong> Accuracy, Precision, Recall, F1-Score</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="algorithm-description">
        <h3>🟣 Multivariate Nonlinear Regression</h3>
        <p><strong>Purpose:</strong> Model complex non-linear relationships</p>
        <p><strong>Application:</strong> Financial modeling, physics simulations, complex predictions</p>
        <p><strong>Key Metrics:</strong> R² Score, Residual Analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="algorithm-description">
        <h3>🟡 DBSCAN / K-Means Clustering</h3>
        <p><strong>Purpose:</strong> Discover natural groupings in data</p>
        <p><strong>Application:</strong> Customer segmentation, anomaly detection, pattern recognition</p>
        <p><strong>Key Metrics:</strong> Silhouette Score, Inertia, Cluster Distribution</p>
        </div>
        """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="algorithm-description">
        <h3>🔴 Decision Tree Classification</h3>
        <p><strong>Purpose:</strong> Rule-based classification with interpretable structure</p>
        <p><strong>Application:</strong> Decision making, risk assessment, rule extraction</p>
        <p><strong>Key Metrics:</strong> Accuracy, Feature Importance, Tree Depth</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="algorithm-description">
        <h3>� PCA Dimensionality Reduction</h3>
        <p><strong>Purpose:</strong> Reduce features while preserving information</p>
        <p><strong>Application:</strong> Visualization, noise reduction, feature extraction</p>
        <p><strong>Key Metrics:</strong> Explained Variance, Reconstruction Error</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="algorithm-description">
        <h3>� Data Visualization</h3>
        <p><strong>Purpose:</strong> Explore and understand dataset characteristics</p>
        <p><strong>Application:</strong> EDA, pattern discovery, data quality check</p>
        <p><strong>Key Metrics:</strong> Distributions, Correlations, Outliers</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("🎯 Quick Stats")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Samples", f"{n_samples}")
    with col2:
        st.metric("Noise Level", f"{noise_level:.2f}")
    with col3:
        st.metric("Random State", f"{random_state}")
    with col4:
        st.metric("Algorithms", "8")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="algorithm-description">
        <h3>🎯 Ensemble Learning (Bagging/Boosting)</h3>
        <p><strong>Purpose:</strong> Combine multiple models for improved accuracy</p>
        <p><strong>Application:</strong> Complex pattern recognition, high-stakes predictions</p>
        <p><strong>Key Metrics:</strong> Accuracy, Feature Importance, Model Comparison</p>
        </div>
        """, unsafe_allow_html=True)

# Data Visualization
elif algorithm == "Data Visualization":
    st.header("📊 Data Visualization & Exploratory Data Analysis")
    
    st.markdown("""
    <div class="algorithm-description">
    <h3>About Data Visualization</h3>
    <p>Data visualization is a crucial first step in understanding your dataset. It helps identify patterns, 
    outliers, correlations, and distributions that guide model selection and feature engineering.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if uploaded_data is not None:
        df = uploaded_data.copy()
        
        # Dataset Overview
        st.subheader("📁 Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", df.shape[0])
        with col2:
            st.metric("Total Columns", df.shape[1])
        with col3:
            st.metric("Numeric Features", len(df.select_dtypes(include=[np.number]).columns))
        with col4:
            st.metric("Missing Values", df.isnull().sum().sum())
        
        # Display sample data
        with st.expander("🔍 View Sample Data"):
            st.dataframe(df.head(10))
        
        # Statistical Summary
        st.subheader("📈 Statistical Summary")
        st.dataframe(df.describe())
        
        # Correlation Heatmap
        st.subheader("🔥 Correlation Heatmap")
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 1:
            corr = numeric_df.corr()
            fig = go.Figure(data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.columns,
                colorscale='RdBu',
                zmid=0,
                text=corr.values.round(2),
                texttemplate='%{text}',
                textfont={"size": 10},
                hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.2f}<extra></extra>'
            ))
            fig.update_layout(
                title="Feature Correlation Matrix",
                height=600,
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Distribution Plots
        st.subheader("📊 Feature Distributions")
        cols = st.multiselect("Select features to visualize", numeric_df.columns.tolist(), 
                             default=numeric_df.columns.tolist()[:3])
        
        if cols:
            fig = make_subplots(rows=1, cols=len(cols), subplot_titles=cols)
            for i, col in enumerate(cols, 1):
                fig.add_trace(
                    go.Histogram(x=numeric_df[col], name=col, nbinsx=30,
                               hovertemplate=f'{col}: %{{x}}<br>Count: %{{y}}<extra></extra>'),
                    row=1, col=i
                )
            fig.update_layout(height=400, showlegend=False, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
        
        # Box Plots for Outlier Detection
        st.subheader("📦 Outlier Detection (Box Plots)")
        if len(numeric_df.columns) > 0:
            fig = go.Figure()
            for col in numeric_df.columns[:6]:  # Limit to 6 features for readability
                fig.add_trace(go.Box(y=numeric_df[col], name=col,
                                   hovertemplate=f'{col}<br>Value: %{{y}}<extra></extra>'))
            fig.update_layout(
                title="Box Plots for Outlier Detection",
                height=450,
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Pairwise Scatter Plots
        st.subheader("🎯 Pairwise Relationships")
        if len(numeric_df.columns) >= 2:
            scatter_cols = st.multiselect("Select 2-3 features for pairwise plots", 
                                         numeric_df.columns.tolist(),
                                         default=numeric_df.columns.tolist()[:2])
            if len(scatter_cols) >= 2:
                if 'target' in df.columns:
                    color_col = 'target'
                else:
                    color_col = None
                
                fig = px.scatter_matrix(
                    df,
                    dimensions=scatter_cols,
                    color=color_col,
                    title="Pairwise Scatter Matrix",
                    height=600
                )
                fig.update_traces(diagonal_visible=False)
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("👆 Please upload a dataset or select a sample dataset to visualize")

# Linear Regression
elif algorithm == "Linear Regression":
    st.header("📈 Linear Regression Analysis")
    
    st.markdown("""
    <div class="algorithm-description">
    <h3>About Linear Regression</h3>
    <p>Linear Regression is a supervised learning algorithm that models the relationship between a dependent 
    variable and one or more independent variables by fitting a linear equation. It's one of the most fundamental 
    and widely used algorithms in machine learning.</p>
    <p><strong>Formula:</strong> y = mx + b (for simple linear regression)</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get data using the smart adapter
    X, y = get_data_for_algorithm(algorithm, uploaded_data, dataset_source, n_samples, noise_level, random_state)
    
    if X is not None and y is not None:
        # Hyperparameter tuning
        col1, col2 = st.columns(2)
        with col1:
            use_regularization = st.checkbox("Use Regularization (Ridge)", value=True)
        with col2:
            if use_regularization:
                alpha = st.slider("Regularization Strength (Alpha)", 0.01, 10.0, 1.0, 0.01)
        
        # Train model with data standardization for better accuracy
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
        
        # Standardize features for better performance
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        if use_regularization:
            model = Ridge(alpha=alpha)
        else:
            model = LinearRegression()
        
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        # Calculate cross-validation score for robustness
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
        cv_r2 = cv_scores.mean()
        
        # Metrics
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        
        # Display metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("R² Score", f"{r2:.4f}")
        with col2:
            st.metric("CV R² Score", f"{cv_r2:.4f}")
        with col3:
            st.metric("MSE", f"{mse:.4f}")
        with col4:
            st.metric("RMSE", f"{rmse:.4f}")
        with col5:
            st.metric("MAE", f"{mae:.4f}")
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Regression line plot (only for univariate)
            if X.shape[1] == 1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=X_train.flatten(), y=y_train,
                    mode='markers',
                    name='Training Data',
                    marker=dict(color='lightblue', size=8, opacity=0.6),
                    hovertemplate='X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>'
                ))
                fig.add_trace(go.Scatter(
                    x=X_test.flatten(), y=y_test,
                    mode='markers',
                    name='Test Data (Actual)',
                    marker=dict(color='orange', size=10, opacity=0.7),
                    hovertemplate='X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>'
                ))
                
                # Regression line
                X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
                y_line = model.predict(X_line)
                fig.add_trace(go.Scatter(
                    x=X_line.flatten(), y=y_line,
                    mode='lines',
                    name='Regression Line',
                    line=dict(color='red', width=3),
                    hovertemplate='X: %{x:.2f}<br>Predicted Y: %{y:.2f}<extra></extra>'
                ))
                
                fig.update_layout(
                    title="Linear Regression: Data and Fitted Line",
                    xaxis_title="X",
                    yaxis_title="Y",
                    hovermode='closest',
                    template="plotly_white",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                # For multivariate, show actual vs predicted
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=y_test, y=y_pred,
                    mode='markers',
                    name='Test Predictions',
                    marker=dict(color='green', size=10, opacity=0.6),
                    hovertemplate='Actual: %{x:.2f}<br>Predicted: %{y:.2f}<extra></extra>'
                ))
                fig.add_trace(go.Scatter(
                    x=[y_test.min(), y_test.max()],
                    y=[y_test.min(), y_test.max()],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(color='red', dash='dash', width=2)
                ))
                
                fig.update_layout(
                    title="Predicted vs Actual Values",
                    xaxis_title="Actual Values",
                    yaxis_title="Predicted Values",
                    hovermode='closest',
                    template="plotly_white",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Residual plot
            residuals = y_test - y_pred
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=y_pred, y=residuals,
                mode='markers',
                marker=dict(color='purple', size=8, opacity=0.6),
                hovertemplate='Predicted: %{x:.2f}<br>Residual: %{y:.2f}<extra></extra>'
            ))
            fig.add_hline(y=0, line_dash="dash", line_color="red", line_width=2)
            
            fig.update_layout(
                title="Residual Plot",
                xaxis_title="Predicted Values",
                yaxis_title="Residuals",
                hovermode='closest',
                template="plotly_white",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Actual vs Predicted Comparison
        fig = go.Figure()
        
        # Actual values
        fig.add_trace(go.Scatter(
            x=list(range(len(y_test))),
            y=y_test,
            mode='markers',
            name='Actual Values',
            marker=dict(color='blue', size=10, symbol='circle'),
            hovertemplate='Index: %{x}<br>Actual: %{y:.2f}<extra></extra>'
        ))
        
        # Predicted values
        fig.add_trace(go.Scatter(
            x=list(range(len(y_pred))),
            y=y_pred,
            mode='markers',
            name='Predicted Values',
            marker=dict(color='red', size=10, symbol='x'),
            hovertemplate='Index: %{x}<br>Predicted: %{y:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Actual vs Predicted Values Comparison",
            xaxis_title="Test Sample Index",
            yaxis_title="Value",
            hovermode='closest',
            template="plotly_white",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Model parameters
        with st.expander("📊 View Model Parameters"):
            st.write(f"**Intercept (b):** {model.intercept_:.4f}")
            if X.shape[1] == 1:
                st.write(f"**Coefficient (m):** {model.coef_[0]:.4f}")
                st.write(f"**Equation:** y = {model.coef_[0]:.4f} * X + {model.intercept_:.4f}")
            else:
                st.write(f"**Coefficients:** {model.coef_}")
                st.write(f"**Number of Features:** {X.shape[1]}")
    else:
        st.error("Unable to prepare data for Linear Regression. Please check your dataset.")

# Support Vector Machine
elif algorithm == "Support Vector Machine":
    st.header("🎯 Support Vector Machine (SVM) Classification")
    
    st.markdown("""
    <div class="algorithm-description">
    <h3>About Support Vector Machine</h3>
    <p>SVM is a powerful supervised learning algorithm used for classification and regression. It works by finding 
    the optimal hyperplane that maximally separates different classes in the feature space. SVMs are particularly 
    effective in high-dimensional spaces.</p>
    <p><strong>Key Concept:</strong> Maximize the margin between decision boundaries</p>
    </div>
    """, unsafe_allow_html=True)
    
    # SVM parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        kernel = st.selectbox("Kernel Type", ["rbf", "linear", "poly"])
    with col2:
        C = st.slider("Regularization (C)", 0.1, 10.0, 1.0, 0.1)
    with col3:
        gamma = st.selectbox("Gamma", ["scale", "auto"])
    
    # Get data
    X, y = get_data_for_algorithm(algorithm, uploaded_data, dataset_source, n_samples, noise_level, random_state)
    
    if X is not None and y is not None:
        # Check if y is continuous and convert to classes if needed
        unique_values = len(np.unique(y))
        if unique_values > 20:  # Too many unique values, likely continuous
            st.warning(f"⚠️ Detected continuous target variable ({unique_values} unique values). Converting to 3 classes for classification.")
            # Convert continuous to classes using quantiles
            y = pd.qcut(y, q=3, labels=[0, 1, 2], duplicates='drop')
            y = y.astype(int)
        
        # Train model with optimized parameters
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Use optimized SVM with probability estimates
        model = SVC(kernel=kernel, C=C, gamma=gamma, random_state=random_state, probability=True, class_weight='balanced')
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        # Calculate cross-validation score
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
        cv_accuracy = cv_scores.mean()
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Accuracy", f"{accuracy:.4f}")
        with col2:
            st.metric("CV Accuracy", f"{cv_accuracy:.4f}")
        with col3:
            st.metric("Precision", f"{precision:.4f}")
        with col4:
            st.metric("Recall", f"{recall:.4f}")
        with col5:
            st.metric("F1-Score", f"{f1:.4f}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Decision boundary (only for 2D data)
            if X.shape[1] == 2:
                st.markdown("#### 🎨 SVM Decision Boundary with Hyperplane & Margins")
                
                # Create a finer mesh for smoother visualization
                h = 0.01  # Finer step size for better resolution
                x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
                y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
                xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
                
                # Get decision function values (distance from hyperplane)
                Z = model.decision_function(scaler.transform(np.c_[xx.ravel(), yy.ravel()]))
                Z = Z.reshape(xx.shape)
                
                # Get predictions for coloring
                Z_pred = model.predict(scaler.transform(np.c_[xx.ravel(), yy.ravel()]))
                Z_pred = Z_pred.reshape(xx.shape)
                
                fig = go.Figure()
                
                # Plot decision regions with smooth gradients
                unique_classes = np.unique(y)
                colors = ['blue', 'red', 'green', 'purple', 'orange']
                
                # Add decision boundary contours (hyperplane and margins)
                fig.add_trace(go.Contour(
                    x=np.arange(x_min, x_max, h),
                    y=np.arange(y_min, y_max, h),
                    z=Z_pred,
                    colorscale=[[0, 'rgba(173, 216, 230, 0.3)'], [1, 'rgba(255, 182, 193, 0.3)']],
                    showscale=False,
                    hoverinfo='skip',
                    contours=dict(
                        showlines=False
                    )
                ))
                
                # Add hyperplane (decision boundary where decision function = 0)
                fig.add_trace(go.Contour(
                    x=np.arange(x_min, x_max, h),
                    y=np.arange(y_min, y_max, h),
                    z=Z,
                    showscale=False,
                    contours=dict(
                        start=0,
                        end=0,
                        size=1,
                        coloring='lines'
                    ),
                    line=dict(width=4, color='black'),
                    hoverinfo='skip',
                    name='Decision Boundary (Hyperplane)'
                ))
                
                # Add margin boundaries (where decision function = ±1)
                fig.add_trace(go.Contour(
                    x=np.arange(x_min, x_max, h),
                    y=np.arange(y_min, y_max, h),
                    z=Z,
                    showscale=False,
                    contours=dict(
                        start=-1,
                        end=-1,
                        size=1,
                        coloring='lines'
                    ),
                    line=dict(width=2, color='darkblue', dash='dash'),
                    hoverinfo='skip',
                    name='Margin (Negative)'
                ))
                
                fig.add_trace(go.Contour(
                    x=np.arange(x_min, x_max, h),
                    y=np.arange(y_min, y_max, h),
                    z=Z,
                    showscale=False,
                    contours=dict(
                        start=1,
                        end=1,
                        size=1,
                        coloring='lines'
                    ),
                    line=dict(width=2, color='darkred', dash='dash'),
                    hoverinfo='skip',
                    name='Margin (Positive)'
                ))
                
                # Plot data points
                for i, class_val in enumerate(unique_classes):
                    mask = y == class_val
                    fig.add_trace(go.Scatter(
                        x=X[mask, 0],
                        y=X[mask, 1],
                        mode='markers',
                        name=f'Class {class_val}',
                        marker=dict(
                            size=10,
                            color=colors[i % len(colors)],
                            opacity=0.8,
                            line=dict(width=1, color='white')
                        ),
                        hovertemplate=f'Class {class_val}<br>X1: %{{x:.2f}}<br>X2: %{{y:.2f}}<extra></extra>'
                    ))
                
                # Highlight support vectors with special markers
                sv = scaler.inverse_transform(model.support_vectors_)
                sv_classes = y[model.support_]
                
                # Plot support vectors by class
                for i, class_val in enumerate(unique_classes):
                    sv_mask = sv_classes == class_val
                    if sv_mask.any():
                        fig.add_trace(go.Scatter(
                            x=sv[sv_mask, 0],
                            y=sv[sv_mask, 1],
                            mode='markers',
                            name=f'Support Vectors (Class {class_val})',
                            marker=dict(
                                size=14,
                                color=colors[i % len(colors)],
                                symbol='circle-open',
                                line=dict(width=3, color='yellow')
                            ),
                            hovertemplate=f'Support Vector<br>Class {class_val}<br>X1: %{{x:.2f}}<br>X2: %{{y:.2f}}<extra></extra>'
                        ))
                
                fig.update_layout(
                    title=f"SVM Decision Boundary ({kernel.upper()} Kernel)<br>Black Line: Hyperplane | Dashed Lines: Margins",
                    xaxis_title="Feature 1",
                    yaxis_title="Feature 2",
                    hovermode='closest',
                    template="plotly_white",
                    height=550,
                    showlegend=True,
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add explanation
                st.info(f"""
                **📘 Understanding the SVM Visualization:**
                - **Black Solid Line**: Decision boundary (hyperplane) that separates classes
                - **Blue & Red Dashed Lines**: Margin boundaries (±1 from hyperplane)
                - **Yellow Circled Points**: Support vectors - the critical points that define the hyperplane
                - **Shaded Regions**: Classification regions for each class
                - **Total Support Vectors**: {len(model.support_vectors_)}
                - **Margin Width**: The distance between the two dashed lines
                
                The SVM maximizes the margin between classes while correctly classifying the training data.
                """)
            else:
                # For higher dimensional data, show first 2 features
                st.info(f"📊 Dataset has {X.shape[1]} features. Showing projection of first 2 features.")
                
                fig = go.Figure()
                unique_classes = np.unique(y)
                colors = ['blue', 'red', 'green', 'purple', 'orange']
                
                for i, class_val in enumerate(unique_classes):
                    mask = y == class_val
                    fig.add_trace(go.Scatter(
                        x=X[mask, 0],
                        y=X[mask, 1],
                        mode='markers',
                        name=f'Class {class_val}',
                        marker=dict(
                            size=8,
                            color=colors[i % len(colors)],
                            opacity=0.7,
                            line=dict(width=1, color='white')
                        ),
                        hovertemplate=f'Class {class_val}<br>Feature 1: %{{x:.2f}}<br>Feature 2: %{{y:.2f}}<extra></extra>'
                    ))
                
                fig.update_layout(
                    title="Feature Space (First 2 Features)",
                    xaxis_title="Feature 1",
                    yaxis_title="Feature 2",
                    hovermode='closest',
                    template="plotly_white",
                    height=450
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Confusion matrix
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=[f'Predicted {i}' for i in unique_classes],
                y=[f'Actual {i}' for i in unique_classes],
                text=cm,
                texttemplate='%{text}',
                colorscale='Blues',
                hovertemplate='%{y}<br>%{x}<br>Count: %{z}<extra></extra>'
            ))
            
            fig.update_layout(
                title="Confusion Matrix",
                template="plotly_white",
                height=450
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Classification report
        with st.expander("📊 Detailed Classification Metrics"):
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            df_report = pd.DataFrame(report).transpose()
            st.dataframe(df_report.style.highlight_max(axis=0))
    else:
        st.error("Unable to prepare data for SVM. Please check your dataset.")

# Decision Tree Classification
elif algorithm == "Decision Tree Classification":
    st.header("🌳 Decision Tree Classification")
    
    st.markdown("""
    <div class="algorithm-description">
    <h3>About Decision Trees</h3>
    <p>Decision Trees are intuitive supervised learning algorithms that create a model predicting target values 
    by learning simple decision rules from data features. They're highly interpretable and can handle both 
    numerical and categorical data.</p>
    <p><strong>Key Concept:</strong> Split data based on feature values to minimize impurity</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Decision Tree parameters
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        max_depth = st.slider("Max Depth", 2, 20, 8)
    with col2:
        min_samples_split = st.slider("Min Samples Split", 2, 20, 5)
    with col3:
        criterion = st.selectbox("Split Criterion", ["gini", "entropy"])
    with col4:
        min_samples_leaf = st.slider("Min Samples Leaf", 1, 10, 2)
    
    # Get data
    X, y = get_data_for_algorithm(algorithm, uploaded_data, dataset_source, n_samples, noise_level, random_state)
    
    if X is not None and y is not None:
        # Check if y is continuous and convert to classes if needed
        unique_values = len(np.unique(y))
        if unique_values > 20:  # Too many unique values, likely continuous
            st.warning(f"⚠️ Detected continuous target variable ({unique_values} unique values). Converting to 3 classes for classification.")
            # Convert continuous to classes using quantiles
            y = pd.qcut(y, q=3, labels=[0, 1, 2], duplicates='drop')
            y = y.astype(int)
        
        # Train model with better parameters
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
        
        model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            criterion=criterion,
            class_weight='balanced',
            random_state=random_state
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        cv_accuracy = cv_scores.mean()
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Accuracy", f"{accuracy:.4f}")
        with col2:
            st.metric("CV Accuracy", f"{cv_accuracy:.4f}")
        with col3:
            st.metric("Precision", f"{precision:.4f}")
        with col4:
            st.metric("Recall", f"{recall:.4f}")
        with col5:
            st.metric("F1-Score", f"{f1:.4f}")
        
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Decision boundary visualization (for 2D data)
            if X.shape[1] == 2:
                h = 0.02
                x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
                y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
                xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
                
                Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                
                fig = go.Figure()
                
                # Decision boundary
                fig.add_trace(go.Contour(
                    x=np.arange(x_min, x_max, h),
                    y=np.arange(y_min, y_max, h),
                    z=Z,
                    colorscale='Viridis',
                    showscale=False,
                    opacity=0.3,
                    hoverinfo='skip'
                ))
                
                # Data points
                unique_classes = np.unique(y)
                colors = px.colors.qualitative.Plotly
                for i, cls in enumerate(unique_classes):
                    mask = y == cls
                    fig.add_trace(go.Scatter(
                        x=X[mask, 0],
                        y=X[mask, 1],
                        mode='markers',
                        name=f'Class {int(cls)}',
                        marker=dict(
                            size=8,
                            color=colors[i % len(colors)],
                            opacity=0.7,
                            line=dict(width=1, color='white')
                        ),
                        hovertemplate=f'Class {int(cls)}<br>X1: %{{x:.2f}}<br>X2: %{{y:.2f}}<extra></extra>'
                    ))
                
                fig.update_layout(
                    title="Decision Tree Decision Boundary",
                    xaxis_title="Feature 1",
                    yaxis_title="Feature 2",
                    hovermode='closest',
                    template="plotly_white",
                    height=450
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Decision boundary visualization requires 2D feature space")
        
        with col2:
            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=[f'Pred {i}' for i in range(len(cm))],
                y=[f'Actual {i}' for i in range(len(cm))],
                text=cm,
                texttemplate='%{text}',
                colorscale='Blues',
                hovertemplate='%{y}<br>%{x}<br>Count: %{z}<extra></extra>'
            ))
            
            fig.update_layout(
                title="Confusion Matrix",
                template="plotly_white",
                height=450
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature Importance
        if X.shape[1] <= 20:  # Only show for reasonable number of features
            st.subheader("📊 Feature Importance")
            feature_names = [f'Feature {i+1}' for i in range(X.shape[1])]
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            fig = go.Figure(go.Bar(
                x=importances[indices],
                y=[feature_names[i] for i in indices],
                orientation='h',
                marker=dict(color='steelblue'),
                hovertemplate='%{y}<br>Importance: %{x:.4f}<extra></extra>'
            ))
            fig.update_layout(
                title="Feature Importance Ranking",
                xaxis_title="Importance",
                yaxis_title="Features",
                height=max(300, len(feature_names) * 25),
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Tree Visualization
        with st.expander("🌳 View Decision Tree Structure"):
            st.write(f"**Tree Depth:** {model.get_depth()}")
            st.write(f"**Number of Leaves:** {model.get_n_leaves()}")
            
            fig, ax = plt.subplots(figsize=(20, 10))
            plot_tree(model, ax=ax, filled=True, feature_names=[f'F{i+1}' for i in range(X.shape[1])],
                     class_names=[f'C{i}' for i in range(len(np.unique(y)))], 
                     rounded=True, fontsize=10)
            st.pyplot(fig)
            plt.close()
        
        # Classification Report
        with st.expander("📊 Detailed Classification Metrics"):
            report = classification_report(y_test, y_pred, output_dict=True)
            df_report = pd.DataFrame(report).transpose()
            st.dataframe(df_report.style.highlight_max(axis=0))
    else:
        st.error("Unable to prepare data for Decision Tree. Please check your dataset.")

# Multivariate Nonlinear Regression
elif algorithm == "Multivariate Nonlinear Regression":
    st.header("📊 Multivariate Nonlinear Regression")
    
    st.markdown("""
    <div class="algorithm-description">
    <h3>About Multivariate Nonlinear Regression</h3>
    <p>This technique extends linear regression to model complex, non-linear relationships between multiple 
    independent variables and a dependent variable. It uses polynomial features to capture non-linear patterns 
    in the data, making it suitable for complex real-world scenarios.</p>
    <p><strong>Application:</strong> Financial modeling, weather prediction, complex system behavior</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Polynomial degree selector
    col1, col2 = st.columns(2)
    with col1:
        degree = st.slider("Polynomial Degree", 1, 5, 2)
    with col2:
        use_ridge = st.checkbox("Use Ridge Regularization", value=True)
    
    # Generate data
    X, y = generate_multivariate_data(n_samples, noise_level, random_state)
    
    # Train model with scaling for better performance
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    
    # Scale the original features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_scaled)
    X_test_poly = poly.transform(X_test_scaled)
    
    if use_ridge:
        model = Ridge(alpha=1.0)
    else:
        model = LinearRegression()
    
    model.fit(X_train_poly, y_train)
    y_pred = model.predict(X_test_poly)
    
    # Calculate cross-validation score
    cv_scores = cross_val_score(model, X_train_poly, y_train, cv=5, scoring='r2')
    cv_r2 = cv_scores.mean()
    
    # Metrics
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("R² Score", f"{r2:.4f}")
    with col2:
        st.metric("CV R² Score", f"{cv_r2:.4f}")
    with col3:
        st.metric("MSE", f"{mse:.4f}")
    with col4:
        st.metric("RMSE", f"{rmse:.4f}")
    with col5:
        st.metric("MAE", f"{mae:.4f}")
    
    st.info(f"✨ **Total Features Created:** {X_train_poly.shape[1]} (from {X.shape[1]} original features using degree {degree} polynomial)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 3D scatter plot with predictions
        fig = go.Figure()
        
        fig.add_trace(go.Scatter3d(
            x=X_train[:, 0],
            y=X_train[:, 1],
            z=y_train,
            mode='markers',
            name='Training Data',
            marker=dict(size=4, color='lightblue', opacity=0.6),
            hovertemplate='X1: %{x:.2f}<br>X2: %{y:.2f}<br>Y: %{z:.2f}<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter3d(
            x=X_test[:, 0],
            y=X_test[:, 1],
            z=y_test,
            mode='markers',
            name='Test Data (Actual)',
            marker=dict(size=6, color='orange', opacity=0.8),
            hovertemplate='X1: %{x:.2f}<br>X2: %{y:.2f}<br>Y: %{z:.2f}<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter3d(
            x=X_test[:, 0],
            y=X_test[:, 1],
            z=y_pred,
            mode='markers',
            name='Test Data (Predicted)',
            marker=dict(size=6, color='red', symbol='diamond', opacity=0.8),
            hovertemplate='X1: %{x:.2f}<br>X2: %{y:.2f}<br>Predicted: %{z:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="3D Visualization: Actual vs Predicted",
            scene=dict(
                xaxis_title='Feature 1',
                yaxis_title='Feature 2',
                zaxis_title='Target Value'
            ),
            height=500,
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Residuals plot
        residuals = y_test - y_pred
        fig = make_subplots(rows=2, cols=1, subplot_titles=('Residuals vs Predicted', 'Residuals Distribution'))
        
        fig.add_trace(
            go.Scatter(
                x=y_pred,
                y=residuals,
                mode='markers',
                marker=dict(color='purple', size=8, opacity=0.6),
                hovertemplate='Predicted: %{x:.2f}<br>Residual: %{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
        
        fig.add_trace(
            go.Histogram(
                x=residuals,
                nbinsx=30,
                marker=dict(color='teal', opacity=0.7),
                hovertemplate='Residual Range: %{x}<br>Count: %{y}<extra></extra>'
            ),
            row=2, col=1
        )
        
        fig.update_xaxes(title_text="Predicted Values", row=1, col=1)
        fig.update_yaxes(title_text="Residuals", row=1, col=1)
        fig.update_xaxes(title_text="Residuals", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        
        fig.update_layout(
            height=500,
            showlegend=False,
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Predicted vs Actual
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_test,
        y=y_pred,
        mode='markers',
        marker=dict(
            size=10,
            color=np.abs(residuals),
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Abs Residual"),
            opacity=0.7
        ),
        hovertemplate='Actual: %{x:.2f}<br>Predicted: %{y:.2f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=[y_test.min(), y_test.max()],
        y=[y_test.min(), y_test.max()],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='red', dash='dash', width=2)
    ))
    
    fig.update_layout(
        title="Predicted vs Actual Values (colored by residual magnitude)",
        xaxis_title="Actual Values",
        yaxis_title="Predicted Values",
        hovermode='closest',
        template="plotly_white",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("📊 Feature Importance Analysis"):
        feature_names = poly.get_feature_names_out(['X1', 'X2'])
        importance = np.abs(model.coef_)
        df_importance = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': model.coef_,
            'Absolute Importance': importance
        }).sort_values('Absolute Importance', ascending=False)
        
        fig = go.Figure(go.Bar(
            x=df_importance['Absolute Importance'][:10],
            y=df_importance['Feature'][:10],
            orientation='h',
            marker=dict(color='steelblue'),
            hovertemplate='Feature: %{y}<br>Importance: %{x:.4f}<extra></extra>'
        ))
        fig.update_layout(
            title="Top 10 Most Important Features",
            xaxis_title="Absolute Coefficient Value",
            yaxis_title="Feature",
            height=400,
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

# Clustering
elif algorithm == "Clustering (DBSCAN/K-Means)":
    st.header("🔍 Clustering Analysis: DBSCAN vs K-Means")
    
    st.markdown("""
    <div class="algorithm-description">
    <h3>About Clustering Algorithms</h3>
    <p><strong>K-Means:</strong> Partitions data into K clusters by minimizing within-cluster variance. 
    Best for spherical, evenly-sized clusters.</p>
    <p><strong>DBSCAN:</strong> Density-based clustering that can find arbitrary-shaped clusters and identify 
    outliers. Great for datasets with noise and varying cluster densities.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Clustering parameters
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("K-Means Parameters")
        n_clusters = st.slider("Number of Clusters (K)", 2, 8, 3)
    
    with col2:
        st.subheader("DBSCAN Parameters")
        eps = st.slider("Epsilon (neighborhood size)", 0.1, 2.0, 0.5, 0.1)
        min_samples = st.slider("Minimum Samples", 2, 20, 5)
    
    # Generate data
    X = generate_clustering_data(n_samples, random_state)
    
    # K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    kmeans_labels = kmeans.fit_predict(X)
    kmeans_silhouette = silhouette_score(X, kmeans_labels)
    
    # DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan_labels = dbscan.fit_predict(X)
    
    # Handle case where DBSCAN finds only one cluster or only noise
    if len(set(dbscan_labels)) > 1 and -1 not in set(dbscan_labels):
        dbscan_silhouette = silhouette_score(X, dbscan_labels)
    elif len(set(dbscan_labels)) > 2:  # Has clusters and noise
        mask = dbscan_labels != -1
        if mask.sum() > 1:
            dbscan_silhouette = silhouette_score(X[mask], dbscan_labels[mask])
        else:
            dbscan_silhouette = 0
    else:
        dbscan_silhouette = 0
    
    # Metrics
    st.subheader("📊 Clustering Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("K-Means Silhouette", f"{kmeans_silhouette:.4f}")
    with col2:
        st.metric("K-Means Inertia", f"{kmeans.inertia_:.2f}")
    with col3:
        st.metric("DBSCAN Silhouette", f"{dbscan_silhouette:.4f}")
    with col4:
        n_dbscan_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
        st.metric("DBSCAN Clusters", n_dbscan_clusters)
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # K-Means visualization
        fig = go.Figure()
        
        # Plot clusters
        for i in range(n_clusters):
            mask = kmeans_labels == i
            fig.add_trace(go.Scatter(
                x=X[mask, 0],
                y=X[mask, 1],
                mode='markers',
                name=f'Cluster {i}',
                marker=dict(size=8, opacity=0.7, line=dict(width=0.5, color='white')),
                hovertemplate=f'Cluster {i}<br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<extra></extra>'
            ))
        
        # Plot centroids
        fig.add_trace(go.Scatter(
            x=kmeans.cluster_centers_[:, 0],
            y=kmeans.cluster_centers_[:, 1],
            mode='markers',
            name='Centroids',
            marker=dict(
                size=15,
                color='black',
                symbol='x',
                line=dict(width=2, color='white')
            ),
            hovertemplate='Centroid<br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="K-Means Clustering Results",
            xaxis_title="Feature 1",
            yaxis_title="Feature 2",
            hovermode='closest',
            template="plotly_white",
            height=450
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # DBSCAN visualization
        fig = go.Figure()
        
        unique_labels = set(dbscan_labels)
        colors = px.colors.qualitative.Plotly
        
        for i, label in enumerate(unique_labels):
            if label == -1:
                # Noise points
                mask = dbscan_labels == label
                fig.add_trace(go.Scatter(
                    x=X[mask, 0],
                    y=X[mask, 1],
                    mode='markers',
                    name='Noise',
                    marker=dict(size=6, color='gray', opacity=0.5, symbol='x'),
                    hovertemplate='Noise Point<br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>'
                ))
            else:
                mask = dbscan_labels == label
                fig.add_trace(go.Scatter(
                    x=X[mask, 0],
                    y=X[mask, 1],
                    mode='markers',
                    name=f'Cluster {label}',
                    marker=dict(size=8, color=colors[i % len(colors)], opacity=0.7,
                              line=dict(width=0.5, color='white')),
                    hovertemplate=f'Cluster {label}<br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<extra></extra>'
                ))
        
        fig.update_layout(
            title="DBSCAN Clustering Results",
            xaxis_title="Feature 1",
            yaxis_title="Feature 2",
            hovermode='closest',
            template="plotly_white",
            height=450
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Comparison plots
    st.subheader("📈 Cluster Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # K-Means cluster sizes
        unique, counts = np.unique(kmeans_labels, return_counts=True)
        fig = go.Figure(data=[
            go.Bar(
                x=[f'Cluster {i}' for i in unique],
                y=counts,
                marker=dict(color=px.colors.qualitative.Plotly[:len(unique)]),
                hovertemplate='%{x}<br>Points: %{y}<extra></extra>'
            )
        ])
        fig.update_layout(
            title="K-Means: Cluster Size Distribution",
            xaxis_title="Cluster",
            yaxis_title="Number of Points",
            template="plotly_white",
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # DBSCAN cluster sizes
        unique, counts = np.unique(dbscan_labels, return_counts=True)
        labels = []
        for u in unique:
            if u == -1:
                labels.append('Noise')
            else:
                labels.append(f'Cluster {u}')
        
        fig = go.Figure(data=[
            go.Bar(
                x=labels,
                y=counts,
                marker=dict(color=['gray' if l == 'Noise' else px.colors.qualitative.Plotly[i % 10] 
                                 for i, l in enumerate(labels)]),
                hovertemplate='%{x}<br>Points: %{y}<extra></extra>'
            )
        ])
        fig.update_layout(
            title="DBSCAN: Cluster Size Distribution",
            xaxis_title="Cluster",
            yaxis_title="Number of Points",
            template="plotly_white",
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Elbow method for K-Means
    with st.expander("📊 K-Means Elbow Method Analysis"):
        st.write("Find the optimal number of clusters by analyzing the elbow curve:")
        
        K_range = range(2, 11)
        inertias = []
        silhouettes = []
        
        for k in K_range:
            km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            km.fit(X)
            inertias.append(km.inertia_)
            silhouettes.append(silhouette_score(X, km.labels_))
        
        fig = make_subplots(rows=1, cols=2, subplot_titles=('Inertia vs K', 'Silhouette Score vs K'))
        
        fig.add_trace(
            go.Scatter(x=list(K_range), y=inertias, mode='lines+markers', 
                      marker=dict(size=10, color='blue'),
                      hovertemplate='K: %{x}<br>Inertia: %{y:.2f}<extra></extra>'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=list(K_range), y=silhouettes, mode='lines+markers',
                      marker=dict(size=10, color='green'),
                      hovertemplate='K: %{x}<br>Silhouette: %{y:.4f}<extra></extra>'),
            row=1, col=2
        )
        
        fig.update_xaxes(title_text="Number of Clusters (K)", row=1, col=1)
        fig.update_xaxes(title_text="Number of Clusters (K)", row=1, col=2)
        fig.update_yaxes(title_text="Inertia", row=1, col=1)
        fig.update_yaxes(title_text="Silhouette Score", row=1, col=2)
        
        fig.update_layout(height=400, showlegend=False, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
        
        optimal_k = list(K_range)[np.argmax(silhouettes)]
        st.info(f"💡 **Suggested optimal K based on Silhouette Score:** {optimal_k}")

# Dimensionality Reduction (PCA)
elif algorithm == "Dimensionality Reduction (PCA)":
    st.header("🔬 Principal Component Analysis (PCA)")
    
    st.markdown("""
    <div class="algorithm-description">
    <h3>About PCA</h3>
    <p>Principal Component Analysis (PCA) is an unsupervised learning technique for dimensionality reduction. 
    It transforms data into a new coordinate system where the axes (principal components) capture the maximum 
    variance in the data. PCA is widely used for data compression, noise reduction, and visualization.</p>
    <p><strong>Key Concept:</strong> Find orthogonal directions of maximum variance</p>
    </div>
    """, unsafe_allow_html=True)
    
    # PCA parameters
    col1, col2 = st.columns(2)
    with col1:
        n_components = st.slider("Number of Components", 2, 10, 3)
    with col2:
        visualize_dims = st.selectbox("Visualization Dimensions", ["2D", "3D"])
    
    # Get data
    X, _ = get_data_for_algorithm(algorithm, uploaded_data, dataset_source, n_samples, noise_level, random_state)
    
    if X is not None:
        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        # Metrics
        explained_var = pca.explained_variance_ratio_
        cumulative_var = np.cumsum(explained_var)
        
        st.subheader("📊 PCA Metrics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Components", n_components)
        with col2:
            st.metric("Original Features", X.shape[1])
        with col3:
            st.metric("Total Var Explained", f"{cumulative_var[-1]:.2%}")
        with col4:
            st.metric("Data Points", X.shape[0])
        
        # Explained Variance
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=[f'PC{i+1}' for i in range(n_components)],
                y=explained_var * 100,
                marker=dict(color='steelblue'),
                hovertemplate='%{x}<br>Variance: %{y:.2f}%<extra></extra>'
            ))
            fig.update_layout(
                title="Explained Variance by Component",
                xaxis_title="Principal Component",
                yaxis_title="Explained Variance (%)",
                template="plotly_white",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(1, n_components + 1)),
                y=cumulative_var * 100,
                mode='lines+markers',
                marker=dict(size=10, color='green'),
                line=dict(width=3),
                hovertemplate='PC%{x}<br>Cumulative: %{y:.2f}%<extra></extra>'
            ))
            fig.update_layout(
                title="Cumulative Explained Variance",
                xaxis_title="Number of Components",
                yaxis_title="Cumulative Variance (%)",
                template="plotly_white",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # PCA Visualization
        st.subheader("🎨 PCA Projection Visualization")
        
        if visualize_dims == "2D":
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=X_pca[:, 0],
                y=X_pca[:, 1],
                mode='markers',
                marker=dict(
                    size=8,
                    color=X_pca[:, 0],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="PC1 Value"),
                    opacity=0.7
                ),
                hovertemplate='PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>'
            ))
            
            # Add axes through origin
            fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1, opacity=0.5)
            fig.add_vline(x=0, line_dash="dash", line_color="gray", line_width=1, opacity=0.5)
            
            fig.update_layout(
                title="2D PCA Projection",
                xaxis_title=f"PC1 ({explained_var[0]:.1%} variance)",
                yaxis_title=f"PC2 ({explained_var[1]:.1%} variance)",
                hovermode='closest',
                template="plotly_white",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        
        else:  # 3D
            if n_components >= 3:
                fig = go.Figure(data=[go.Scatter3d(
                    x=X_pca[:, 0],
                    y=X_pca[:, 1],
                    z=X_pca[:, 2],
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=X_pca[:, 0],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="PC1 Value"),
                        opacity=0.7
                    ),
                    hovertemplate='PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>PC3: %{z:.2f}<extra></extra>'
                )])
                
                fig.update_layout(
                    title="3D PCA Projection",
                    scene=dict(
                        xaxis_title=f"PC1 ({explained_var[0]:.1%})",
                        yaxis_title=f"PC2 ({explained_var[1]:.1%})",
                        zaxis_title=f"PC3 ({explained_var[2]:.1%})"
                    ),
                    height=600,
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Need at least 3 components for 3D visualization")
        
        # Component Loadings
        with st.expander("📊 Principal Component Loadings"):
            if X.shape[1] <= 20:  # Only show for reasonable number of features
                feature_names = [f'Feature {i+1}' for i in range(X.shape[1])]
                loadings_df = pd.DataFrame(
                    pca.components_.T,
                    columns=[f'PC{i+1}' for i in range(n_components)],
                    index=feature_names
                )
                
                fig = go.Figure(data=go.Heatmap(
                    z=loadings_df.values,
                    x=loadings_df.columns,
                    y=loadings_df.index,
                    colorscale='RdBu',
                    zmid=0,
                    text=loadings_df.values.round(2),
                    texttemplate='%{text}',
                    textfont={"size": 10},
                    hovertemplate='%{y}<br>%{x}<br>Loading: %{z:.3f}<extra></extra>'
                ))
                
                fig.update_layout(
                    title="Feature Loadings on Principal Components",
                    height=max(400, len(feature_names) * 20),
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.dataframe(loadings_df.style.background_gradient(cmap='RdBu', axis=None, vmin=-1, vmax=1))
            else:
                st.info(f"Too many features ({X.shape[1]}) to display loadings clearly.")
    else:
        st.error("Unable to prepare data for PCA. Please check your dataset.")

# Ensemble Learning (Bagging/Boosting)
elif algorithm == "Ensemble Learning (Bagging/Boosting)":
    st.header("🎯 Ensemble Learning - Bagging & Boosting")
    
    st.markdown("""
    <div class="algorithm-description">
    <h3>About Ensemble Learning</h3>
    <p>Ensemble methods combine multiple machine learning models to improve prediction accuracy and robustness. 
    They work by aggregating predictions from several base models to reduce overfitting and variance.</p>
    <ul>
        <li><strong>Bagging (Bootstrap Aggregating):</strong> Trains multiple models on different random subsets of the data and averages their predictions</li>
        <li><strong>Boosting:</strong> Sequentially trains models, with each new model focusing on correcting errors made by previous models</li>
        <li><strong>Applications:</strong> Complex pattern recognition, fraud detection, medical diagnosis, financial predictions</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Get data
    X, y = get_data_for_algorithm(algorithm, uploaded_data, dataset_source, n_samples, noise_level, random_state)
    
    if X is not None and y is not None:
        # Check if it's a classification problem
        unique_labels = len(np.unique(y))
        is_classification = unique_labels < 20  # Heuristic: if less than 20 unique values, treat as classification
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)
        
        st.markdown("### 📊 Dataset Information")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Samples", X.shape[0])
        with col2:
            st.metric("Features", X.shape[1])
        with col3:
            st.metric("Training Samples", X_train.shape[0])
        with col4:
            st.metric("Test Samples", X_test.shape[0])
        
        if is_classification:
            st.info(f"🎯 **Task Type:** Classification ({unique_labels} classes)")
        else:
            st.info(f"📈 **Task Type:** Regression")
        
        st.markdown("---")
        
        # Ensemble parameters
        st.markdown("### 🎯 Ensemble Parameters")
        col1, col2 = st.columns(2)
        with col1:
            n_estimators = st.slider("Number of Estimators", 10, 200, 100, 10)
        with col2:
            max_depth = st.slider("Max Depth", 1, 20, 5)
        
        if is_classification:
            # Classification models
            st.markdown("### 🎯 Classification Ensemble Models")
            
            with st.spinner("Training ensemble models..."):
                from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
                from sklearn.tree import DecisionTreeClassifier
                
                # Initialize models
                models = {
                    "Bagging (Random Forest)": RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state),
                    "Bagging (Bagging Classifier)": BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=max_depth), n_estimators=n_estimators, random_state=random_state),
                    "Boosting (Gradient Boosting)": GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state),
                    "Boosting (AdaBoost)": AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=max_depth), n_estimators=n_estimators, random_state=random_state)
                }
                
                results = {}
                predictions = {}
                
                for name, model in models.items():
                    # Train model
                    model.fit(X_train, y_train)
                    
                    # Predictions
                    y_pred = model.predict(X_test)
                    predictions[name] = y_pred
                    
                    # Metrics
                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
                    
                    results[name] = {
                        "Accuracy": accuracy_score(y_test, y_pred),
                        "Precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
                        "Recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
                        "F1-Score": f1_score(y_test, y_pred, average='weighted', zero_division=0)
                    }
                
                st.success("✅ All models trained successfully!")
            
            # Display results comparison
            st.markdown("### 📊 Model Performance Comparison")
            
            results_df = pd.DataFrame(results).T
            st.dataframe(results_df.style.highlight_max(axis=0, color='lightgreen'))
            
            # Visualization: Bar chart comparison
            fig = go.Figure()
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
            
            for i, metric in enumerate(metrics):
                fig.add_trace(go.Bar(
                    name=metric,
                    x=list(results.keys()),
                    y=[results[model][metric] for model in results.keys()],
                    marker_color=colors[i]
                ))
            
            fig.update_layout(
                title="Model Performance Metrics Comparison",
                xaxis_title="Model",
                yaxis_title="Score",
                barmode='group',
                template="plotly_white",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Best model
            best_model = max(results, key=lambda x: results[x]['Accuracy'])
            st.markdown(f"### 🏆 Best Model: **{best_model}**")
            st.metric("Best Accuracy", f"{results[best_model]['Accuracy']:.4f}")
            
            # Confusion Matrix for best model
            st.markdown(f"### 📊 Confusion Matrix - {best_model}")
            from sklearn.metrics import confusion_matrix
            
            cm = confusion_matrix(y_test, predictions[best_model])
            
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=[f'Predicted {i}' for i in range(len(cm))],
                y=[f'Actual {i}' for i in range(len(cm))],
                colorscale='Blues',
                text=cm,
                texttemplate='%{text}',
                textfont={"size": 16},
                hovertemplate='Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>'
            ))
            
            fig.update_layout(
                title=f"Confusion Matrix - {best_model}",
                xaxis_title="Predicted Label",
                yaxis_title="Actual Label",
                height=500,
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature Importance (for tree-based models)
            st.markdown(f"### 🎯 Feature Importance - {best_model}")
            
            best_model_obj = models[best_model]
            if hasattr(best_model_obj, 'feature_importances_'):
                importances = best_model_obj.feature_importances_
                feature_names = [f'Feature {i+1}' for i in range(X.shape[1])]
                
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)
                
                fig = go.Figure(go.Bar(
                    x=importance_df['Importance'],
                    y=importance_df['Feature'],
                    orientation='h',
                    marker_color='#45B7D1'
                ))
                
                fig.update_layout(
                    title="Feature Importance",
                    xaxis_title="Importance Score",
                    yaxis_title="Feature",
                    height=max(400, len(feature_names) * 30),
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.dataframe(importance_df)
            
            # Detailed classification report
            with st.expander("📋 Detailed Classification Report"):
                for name, pred in predictions.items():
                    st.markdown(f"#### {name}")
                    report = classification_report(y_test, pred, output_dict=True, zero_division=0)
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df)
        
        else:
            # Regression models
            st.markdown("### 📈 Regression Ensemble Models")
            
            with st.spinner("Training ensemble models..."):
                from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
                from sklearn.tree import DecisionTreeRegressor
                
                # Initialize models
                models = {
                    "Bagging (Random Forest)": RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state),
                    "Bagging (Bagging Regressor)": BaggingRegressor(estimator=DecisionTreeRegressor(max_depth=max_depth), n_estimators=n_estimators, random_state=random_state),
                    "Boosting (Gradient Boosting)": GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state),
                    "Boosting (AdaBoost)": AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=max_depth), n_estimators=n_estimators, random_state=random_state)
                }
                
                results = {}
                predictions = {}
                
                for name, model in models.items():
                    # Train model
                    model.fit(X_train, y_train)
                    
                    # Predictions
                    y_pred = model.predict(X_test)
                    predictions[name] = y_pred
                    
                    # Metrics
                    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
                    
                    mse = mean_squared_error(y_test, y_pred)
                    results[name] = {
                        "R² Score": r2_score(y_test, y_pred),
                        "MSE": mse,
                        "RMSE": np.sqrt(mse),
                        "MAE": mean_absolute_error(y_test, y_pred)
                    }
                
                st.success("✅ All models trained successfully!")
            
            # Display results comparison
            st.markdown("### 📊 Model Performance Comparison")
            
            results_df = pd.DataFrame(results).T
            
            # Color formatting: higher is better for R², lower is better for errors
            def color_metrics(s):
                if s.name == "R² Score":
                    return ['background-color: lightgreen' if v == s.max() else '' for v in s]
                else:
                    return ['background-color: lightgreen' if v == s.min() else '' for v in s]
            
            st.dataframe(results_df.style.apply(color_metrics, axis=0))
            
            # Visualization: Bar chart comparison
            fig = go.Figure()
            metrics = ['R² Score', 'RMSE', 'MAE']
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
            
            for i, metric in enumerate(metrics):
                fig.add_trace(go.Bar(
                    name=metric,
                    x=list(results.keys()),
                    y=[results[model][metric] for model in results.keys()],
                    marker_color=colors[i]
                ))
            
            fig.update_layout(
                title="Model Performance Metrics Comparison",
                xaxis_title="Model",
                yaxis_title="Score",
                barmode='group',
                template="plotly_white",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Best model
            best_model = max(results, key=lambda x: results[x]['R² Score'])
            st.markdown(f"### 🏆 Best Model: **{best_model}**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("R² Score", f"{results[best_model]['R² Score']:.4f}")
            with col2:
                st.metric("RMSE", f"{results[best_model]['RMSE']:.4f}")
            with col3:
                st.metric("MAE", f"{results[best_model]['MAE']:.4f}")
            
            # Actual vs Predicted for best model
            st.markdown(f"### 📊 Actual vs Predicted - {best_model}")
            
            fig = go.Figure()
            
            # Scatter plot
            fig.add_trace(go.Scatter(
                x=y_test,
                y=predictions[best_model],
                mode='markers',
                name='Predictions',
                marker=dict(size=8, color='#4ECDC4', opacity=0.6)
            ))
            
            # Perfect prediction line
            min_val = min(y_test.min(), predictions[best_model].min())
            max_val = max(y_test.max(), predictions[best_model].max())
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash', width=2)
            ))
            
            fig.update_layout(
                title=f"Actual vs Predicted Values - {best_model}",
                xaxis_title="Actual Values",
                yaxis_title="Predicted Values",
                template="plotly_white",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Residual plot
            st.markdown(f"### 📉 Residual Plot - {best_model}")
            
            residuals = y_test - predictions[best_model]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=predictions[best_model],
                y=residuals,
                mode='markers',
                marker=dict(size=8, color='#FF6B6B', opacity=0.6)
            ))
            
            fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=2)
            
            fig.update_layout(
                title="Residual Plot",
                xaxis_title="Predicted Values",
                yaxis_title="Residuals",
                template="plotly_white",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature Importance (for tree-based models)
            st.markdown(f"### 🎯 Feature Importance - {best_model}")
            
            best_model_obj = models[best_model]
            if hasattr(best_model_obj, 'feature_importances_'):
                importances = best_model_obj.feature_importances_
                feature_names = [f'Feature {i+1}' for i in range(X.shape[1])]
                
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)
                
                fig = go.Figure(go.Bar(
                    x=importance_df['Importance'],
                    y=importance_df['Feature'],
                    orientation='h',
                    marker_color='#45B7D1'
                ))
                
                fig.update_layout(
                    title="Feature Importance",
                    xaxis_title="Importance Score",
                    yaxis_title="Feature",
                    height=max(400, len(feature_names) * 30),
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.dataframe(importance_df)
        
        # Model comparison insights
        st.markdown("---")
        st.markdown("### 💡 Key Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🎒 Bagging Methods:**
            - Reduces variance by averaging multiple models
            - Works well with high-variance models (e.g., deep trees)
            - Random Forest uses random feature subsets
            - Good for reducing overfitting
            """)
        
        with col2:
            st.markdown("""
            **🚀 Boosting Methods:**
            - Reduces bias by sequentially improving models
            - Focuses on hard-to-predict samples
            - Gradient Boosting optimizes loss function
            - Can achieve higher accuracy but may overfit
            """)
    
    else:
        st.error("Unable to prepare data for Ensemble Learning. Please check your dataset.")

# # Footerrr
# st.markdown("---")
# st.markdown("""
#     <div style='text-align: center; color: gray; padding: 20px;'>
#         <p>🎓 Machine Learning Dashboard | Built with Streamlit & Plotly</p>
#         <p>Adjust parameters in the sidebar to explore different scenarios</p>
#     </div>
#     """, unsafe_allow_html=True)
