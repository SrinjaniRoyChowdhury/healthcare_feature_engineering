import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import io

# Page configuration
st.set_page_config(
    page_title="Healthcare Feature Engineering Tool",
    page_icon="🏥",
    layout="wide"
)

# Title and description
st.title("Healthcare Feature Engineering Tool")
st.markdown("""
This tool helps you perform automated feature engineering on healthcare datasets.
Upload your dataset and get enhanced features with visualizations!
""")

# Sidebar for navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Upload Data", "Feature Engineering", "Visualizations", "Download"])

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'engineered_data' not in st.session_state:
    st.session_state.engineered_data = None

# Feature Engineering Functions
def create_bmi_category(bmi):
    """Categorize BMI into standard categories"""
    if bmi < 18.5:
        return 'Underweight'
    elif 18.5 <= bmi < 25:
        return 'Normal'
    elif 25 <= bmi < 30:
        return 'Overweight'
    else:
        return 'Obese'

def create_age_group(age):
    """Create age groups"""
    if age < 30:
        return 'Young Adult'
    elif 30 <= age < 50:
        return 'Middle Aged'
    elif 50 <= age < 65:
        return 'Senior'
    else:
        return 'Elderly'

def create_risk_score(row, numeric_cols):
    """Create a simple risk score based on normalized values"""
    score = 0
    for col in numeric_cols:
        if col in row.index:
            score += abs(row[col])
    return score / len(numeric_cols)

def engineer_features(df):
    """Perform comprehensive feature engineering"""
    engineered_df = df.copy()
    
    # Identify numeric and categorical columns
    numeric_cols = engineered_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = engineered_df.select_dtypes(include=['object']).columns.tolist()
    
    st.info(f"Found {len(numeric_cols)} numeric features and {len(categorical_cols)} categorical features")
    
    # 1. Create interaction features (multiply pairs of numeric features)
    if len(numeric_cols) >= 2:
        st.write("Creating interaction features...")
        for i in range(min(3, len(numeric_cols)-1)):
            for j in range(i+1, min(4, len(numeric_cols))):
                col1, col2 = numeric_cols[i], numeric_cols[j]
                engineered_df[f'{col1}_x_{col2}'] = engineered_df[col1] * engineered_df[col2]
    
    # 2. Create polynomial features (squares and cubes)
    st.write("Creating polynomial features...")
    for col in numeric_cols[:5]:  # Limit to first 5 numeric columns
        engineered_df[f'{col}_squared'] = engineered_df[col] ** 2
        engineered_df[f'{col}_cubed'] = engineered_df[col] ** 3
    
    # 3. Create ratio features
    if len(numeric_cols) >= 2:
        st.write("Creating ratio features...")
        for i in range(min(2, len(numeric_cols)-1)):
            col1, col2 = numeric_cols[i], numeric_cols[i+1]
            # Avoid division by zero
            engineered_df[f'{col1}_to_{col2}_ratio'] = engineered_df[col1] / (engineered_df[col2] + 1e-6)
    

    # 4. Create binned features
    st.write("Creating binned features...")
    for col in numeric_cols[:3]:
        try:
            engineered_df[f'{col}_binned'] = pd.qcut(engineered_df[col], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
        except ValueError:
            # If qcut fails, use cut instead
            engineered_df[f'{col}_binned'] = pd.cut(engineered_df[col], bins=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
    
    # 5. Create statistical features (rolling if enough data)
    if len(engineered_df) > 10:
        st.write("Creating statistical features...")
        for col in numeric_cols[:3]:
            engineered_df[f'{col}_rolling_mean'] = engineered_df[col].rolling(window=min(5, len(engineered_df)//2), min_periods=1).mean()
            engineered_df[f'{col}_rolling_std'] = engineered_df[col].rolling(window=min(5, len(engineered_df)//2), min_periods=1).std()
    
    # 6. Create domain-specific features if columns exist
    st.write("Creating domain-specific features...")
    
    # BMI-related features
    if 'BMI' in engineered_df.columns:
        engineered_df['BMI_Category'] = engineered_df['BMI'].apply(create_bmi_category)
    
    # Age-related features
    if 'Age' in engineered_df.columns:
        engineered_df['Age_Group'] = engineered_df['Age'].apply(create_age_group)
    
    # Blood pressure features
    if 'HighBP' in engineered_df.columns and 'HighChol' in engineered_df.columns:
        engineered_df['BP_Chol_Risk'] = engineered_df['HighBP'] + engineered_df['HighChol']
    
    # 7. Create aggregate risk score
    st.write("Creating risk score...")
    # Normalize numeric columns first
    scaler = StandardScaler()
    normalized_cols = numeric_cols[:5]  # Use first 5 numeric columns
    if normalized_cols:
        normalized_data = scaler.fit_transform(engineered_df[normalized_cols].fillna(0))
        engineered_df['Risk_Score'] = np.mean(np.abs(normalized_data), axis=1)
    
    # 8. One-hot encode categorical variables
    if categorical_cols:
        st.write("Encoding categorical features...")
        for col in categorical_cols[:3]:  # Limit to first 3 categorical columns
            dummies = pd.get_dummies(engineered_df[col], prefix=col, drop_first=True)
            engineered_df = pd.concat([engineered_df, dummies], axis=1)
    
    return engineered_df

# PAGE 1: Upload Data
if page == "Upload Data":
    st.header("Upload Your Healthcare Dataset")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    # Sample data option
    if st.button("Use Sample Diabetes Dataset"):
        # Create a sample diabetes-like dataset
        np.random.seed(42)
        n_samples = 200
        sample_data = pd.DataFrame({
            'Age': np.random.randint(20, 80, n_samples),
            'BMI': np.random.normal(28, 6, n_samples),
            'HighBP': np.random.choice([0, 1], n_samples),
            'HighChol': np.random.choice([0, 1], n_samples),
            'CholCheck': np.random.choice([0, 1], n_samples),
            'Smoker': np.random.choice([0, 1], n_samples),
            'HeartDisease': np.random.choice([0, 1], n_samples),
            'PhysActivity': np.random.choice([0, 1], n_samples),
            'Fruits': np.random.choice([0, 1], n_samples),
            'Veggies': np.random.choice([0, 1], n_samples),
            'HvyAlcoholConsump': np.random.choice([0, 1], n_samples),
            'GenHlth': np.random.randint(1, 6, n_samples),
            'MentHlth': np.random.randint(0, 31, n_samples),
            'PhysHlth': np.random.randint(0, 31, n_samples),
            'DiffWalk': np.random.choice([0, 1], n_samples),
            'Diabetes': np.random.choice([0, 1], n_samples)
        })
        st.session_state.data = sample_data
        st.success("Sample dataset loaded successfully!")
    
    if uploaded_file is not None:
        try:
            st.session_state.data = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!")
        except Exception as e:
            st.error(f"Error loading file: {e}")
    
    if st.session_state.data is not None:
        st.subheader("Dataset Preview")
        st.dataframe(st.session_state.data.head(10))
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", st.session_state.data.shape[0])
        with col2:
            st.metric("Total Columns", st.session_state.data.shape[1])
        with col3:
            st.metric("Missing Values", st.session_state.data.isnull().sum().sum())
        
        st.subheader("Column Information")
        st.dataframe(st.session_state.data.dtypes.to_frame('Data Type'))

# PAGE 2: Feature Engineering
elif page == "Feature Engineering":
    st.header("⚙️ Feature Engineering")
    
    if st.session_state.data is None:
        st.warning("Please upload a dataset first!")
    else:
        st.write("Click the button below to perform automated feature engineering on your dataset.")
        
        if st.button("🚀 Start Feature Engineering", type="primary"):
            with st.spinner("Engineering features... This may take a moment..."):
                try:
                    st.session_state.engineered_data = engineer_features(st.session_state.data)
                    st.success("Feature engineering completed!")
                except Exception as e:
                    st.error(f"Error during feature engineering: {e}")
        
        if st.session_state.engineered_data is not None:
            st.subheader("Engineered Dataset Preview")
            st.dataframe(st.session_state.engineered_data.head(10))
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Original Features", st.session_state.data.shape[1])
            with col2:
                new_features = st.session_state.engineered_data.shape[1] - st.session_state.data.shape[1]
                st.metric("New Features Created", new_features)
            
            st.subheader("New Features Summary")
            new_cols = [col for col in st.session_state.engineered_data.columns if col not in st.session_state.data.columns]
            if new_cols:
                st.write(f"**{len(new_cols)} new features created:**")
                st.write(", ".join(new_cols[:20]))  # Show first 20
                if len(new_cols) > 20:
                    st.write(f"... and {len(new_cols) - 20} more")

# PAGE 3: Visualizations
elif page == "Visualizations":
    st.header("📊 Data Visualizations")
    
    if st.session_state.engineered_data is None:
        st.warning("Please perform feature engineering first!")
    else:
        df = st.session_state.engineered_data
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Visualization 1: Feature Distribution
        st.subheader("Feature Distributions")
        selected_feature = st.selectbox("Select a feature to visualize:", numeric_cols)
        
        fig = px.histogram(df, x=selected_feature, nbins=30, 
                          title=f"Distribution of {selected_feature}")
        st.plotly_chart(fig, use_container_width=True)
        
        # Visualization 2: Correlation Heatmap
        st.subheader("Feature Correlation Heatmap")
        corr_features = st.multiselect("Select features for correlation:", 
                                      numeric_cols, 
                                      default=numeric_cols[:min(10, len(numeric_cols))])
        
        if corr_features:
            corr_matrix = df[corr_features].corr()
            fig = px.imshow(corr_matrix, 
                           text_auto=True, 
                           aspect="auto",
                           title="Correlation Matrix",
                           color_continuous_scale='RdBu_r')
            st.plotly_chart(fig, use_container_width=True)
        
        # Visualization 3: Feature Comparison
        st.subheader("Feature Comparison")
        col1, col2 = st.columns(2)
        with col1:
            x_axis = st.selectbox("X-axis:", numeric_cols, key='x')
        with col2:
            y_axis = st.selectbox("Y-axis:", numeric_cols, key='y', index=min(1, len(numeric_cols)-1))
        
        fig = px.scatter(df, x=x_axis, y=y_axis, 
                        title=f"{x_axis} vs {y_axis}",
                        opacity=0.6)
        st.plotly_chart(fig, use_container_width=True)
        
        # Visualization 4: Summary Statistics
        st.subheader("Summary Statistics")
        st.dataframe(df[numeric_cols].describe())

# PAGE 4: Download
elif page == "Download":
    st.header("💾 Download Engineered Dataset")
    
    if st.session_state.engineered_data is None:
        st.warning("Please perform feature engineering first!")
    else:
        st.write("Your engineered dataset is ready for download!")
        
        # Convert to CSV
        csv_buffer = io.StringIO()
        st.session_state.engineered_data.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        
        st.download_button(
            label="📥 Download Engineered Dataset (CSV)",
            data=csv_data,
            file_name="engineered_healthcare_data.csv",
            mime="text/csv",
            type="primary"
        )
        
        # Show summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Features", st.session_state.engineered_data.shape[1])
        with col2:
            st.metric("Total Records", st.session_state.engineered_data.shape[0])
        with col3:
            original_features = st.session_state.data.shape[1]
            new_features = st.session_state.engineered_data.shape[1] - original_features
            st.metric("Features Added", new_features)

# Footer
st.sidebar.markdown("---")
st.sidebar.info("""
**About this tool:**
- Upload healthcare CSV files
- Automated feature engineering
- Interactive visualizations
- Download enhanced datasets

**Feature Engineering Techniques:**
- Interaction features
- Polynomial features
- Ratio features
- Binning
- Statistical aggregations
- Domain-specific features
""")