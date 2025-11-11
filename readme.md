# 🏥 Healthcare Feature Engineering Tool

A user-friendly web application for automated feature engineering on healthcare datasets.

## 🎯 Project Overview

This tool helps healthcare data analysts and researchers perform comprehensive feature engineering on patient data including demographics, medical history, and lab test results. The application provides:

- **Data Upload**: Easy CSV file upload or use sample datasets
- **Automated Feature Engineering**: Creates multiple types of engineered features
- **Interactive Visualizations**: Explore your data with dynamic charts
- **Export Capability**: Download the enhanced dataset

## ✨ Features

### Feature Engineering Techniques
1. **Interaction Features**: Multiply pairs of numeric features
2. **Polynomial Features**: Create squared and cubed terms
3. **Ratio Features**: Calculate ratios between features
4. **Binned Features**: Categorize continuous variables
5. **Statistical Features**: Rolling means and standard deviations
6. **Domain-Specific Features**: BMI categories, age groups, risk scores
7. **One-Hot Encoding**: Convert categorical variables

### Visualizations
- Feature distribution histograms
- Correlation heatmaps
- Scatter plots for feature comparison
- Summary statistics tables

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/healthcare-feature-engineering.git
cd healthcare-feature-engineering
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

4. Open your browser and navigate to `http://localhost:8501`

## 📊 Sample Dataset

The application includes a built-in sample diabetes dataset with the following features:
- Age
- BMI (Body Mass Index)
- Blood Pressure indicators
- Cholesterol levels
- Lifestyle factors (smoking, physical activity)
- Health indicators

## 🌐 Deployment on Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Click "New app"
5. Select your repository and branch
6. Set main file path as `app.py`
7. Click "Deploy"

## 📁 Project Structure

```
healthcare-feature-engineering/
│
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
└── .gitignore            # Git ignore file
```

## 🛠️ Technologies Used

- **Streamlit**: Web application framework
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Plotly**: Interactive visualizations
- **Scikit-learn**: Machine learning utilities

## 📝 How to Use

1. **Upload Data**: Go to "Upload Data" page and upload your CSV file or use the sample dataset
2. **Engineer Features**: Navigate to "Feature Engineering" and click the button to generate new features
3. **Visualize**: Explore your data using various visualization options
4. **Download**: Export your enhanced dataset from the "Download" page

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 👨‍💻 Author

Your Name - [Your GitHub Profile](https://github.com/SrinjaniRoyChowdhury)

## 🙏 Acknowledgments

- Dataset inspired by diabetes health indicators research
- Built with Streamlit framework
- Feature engineering techniques from machine learning best practices

---

**Note**: This tool is for educational and research purposes. Always validate results with domain experts before using engineered features in production healthcare applications.