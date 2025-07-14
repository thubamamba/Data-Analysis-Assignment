# Week 3 Project: Sales Data Analysis

## Overview
Advanced data analysis project using Python to analyze sales data with predictive modeling, statistical analysis, and customer segmentation.

## Project Structure
```
week3-project/
├── README.md
├── sales_data_analysis.py
├── raw_sales_data.xlsx
├── output/
│   ├── cleaned_sales_data_TIMESTAMP.xlsx
│   ├── analysis_summary_TIMESTAMP.xlsx
│   ├── customer_segments_TIMESTAMP.xlsx
│   ├── regional_analysis_TIMESTAMP.xlsx
│   ├── model_predictions_TIMESTAMP.xlsx
│   ├── sales_analysis_charts_TIMESTAMP.png
│   ├── sales_analysis_charts_TIMESTAMP.pdf
│   ├── regional_sales_chart_TIMESTAMP.png
│   ├── customer_segmentation_chart_TIMESTAMP.png
│   └── churn_analysis_chart_TIMESTAMP.png
└── requirements.txt
```

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/week3-project.git
cd week3-project
```

### 2. Create Virtual Environment

**Windows:**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate
```

**Mac/Linux:**
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

### 3. Install Required Packages

```bash
# Install all required packages
pip install -r requirements.txt
```

## Usage

### 1. Prepare Data
- Place `raw_sales_data.xlsx` in the project folder

### 2. Run Analysis
```bash
python sales_data_analysis.py
```

### 3. Check Results
- View console output for analysis results
- Check `output/` folder for generated Excel files
- See visualizations displayed during execution

## What You Get

### Generated Files
**Excel Files:**
- `cleaned_sales_data_TIMESTAMP.xlsx` - Cleaned dataset
- `analysis_summary_TIMESTAMP.xlsx` - Key metrics
- `customer_segments_TIMESTAMP.xlsx` - Segmentation results
- `regional_analysis_TIMESTAMP.xlsx` - Regional performance
- `model_predictions_TIMESTAMP.xlsx` - Model predictions and errors

**Visualization Files:**
- `sales_analysis_charts_TIMESTAMP.png` - All 4 charts combined
- `sales_analysis_charts_TIMESTAMP.pdf` - High quality PDF version
- `regional_sales_chart_TIMESTAMP.png` - Regional sales chart only
- `customer_segmentation_chart_TIMESTAMP.png` - Segmentation pie chart only  
- `churn_analysis_chart_TIMESTAMP.png` - Churn analysis chart only

### Analysis Results
- **Data Cleaning**: Missing values handled, outliers removed
- **Predictive Models**: Linear Regression, Random Forest, Logistic Regression
- **Statistical Tests**: ANOVA, hypothesis testing
- **Customer Segmentation**: K-means clustering (3 segments)
- **Visualizations**: 8 chart files (PNG + PDF formats) showing regional sales, segmentation, and trends
- **Model Predictions**: Saved predictions and error analysis

### Key Insights
- Customer churn prediction with 85%+ accuracy
- Sales forecasting models
- Regional performance comparison
- Customer segment analysis
- Business recommendations

## Troubleshooting

**Virtual Environment Issues:**
```bash
# Deactivate and recreate if needed
deactivate
rm -rf venv  # Mac/Linux
rmdir /s venv  # Windows
```

**Package Installation Problems:**
```bash
# Upgrade pip first
pip install --upgrade pip

# Then install packages
pip install pandas numpy matplotlib seaborn scikit-learn scipy openpyxl
```

**File Not Found:**
- Ensure `raw_sales_data.xlsx` is in the same folder as the Python script

## Deactivate Environment
When finished:
```bash
deactivate
```

## Requirements
- Python 3.7+
- Virtual environment
- Excel file with sales data

## Deliverables
1. ✅ Cleaned dataset (Excel)
2. ✅ Python analysis script  
3. ✅ Five visualization files (PNG + PDF formats)
4. ✅ Four additional analysis Excel files
5. ✅ Business insights report