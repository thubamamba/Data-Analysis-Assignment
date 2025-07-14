"""
Week 3 Project: Advanced Data Analysis Techniques and Business Insights
Complete Python Solution for Real Excel Sales Data Analysis

This script reads the actual raw_sales_data.xlsx file and performs all required analyses.
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, confusion_matrix
from scipy.stats import f_oneway, ttest_ind, zscore
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('default')
sns.set_palette("husl")

print("=== WEEK 3 PROJECT: ADVANCED DATA ANALYSIS ===")
print("Loading data from raw_sales_data.xlsx...")

# Create output folder if it doesn't exist
output_folder = "output"
os.makedirs(output_folder, exist_ok=True)

# Generate timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 1. LOAD THE ACTUAL EXCEL FILE
try:
    # Read the Excel file - adjust sheet name if needed
    df = pd.read_excel('raw_sales_data.xlsx', sheet_name=0)
    print(f"✓ Data loaded successfully!")
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

except Exception as e:
    print(f"Error loading Excel file: {e}")
    print("Please ensure 'raw_sales_data.xlsx' is in the same directory")

# Display initial data exploration
print("\n=== INITIAL DATA EXPLORATION ===")
print("First 5 rows:")
print(df.head())
print(f"\nData types:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nBasic statistics:\n{df.describe()}")

# Check for duplicates
duplicates = df.duplicated().sum()
print(f"\nDuplicate rows: {duplicates}")

print("\n" + "="*60)
print("1. DATA PREPROCESSING AND CLEANING")
print("="*60)

# Create a copy for cleaning
df_clean = df.copy()

# Handle missing values
print("Handling missing values...")
# Fill missing numerical values with mean/median
numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
for col in numerical_cols:
    if df_clean[col].isnull().sum() > 0:
        if df_clean[col].skew() > 1:
            df_clean[col].fillna(df_clean[col].median(), inplace=True)
            print(f"  - Filled {col} missing values with median")
        else:
            df_clean[col].fillna(df_clean[col].mean(), inplace=True)
            print(f"  - Filled {col} missing values with mean")

# Fill missing categorical values with mode
categorical_cols = df_clean.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if df_clean[col].isnull().sum() > 0:
        mode_value = df_clean[col].mode()
        if len(mode_value) > 0:
            df_clean[col].fillna(mode_value[0], inplace=True)
            print(f"  - Filled {col} missing values with mode: {mode_value[0]}")

# Remove outliers using Z-score method
print("\nRemoving outliers using Z-score method...")
initial_rows = len(df_clean)

for col in numerical_cols:
    if col in df_clean.columns and col != 'Customer_ID':
        z_scores = np.abs(zscore(df_clean[col]))
        df_clean = df_clean[z_scores < 3]

outliers_removed = initial_rows - len(df_clean)
print(f"Removed {outliers_removed} outlier rows")

# Standardize categorical variables
print("Standardizing categorical variables...")
for col in categorical_cols:
    if col in df_clean.columns and col not in ['Customer_ID', 'Customer_Name']:
        df_clean[col] = df_clean[col].astype(str).str.strip()
        if 'region' in col.lower():
            df_clean[col] = df_clean[col].str.title()

print(f"✓ Data cleaning completed. Final shape: {df_clean.shape}")

print("\n" + "="*60)
print("2. PREDICTIVE MODELING FOR SALES FORECASTING")
print("="*60)

# Prepare features for modeling
# Convert Churned to binary
le = LabelEncoder()
df_clean['Churned_Binary'] = le.fit_transform(df_clean['Churned'])

# Linear Regression: Predict Total_Spend based on Marketing_Spend and Seasonality_Index
print("\n2.1 Linear Regression for Sales Prediction:")
print("-" * 40)

# Prepare features and target
X_lr = df_clean[['Marketing_Spend', 'Seasonality_Index']].copy()
y_lr = df_clean['Total_Spend'].copy()

# Split data
X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(X_lr, y_lr, test_size=0.3, random_state=42)

# Train Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train_lr, y_train_lr)

# Make predictions
y_pred_lr = lr_model.predict(X_test_lr)

# Calculate metrics
mse_lr = mean_squared_error(y_test_lr, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)
r2_lr = lr_model.score(X_test_lr, y_test_lr)

print(f"Linear Regression Results:")
print(f"  RMSE: {rmse_lr:.2f}")
print(f"  R² Score: {r2_lr:.3f}")
print(f"  Coefficients: Marketing_Spend={lr_model.coef_[0]:.3f}, Seasonality_Index={lr_model.coef_[1]:.3f}")
print(f"  Intercept: {lr_model.intercept_:.2f}")

# Logistic Regression: Predict customer churn
print("\n2.2 Logistic Regression for Customer Churn:")
print("-" * 40)

# Prepare features for churn prediction
X_churn = df_clean[['Total_Spend', 'Purchase_Frequency', 'Marketing_Spend']].copy()
y_churn = df_clean['Churned_Binary'].copy()

# Split data
X_train_churn, X_test_churn, y_train_churn, y_test_churn = train_test_split(
    X_churn, y_churn, test_size=0.3, random_state=42, stratify=y_churn
)

# Train Logistic Regression model
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_churn, y_train_churn)

# Make predictions
y_pred_churn = log_reg.predict(X_test_churn)
churn_accuracy = accuracy_score(y_test_churn, y_pred_churn)

print(f"Churn Prediction Results:")
print(f"  Accuracy: {churn_accuracy:.3f}")
print(f"  Classification Report:")
print(classification_report(y_test_churn, y_pred_churn, target_names=['Not Churned', 'Churned']))

# Random Forest for enhanced prediction
print("\n2.3 Random Forest for Enhanced Sales Prediction:")
print("-" * 45)

# Train Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_lr, y_train_lr)

# Make predictions
y_pred_rf = rf_model.predict(X_test_lr)
rmse_rf = np.sqrt(mean_squared_error(y_test_lr, y_pred_rf))
r2_rf = rf_model.score(X_test_lr, y_test_lr)

print(f"Random Forest Results:")
print(f"  RMSE: {rmse_rf:.2f}")
print(f"  R² Score: {r2_rf:.3f}")

# Feature importance
feature_importance = dict(zip(['Marketing_Spend', 'Seasonality_Index'], rf_model.feature_importances_))
print(f"  Feature Importance: {feature_importance}")

print("\n" + "="*60)
print("3. STATISTICAL ANALYSIS FOR BUSINESS INSIGHTS")
print("="*60)

# ANOVA: Compare Total_Spend across regions
print("\n3.1 ANOVA: Sales Performance Across Regions")
print("-" * 42)

# Group data by region
regions = df_clean['Region'].unique()
region_groups = [df_clean[df_clean['Region'] == region]['Total_Spend'] for region in regions]

# Perform ANOVA
f_stat, p_value_anova = f_oneway(*region_groups)

print(f"ANOVA Results:")
print(f"  F-statistic: {f_stat:.4f}")
print(f"  P-value: {p_value_anova:.4f}")

if p_value_anova < 0.05:
    print("  ✓ Significant difference in sales across regions (p < 0.05)")
else:
    print("  ✗ No significant difference in sales across regions (p >= 0.05)")

# Regional statistics
regional_stats = df_clean.groupby('Region')['Total_Spend'].agg(['mean', 'std', 'count']).round(2)
print(f"\nRegional Sales Statistics:")
print(regional_stats)

# Hypothesis Testing: Impact of high marketing spend on sales
print("\n3.2 Hypothesis Testing: Marketing Spend Impact")
print("-" * 44)

# Split into high and low marketing spend groups
marketing_median = df_clean['Marketing_Spend'].median()
high_marketing = df_clean[df_clean['Marketing_Spend'] > marketing_median]['Total_Spend']
low_marketing = df_clean[df_clean['Marketing_Spend'] <= marketing_median]['Total_Spend']

# Perform t-test
t_stat, p_value_ttest = ttest_ind(high_marketing, low_marketing)

print(f"T-test Results (High vs Low Marketing Spend):")
print(f"  T-statistic: {t_stat:.4f}")
print(f"  P-value: {p_value_ttest:.4f}")
print(f"  High marketing group mean sales: ${high_marketing.mean():.2f}")
print(f"  Low marketing group mean sales: ${low_marketing.mean():.2f}")

if p_value_ttest < 0.05:
    print("  ✓ Marketing spend has significant impact on sales (p < 0.05)")
else:
    print("  ✗ No significant impact of marketing spend on sales (p >= 0.05)")

print("\n" + "="*60)
print("4. MACHINE LEARNING FOR CUSTOMER SEGMENTATION")
print("="*60)

# K-Means Clustering
print("\n4.1 K-Means Customer Segmentation")
print("-" * 34)

# Prepare features for clustering
cluster_features = ['Total_Spend', 'Purchase_Frequency']
X_cluster = df_clean[cluster_features].copy()

# Standardize features
scaler = StandardScaler()
X_cluster_scaled = scaler.fit_transform(X_cluster)

# Perform K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_cluster_scaled)

# Add cluster labels to dataframe
df_clean['Customer_Segment'] = clusters

print(f"Customer Segmentation Results:")
print(f"  Number of clusters: 3")

# Analyze clusters
cluster_summary = df_clean.groupby('Customer_Segment')[cluster_features].mean().round(2)
print(f"\nCluster Centers (original scale):")
print(cluster_summary)

# Cluster sizes
cluster_sizes = df_clean['Customer_Segment'].value_counts().sort_index()
print(f"\nCluster sizes:")
for i, size in enumerate(cluster_sizes):
    print(f"  Segment {i}: {size} customers ({size/len(df_clean)*100:.1f}%)")

# Cluster characteristics
print(f"\nCluster Characteristics:")
for segment in sorted(df_clean['Customer_Segment'].unique()):
    segment_data = df_clean[df_clean['Customer_Segment'] == segment]
    avg_spend = segment_data['Total_Spend'].mean()
    avg_freq = segment_data['Purchase_Frequency'].mean()
    churn_rate = (segment_data['Churned'] == 'Yes').mean() * 100

    if avg_spend >= 5000:
        segment_type = "High-Value"
    elif avg_spend >= 3500:
        segment_type = "Medium-Value"
    else:
        segment_type = "Low-Value"

    print(f"  Segment {segment} ({segment_type}): Avg Spend=${avg_spend:.0f}, Avg Frequency={avg_freq:.1f}, Churn Rate={churn_rate:.1f}%")

print("\n" + "="*60)
print("5. CREATING VISUALIZATIONS")
print("="*60)

# Create the three required visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Sales Data Analysis - Key Insights', fontsize=16, fontweight='bold')

# 1. Sales by Region (ANOVA visualization)
regional_means = df_clean.groupby('Region')['Total_Spend'].mean()
bars1 = axes[0, 0].bar(regional_means.index, regional_means.values, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
axes[0, 0].set_title('Average Sales by Region (ANOVA Results)', fontweight='bold')
axes[0, 0].set_xlabel('Region')
axes[0, 0].set_ylabel('Average Total Spend ($)')
axes[0, 0].tick_params(axis='x', rotation=45)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 50,
                    f'${height:.0f}', ha='center', va='bottom', fontweight='bold')

# 2. Customer Segmentation (Pie Chart)
segment_counts = df_clean['Customer_Segment'].value_counts().sort_index()
segment_labels = [f'Segment {i}\n({cluster_summary.loc[i, "Total_Spend"]:.0f}$ avg)'
                 for i in segment_counts.index]
colors = ['#ff9999', '#66b3ff', '#99ff99']
wedges, texts, autotexts = axes[0, 1].pie(segment_counts.values, labels=segment_labels,
                                         autopct='%1.1f%%', startangle=90, colors=colors)
axes[0, 1].set_title('Customer Segmentation Distribution', fontweight='bold')

# 3. Marketing Spend vs Total Spend (Linear Regression)
axes[1, 0].scatter(df_clean['Marketing_Spend'], df_clean['Total_Spend'],
                   c=df_clean['Customer_Segment'], cmap='viridis', alpha=0.7, s=60)
axes[1, 0].set_xlabel('Marketing Spend ($)')
axes[1, 0].set_ylabel('Total Spend ($)')
axes[1, 0].set_title('Marketing Spend vs Total Spend by Segment', fontweight='bold')

# Add regression line
x_line = np.linspace(df_clean['Marketing_Spend'].min(), df_clean['Marketing_Spend'].max(), 100)
y_line = lr_model.predict(np.column_stack([x_line, np.ones(100) * df_clean['Seasonality_Index'].mean()]))
axes[1, 0].plot(x_line, y_line, 'r--', linewidth=2, alpha=0.8, label='Regression Line')
axes[1, 0].legend()

# 4. Churn Rate by Region
churn_by_region = df_clean.groupby('Region')['Churned'].apply(lambda x: (x == 'Yes').mean() * 100)
bars4 = axes[1, 1].bar(churn_by_region.index, churn_by_region.values,
                       color=['lightcoral' if x > 50 else 'lightgreen' for x in churn_by_region.values])
axes[1, 1].set_title('Customer Churn Rate by Region', fontweight='bold')
axes[1, 1].set_xlabel('Region')
axes[1, 1].set_ylabel('Churn Rate (%)')
axes[1, 1].tick_params(axis='x', rotation=45)

# Add value labels on bars
for bar in bars4:
    height = bar.get_height()
    axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

# Create the three required visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Sales Data Analysis - Key Insights', fontsize=16, fontweight='bold')

# 1. Sales by Region (ANOVA visualization)
regional_means = df_clean.groupby('Region')['Total_Spend'].mean()
bars1 = axes[0, 0].bar(regional_means.index, regional_means.values, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
axes[0, 0].set_title('Average Sales by Region (ANOVA Results)', fontweight='bold')
axes[0, 0].set_xlabel('Region')
axes[0, 0].set_ylabel('Average Total Spend ($)')
axes[0, 0].tick_params(axis='x', rotation=45)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 50,
                    f'${height:.0f}', ha='center', va='bottom', fontweight='bold')

# 2. Customer Segmentation (Pie Chart)
segment_counts = df_clean['Customer_Segment'].value_counts().sort_index()
segment_labels = [f'Segment {i}\n({cluster_summary.loc[i, "Total_Spend"]:.0f}$ avg)'
                 for i in segment_counts.index]
colors = ['#ff9999', '#66b3ff', '#99ff99']
wedges, texts, autotexts = axes[0, 1].pie(segment_counts.values, labels=segment_labels,
                                         autopct='%1.1f%%', startangle=90, colors=colors)
axes[0, 1].set_title('Customer Segmentation Distribution', fontweight='bold')

# 3. Marketing Spend vs Total Spend (Linear Regression)
axes[1, 0].scatter(df_clean['Marketing_Spend'], df_clean['Total_Spend'],
                   c=df_clean['Customer_Segment'], cmap='viridis', alpha=0.7, s=60)
axes[1, 0].set_xlabel('Marketing Spend ($)')
axes[1, 0].set_ylabel('Total Spend ($)')
axes[1, 0].set_title('Marketing Spend vs Total Spend by Segment', fontweight='bold')

# Add regression line
x_line = np.linspace(df_clean['Marketing_Spend'].min(), df_clean['Marketing_Spend'].max(), 100)
y_line = lr_model.predict(np.column_stack([x_line, np.ones(100) * df_clean['Seasonality_Index'].mean()]))
axes[1, 0].plot(x_line, y_line, 'r--', linewidth=2, alpha=0.8, label='Regression Line')
axes[1, 0].legend()

# 4. Churn Rate by Region
churn_by_region = df_clean.groupby('Region')['Churned'].apply(lambda x: (x == 'Yes').mean() * 100)
bars4 = axes[1, 1].bar(churn_by_region.index, churn_by_region.values,
                       color=['lightcoral' if x > 50 else 'lightgreen' for x in churn_by_region.values])
axes[1, 1].set_title('Customer Churn Rate by Region', fontweight='bold')
axes[1, 1].set_xlabel('Region')
axes[1, 1].set_ylabel('Churn Rate (%)')
axes[1, 1].tick_params(axis='x', rotation=45)

# Add value labels on bars
for bar in bars4:
    height = bar.get_height()
    axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()

# Save the figure to output folder with timestamp
print("\n📊 Saving visualization charts...")
viz_filename = f"sales_analysis_charts_{timestamp}.png"
viz_filepath = os.path.join(output_folder, viz_filename)

try:
    plt.savefig(viz_filepath, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Charts saved: {viz_filepath}")
except Exception as e:
    print(f"❌ Error saving charts: {e}")

# Also save as PDF for high quality
pdf_filename = f"sales_analysis_charts_{timestamp}.pdf"
pdf_filepath = os.path.join(output_folder, pdf_filename)

try:
    plt.savefig(pdf_filepath, bbox_inches='tight', facecolor='white')
    print(f"✅ PDF charts saved: {pdf_filepath}")
except Exception as e:
    print(f"❌ Error saving PDF: {e}")

# Display the figure
plt.show()

# Create individual charts for better quality/clarity
print("\n📈 Saving individual charts...")

# Individual Chart 1: Regional Sales
plt.figure(figsize=(10, 6))
bars = plt.bar(regional_means.index, regional_means.values,
               color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
plt.title('Average Sales by Region (ANOVA Results)', fontsize=14, fontweight='bold')
plt.xlabel('Region')
plt.ylabel('Average Total Spend ($)')
plt.xticks(rotation=45)

# Add value labels
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 50,
             f'${height:.0f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
individual_chart1 = f"regional_sales_chart_{timestamp}.png"
chart1_path = os.path.join(output_folder, individual_chart1)
plt.savefig(chart1_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ Regional chart saved: {chart1_path}")
plt.close()

# Individual Chart 2: Customer Segmentation
plt.figure(figsize=(8, 8))
colors = ['#ff9999', '#66b3ff', '#99ff99']
wedges, texts, autotexts = plt.pie(segment_counts.values, labels=segment_labels,
                                  autopct='%1.1f%%', startangle=90, colors=colors)
plt.title('Customer Segmentation Distribution', fontsize=14, fontweight='bold')

individual_chart2 = f"customer_segmentation_chart_{timestamp}.png"
chart2_path = os.path.join(output_folder, individual_chart2)
plt.savefig(chart2_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ Segmentation chart saved: {chart2_path}")
plt.close()

# Individual Chart 3: Churn Analysis
plt.figure(figsize=(10, 6))
bars = plt.bar(churn_by_region.index, churn_by_region.values,
               color=['lightcoral' if x > 50 else 'lightgreen' for x in churn_by_region.values])
plt.title('Customer Churn Rate by Region', fontsize=14, fontweight='bold')
plt.xlabel('Region')
plt.ylabel('Churn Rate (%)')
plt.xticks(rotation=45)

# Add value labels
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
individual_chart3 = f"churn_analysis_chart_{timestamp}.png"
chart3_path = os.path.join(output_folder, individual_chart3)
plt.savefig(chart3_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ Churn chart saved: {chart3_path}")
plt.close()

print("\n" + "="*60)
print("6. BUSINESS INSIGHTS & RECOMMENDATIONS")
print("="*60)

print("\n📊 KEY FINDINGS:")
print("-" * 15)

# Model Performance Insights
better_model = "Random Forest" if rmse_rf < rmse_lr else "Linear Regression"
print(f"1. {better_model} performs better for sales prediction (RMSE: {min(rmse_rf, rmse_lr):.2f})")

# Regional Analysis
best_region = regional_means.idxmax()
worst_region = regional_means.idxmin()
print(f"2. {best_region} region has highest average sales (${regional_means[best_region]:.0f})")
print(f"3. {worst_region} region has lowest average sales (${regional_means[worst_region]:.0f})")

# Customer Segmentation Insights
for segment in sorted(df_clean['Customer_Segment'].unique()):
    segment_data = df_clean[df_clean['Customer_Segment'] == segment]
    size = len(segment_data)
    avg_spend = segment_data['Total_Spend'].mean()
    if segment == 0:
        print(f"4. Customer Segment {segment}: {size} customers ({size/len(df_clean)*100:.1f}%) - Average spend: ${avg_spend:.0f}")

# Churn Analysis
overall_churn = (df_clean['Churned'] == 'Yes').mean() * 100
highest_churn_region = churn_by_region.idxmax()
print(f"5. Overall churn rate: {overall_churn:.1f}% (highest in {highest_churn_region}: {churn_by_region[highest_churn_region]:.1f}%)")

# Statistical Significance
if p_value_anova < 0.05:
    print(f"6. Regional differences in sales are statistically significant (p={p_value_anova:.4f})")
if p_value_ttest < 0.05:
    print(f"7. Marketing spend significantly impacts sales (p={p_value_ttest:.4f})")

print(f"8. Customer churn prediction accuracy: {churn_accuracy:.1%}")

print("\n🎯 RECOMMENDED BUSINESS ACTIONS:")
print("-" * 32)

recommendations = [
    f"Focus marketing investment on {best_region} region success factors",
    f"Develop retention strategies for {highest_churn_region} region (highest churn: {churn_by_region[highest_churn_region]:.1f}%)",
    "Implement personalized marketing for each customer segment",
    f"Use {better_model} model for sales forecasting and budget planning",
    "Increase marketing spend for high-value customer segments",
    "Deploy churn prediction model for proactive customer retention",
    f"Investigate why {worst_region} region underperforms and develop improvement strategies",
    "Create targeted promotions based on seasonality patterns"
]

for i, rec in enumerate(recommendations, 1):
    print(f"{i}. {rec}")

print("\n" + "="*60)
print("7. MODEL PERFORMANCE SUMMARY")
print("="*60)

print(f"\n📈 PREDICTIVE MODEL RESULTS:")
print(f"  • Linear Regression RMSE: {rmse_lr:.2f} (R² = {r2_lr:.3f})")
print(f"  • Random Forest RMSE: {rmse_rf:.2f} (R² = {r2_rf:.3f})")
print(f"  • Churn Prediction Accuracy: {churn_accuracy:.1%}")
print(f"  • Best Performing Model: {better_model}")

print(f"\n📊 STATISTICAL TEST RESULTS:")
print(f"  • ANOVA (Regional Sales): F={f_stat:.3f}, p={p_value_anova:.4f}")
print(f"  • T-test (Marketing Impact): t={t_stat:.3f}, p={p_value_ttest:.4f}")

print(f"\n🎯 SEGMENTATION RESULTS:")
for segment in sorted(df_clean['Customer_Segment'].unique()):
    segment_data = df_clean[df_clean['Customer_Segment'] == segment]
    print(f"  • Segment {segment}: {len(segment_data)} customers, "
          f"Avg Spend=${segment_data['Total_Spend'].mean():.0f}, "
          f"Churn Rate={(segment_data['Churned']=='Yes').mean()*100:.1f}%")

print("\n" + "="*60)
print("✅ ANALYSIS COMPLETED SUCCESSFULLY!")
print("="*60)

print(f"\nFinal dataset shape: {df_clean.shape}")
print(f"Models trained: Linear Regression, Random Forest, Logistic Regression, K-Means")
print(f"Visualizations created: 3 charts showing regional analysis, segmentation, and trends")
print(f"Business insights generated: {len(recommendations)} actionable recommendations")

# Create output folder and save all analysis files
print("\n" + "="*60)
print("SAVING ALL ANALYSIS OUTPUTS")
print("="*60)

def save_with_timestamp(dataframe, base_filename, folder="output"):
    """Save DataFrame to Excel with timestamp in specified folder"""
    filename = f"{base_filename}_{timestamp}.xlsx"
    file_path = os.path.join(folder, filename)

    try:
        dataframe.to_excel(file_path, index=False)
        print(f"✅ Saved: {file_path}")
        return file_path
    except Exception as e:
        print(f"❌ Error saving {filename}: {e}")
        return None

# 1. Save cleaned dataset
print("\n1. Saving cleaned dataset...")
save_with_timestamp(df_clean, "cleaned_sales_data")

# 2. Save customer segments summary
print("\n2. Saving customer segmentation analysis...")
if 'Customer_Segment' in df_clean.columns:
    segment_summary = df_clean.groupby('Customer_Segment').agg({
        'Total_Spend': ['mean', 'sum', 'count'],
        'Purchase_Frequency': 'mean',
        'Marketing_Spend': 'mean',
        'Churned': lambda x: (x == 'Yes').mean() * 100  # Churn rate percentage
    }).round(2)

    # Flatten column names
    segment_summary.columns = ['_'.join(col).strip() for col in segment_summary.columns]
    segment_summary.reset_index(inplace=True)

    save_with_timestamp(segment_summary, "customer_segments")
else:
    print("❌ Customer segmentation not available")

# 3. Save regional analysis
print("\n3. Saving regional analysis...")
regional_analysis = df_clean.groupby('Region').agg({
    'Total_Spend': ['mean', 'median', 'std', 'count'],
    'Purchase_Frequency': ['mean', 'median'],
    'Marketing_Spend': ['mean', 'sum'],
    'Churned': lambda x: (x == 'Yes').mean() * 100  # Churn rate percentage
}).round(2)

# Flatten column names
regional_analysis.columns = ['_'.join(col).strip() for col in regional_analysis.columns]
regional_analysis.reset_index(inplace=True)
save_with_timestamp(regional_analysis, "regional_analysis")

# 4. Save analysis summary with key metrics
print("\n4. Saving analysis summary...")
summary_data = {
    'Metric': [
        'Total Customers',
        'Average Total Spend',
        'Overall Churn Rate (%)',
        'Linear Regression RMSE',
        'Random Forest RMSE',
        'Churn Prediction Accuracy',
        'Number of Customer Segments',
        'Best Performing Region',
        'Worst Performing Region',
        'ANOVA P-Value',
        'Marketing Impact P-Value'
    ],
    'Value': [
        len(df_clean),
        f"${df_clean['Total_Spend'].mean():.2f}",
        f"{(df_clean['Churned'] == 'Yes').mean() * 100:.1f}%",
        f"{rmse_lr:.2f}" if 'rmse_lr' in locals() else 'N/A',
        f"{rmse_rf:.2f}" if 'rmse_rf' in locals() else 'N/A',
        f"{churn_accuracy:.1%}" if 'churn_accuracy' in locals() else 'N/A',
        df_clean['Customer_Segment'].nunique() if 'Customer_Segment' in df_clean.columns else 'N/A',
        df_clean.groupby('Region')['Total_Spend'].mean().idxmax(),
        df_clean.groupby('Region')['Total_Spend'].mean().idxmin(),
        f"{p_value_anova:.4f}" if 'p_value_anova' in locals() else 'N/A',
        f"{p_value_ttest:.4f}" if 'p_value_ttest' in locals() else 'N/A'
    ]
}

summary_df = pd.DataFrame(summary_data)
save_with_timestamp(summary_df, "analysis_summary")

# 5. Save model predictions (if available)
print("\n5. Saving model predictions...")
if 'y_pred_lr' in locals() and 'X_test_lr' in locals():
    predictions_df = pd.DataFrame({
        'Actual_Sales': y_test_lr.values,
        'Predicted_Sales_LR': y_pred_lr,
        'Predicted_Sales_RF': y_pred_rf if 'y_pred_rf' in locals() else None,
        'Marketing_Spend': X_test_lr['Marketing_Spend'].values,
        'Seasonality_Index': X_test_lr['Seasonality_Index'].values,
        'Prediction_Error_LR': y_test_lr.values - y_pred_lr,
        'Prediction_Error_RF': y_test_lr.values - y_pred_rf if 'y_pred_rf' in locals() else None
    })
    save_with_timestamp(predictions_df, "model_predictions")
else:
    print("❌ Model predictions not available")

print(f"\n📂 All files saved to: {os.path.abspath(output_folder)}")
print(f"🕒 Timestamp used: {timestamp}")
print(f"📝 File naming format: filename_{timestamp}.xlsx")
print(f"📊 Chart naming format: chartname_{timestamp}.png")

print(f"\n📋 COMPLETE FILE LIST:")
print(f"📊 Excel Files:")
print(f"  • cleaned_sales_data_{timestamp}.xlsx")
print(f"  • customer_segments_{timestamp}.xlsx")
print(f"  • regional_analysis_{timestamp}.xlsx")
print(f"  • analysis_summary_{timestamp}.xlsx")
print(f"  • model_predictions_{timestamp}.xlsx")

print(f"📈 Visualization Files:")
print(f"  • sales_analysis_charts_{timestamp}.png (all 4 charts combined)")
print(f"  • sales_analysis_charts_{timestamp}.pdf (high quality PDF)")
print(f"  • regional_sales_chart_{timestamp}.png (individual chart)")
print(f"  • customer_segmentation_chart_{timestamp}.png (individual chart)")
print(f"  • churn_analysis_chart_{timestamp}.png (individual chart)")

print(f"\n📋 DELIVERABLES READY:")
print(f"1. ✅ Cleaned dataset (df_clean)")
print(f"2. ✅ Python scripts for predictive modeling and statistical analysis")
print(f"3. ✅ Three visualizations (regional sales, customer segmentation, churn analysis)")
print(f"4. ✅ Summary report with key findings and business recommendations")
