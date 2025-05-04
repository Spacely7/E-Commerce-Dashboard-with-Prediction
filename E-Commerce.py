import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv("E-commerce Dataset .csv")
    df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d-%m-%Y', errors='coerce')
    df['Profit Margin'] = df['Profit Margin'].str.rstrip('%').astype(float)
    df['Profitability'] = df['Profit Per Order'].apply(lambda x: 'Profitable' if x > 0 else 'Loss')
    return df.dropna(subset=['Order Date'])

df = load_data()

# Sidebar filters
st.sidebar.header("Filters")
regions = st.sidebar.multiselect("Select Regions", options=df['Order Region'].unique(), default=df['Order Region'].unique())
categories = st.sidebar.multiselect("Select Categories", options=df['Category Name'].unique(), default=df['Category Name'].unique())
date_range = st.sidebar.date_input("Select Date Range", [df['Order Date'].min(), df['Order Date'].max()])

# Filtered Data
filtered_df = df[
    (df['Order Region'].isin(regions)) &
    (df['Category Name'].isin(categories)) &
    (df['Order Date'] >= pd.to_datetime(date_range[0])) &
    (df['Order Date'] <= pd.to_datetime(date_range[1]))
]

# Dashboard Title
st.title("📊 E-commerce Data Dashboard with Prediction")

# KPIs
col1, col2, col3 = st.columns(3)
col1.metric("Total Sales", f"${filtered_df['Sales'].sum():,.2f}")
col2.metric("Total Profit", f"${filtered_df['Profit Per Order'].sum():,.2f}")
col3.metric("Avg Profit Margin", f"{filtered_df['Profit Margin'].mean():.2f}%")

# Charts
st.subheader("📈 Sales & Profit Over Time")
sales_profit = filtered_df.groupby('Order Date')[['Sales', 'Profit Per Order']].sum().reset_index()
fig, ax = plt.subplots()
sns.lineplot(data=sales_profit, x='Order Date', y='Sales', label='Sales', ax=ax)
sns.lineplot(data=sales_profit, x='Order Date', y='Profit Per Order', label='Profit', ax=ax)
st.pyplot(fig)

# Category Performance
st.subheader("📦 Top Product Categories by Sales")
top_categories = filtered_df.groupby('Category Name')['Sales'].sum().sort_values(ascending=False).head(10)
st.bar_chart(top_categories)

# Regional Insights
st.subheader("🌍 Sales by Region")
region_sales = filtered_df.groupby('Order Region')['Sales'].sum().sort_values(ascending=False)
st.bar_chart(region_sales)

# Decision-Informed Insights
st.subheader("🧠 Decision Insights")

loss_orders = filtered_df[filtered_df['Profit Per Order'] < 0]
high_margin = filtered_df[filtered_df['Profit Margin'] > 50]

st.markdown("- ⚠️ **Warning:** There are **{} orders** with negative profit.".format(len(loss_orders)))
st.markdown("- 🟢 **Opportunity:** There are **{} high-margin items** (> 50%).".format(len(high_margin)))
if (filtered_df['Profit Per Order'].sum() < 0):
    st.error("❗ Net loss detected in the filtered period. Consider reevaluating product pricing or cost.")
elif (filtered_df['Profit Margin'].mean() < 10):
    st.warning("📉 Low average profit margin. Optimization recommended.")

# Extra Decision Insights
st.subheader("💡 Strategic Profitability Insights")

# Underperforming categories
st.markdown("### 🚨 Underperforming Categories")
category_perf = filtered_df.groupby('Category Name')[['Sales', 'Profit Per Order']].mean()
low_perf = category_perf[(category_perf['Sales'] < 100) & (category_perf['Profit Per Order'] < 0)]
st.dataframe(low_perf)

# Segment performance
st.markdown("### 🧑‍💼 Customer Segment Performance")
segment_perf = filtered_df.groupby('Customer Segment')[['Sales', 'Profit Per Order']].sum()
st.dataframe(segment_perf)

# Loss-prone regions
st.markdown("### 📍 Regions with High Loss %")
loss_by_region = filtered_df.groupby('Order Region')['Profit Per Order'].apply(lambda x: (x < 0).mean())
st.bar_chart(loss_by_region)

# --- PREDICTIVE MODEL SECTION ---
st.subheader("🔮 Predictive Model: Estimate Profit Per Order")

# Select relevant features
model_df = filtered_df[['Sales', 'Order Quantity', 'Category Name', 'Order Region', 'Profit Per Order']].dropna()

# Encode categorical features
categorical = ['Category Name', 'Order Region']
encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded = encoder.fit_transform(model_df[categorical])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical))

X = pd.concat([model_df[['Sales', 'Order Quantity']].reset_index(drop=True), encoded_df], axis=1)
y = model_df['Profit Per Order']

# Train/test split and model training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Metrics
st.write("**Model Performance:**")
st.write(f"R² Score: {r2_score(y_test, y_pred):.3f}")
st.write(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

# User input prediction
st.markdown("### 🔍 Try It Yourself: Predict Profit")
user_sales = st.number_input("Sales", min_value=0.0, value=100.0)
user_qty = st.number_input("Quantity Ordered", min_value=1, value=1)
user_cat = st.selectbox("Category", model_df['Category Name'].unique())
user_region = st.selectbox("Region", model_df['Order Region'].unique())

# Build user input for model
user_input = pd.DataFrame({
    'Sales': [user_sales],
    'Order Quantity': [user_qty]
})
encoded_input = encoder.transform([[user_cat, user_region]])
encoded_input_df = pd.DataFrame(encoded_input, columns=encoder.get_feature_names_out(categorical))
final_input = pd.concat([user_input, encoded_input_df], axis=1)

# Align columns (in case some categories were missing in current filters)
final_input = final_input.reindex(columns=X.columns, fill_value=0)

predicted_profit = model.predict(final_input)[0]
st.success(f"💸 Predicted Profit: ${predicted_profit:.2f}")

# Actionable Recommendations
st.subheader("✅ Actionable Recommendations")
if predicted_profit < 0:
    st.warning("Consider adjusting pricing or marketing for this combination.")
if low_perf.shape[0] > 0:
    st.info("Review underperforming categories for potential discontinuation.")
if loss_by_region.max() > 0.5:
    st.info("Investigate logistics or customer experience in high-loss regions.")

# Raw data toggle
with st.expander("🔍 Show Raw Data"):
    st.dataframe(filtered_df)
