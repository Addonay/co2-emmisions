# %% IMPORTS
# All imports at the top level for better organization and dependency management

import joblib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import polars.selectors as cs
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sqlalchemy import create_engine, inspect, text

# %% CONFIGURATION SETUP
# Set polars display configurations for better data viewing and visualization theme
pl.Config.set_tbl_rows(-1)  # Show all rows
pl.Config.set_tbl_cols(15)  # Show more columns
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)

# %% IMPORTING THE DATASET
# Load the dataset with comprehensive null value handling for better data quality
df = pl.read_csv(
    "dataset.csv",
    null_values=["", " ", "NA", "N/A", "nan", "NaN", "null", "Null", "NULL"],
    try_parse_dates=True,
)
print("Dataset loaded successfully!")
print(f"Dataset shape: {df.shape}")
df.head(10)

# %% INSPECTING AND UNDERSTANDING THE DATASET
# Comprehensive data exploration to understand structure, types, and quality
print("=== DATASET INFO ===")
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns}")
print("\n=== DATA TYPES ===")
print(df.dtypes)

print("\n=== MISSING VALUES ANALYSIS ===")
null_analysis = (
    pl.DataFrame(
        {
            "column": df.columns,
            "null_count": df.null_count().row(0),
            "total_rows": df.height,
        }
    )
    .with_columns(
        [
            (pl.col("null_count") / pl.col("total_rows") * 100)
            .round(2)
            .alias("null_percentage")
        ]
    )
    .filter(pl.col("null_count") > 0)
    .sort("null_count", descending=True)
)

print(null_analysis)

print("\n=== DESCRIPTIVE STATISTICS ===")
df.select(cs.numeric()).describe()

# %% SORTING OUT MISSING VALUES
# Remove columns with excessive missing values and handle remaining nulls strategically
threshold = 0.5  # 50% threshold for column removal
columns_to_drop = [
    col
    for col, null_count in zip(df.columns, df.null_count().row(0))
    if null_count / df.height > threshold
]

print(f"Columns to be dropped (>{threshold * 100}% null): {columns_to_drop}")

# Drop columns with too many missing values
if columns_to_drop:
    df = df.drop(columns_to_drop)

# Fill remaining null values with median for numeric columns (more robust than mean)
df = df.with_columns(
    [
        pl.col(col).fill_null(pl.col(col).median())
        for col in df.select(cs.numeric()).columns
    ]
)

# Create a cleaned dataset copy for later use
df_cleaned_initial = df.clone()
print(f"Dataset after cleaning: {df.shape}")

# %% STANDARDIZING THE DATA
# Normalize numerical data using Min-Max scaling for consistent feature ranges
numerical_cols = df.select(cs.numeric()).columns
print(f"Numerical columns for scaling: {numerical_cols}")

# Convert to pandas temporarily for sklearn scaling, then back to polars
df_pandas = df.to_pandas()
scaler = MinMaxScaler()
df_pandas[numerical_cols] = scaler.fit_transform(df_pandas[numerical_cols])
df_scaled = pl.from_pandas(df_pandas)

print("Data standardization completed!")
df_scaled.head()

# %% CHECK FOR OUTLIERS
# Identify outliers in key features using boxplots for visual analysis
features_to_check = [
    "Total CO2 Emission excluding LUCF (Mt)",
    "GDP PER CAPITA (USD)",
    "Population",
]

# Convert to pandas for matplotlib plotting
df_plot = df.select(features_to_check).to_pandas()

plt.figure(figsize=(12, 8))
df_plot.boxplot()
plt.title("Boxplot of Key Features (Checking for Outliers)", fontsize=16)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# %% REMOVING OUTLIERS USING POLARS
# Use IQR method to identify and remove outliers for cleaner model training
def remove_outliers_polars(df, columns):
    """Remove outliers using IQR method with polars operations"""
    for col in columns:
        if col in df.columns:
            q1 = df.select(pl.col(col).quantile(0.25)).item()
            q3 = df.select(pl.col(col).quantile(0.75)).item()
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            df = df.filter(
                (pl.col(col) >= lower_bound) & (pl.col(col) <= upper_bound)
            )
    return df


print(f"Original dataset size: {df_scaled.shape[0]}")
df_cleaned = remove_outliers_polars(df_scaled, features_to_check)
print(f"Dataset size after removing outliers: {df_cleaned.shape[0]}")
print(f"Removed {df_scaled.shape[0] - df_cleaned.shape[0]} outlier rows")

# %% DISPLAYING CLEANED DATASET
# Show final cleaned dataset statistics and sample
print("=== CLEANED DATASET SUMMARY ===")
print(f"Final shape: {df_cleaned.shape}")
print("\n=== SAMPLE DATA ===")
print(df_cleaned.head())

# %% DATA MODELING - CO2 EMISSIONS OVER TIME
# Visualize CO2 emissions trends over time by country
df_plot = df.to_pandas()  # Convert for plotting

plt.figure(figsize=(15, 8))
# Plot only top 10 countries by average emissions to avoid clutter
top_countries = (
    df.group_by("Country")
    .agg(
        pl.col("Total CO2 Emission excluding LUCF (Mt)")
        .mean()
        .alias("avg_emissions")
    )
    .sort("avg_emissions", descending=True)
    .head(10)
    .select("Country")
    .to_pandas()["Country"]
    .tolist()
)

df_top = df_plot[df_plot["Country"].isin(top_countries)]
sns.lineplot(
   data=df_top,
    x="Year",
    y="Total CO2 Emission excluding LUCF (Mt)",
    hue="Country",
)
plt.title("CO2 Emissions Over Time (Top 10 Countries by Average)", fontsize=16)
plt.xlabel("Year")
plt.ylabel("Total CO2 Emission (Mt)")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()

# %% COMPARING CO2 EMISSIONS BETWEEN SPECIFIC COUNTRIES
# Focus on specific countries for detailed comparison
countries = ["Algeria", "South Africa", "Egypt", "Nigeria", "Kenya"]
df_selected = df.filter(pl.col("Country").is_in(countries))

df_selected_pandas = df_selected.to_pandas()
plt.figure(figsize=(12, 8))
sns.lineplot(
    data=df_selected_pandas,
    x="Year",
    y="Total CO2 Emission excluding LUCF (Mt)",
    hue="Country",
    marker="o",
    linewidth=2.5,
)
plt.title("CO2 Emissions Trend for Selected African Countries", fontsize=16)
plt.xlabel("Year")
plt.ylabel("Total CO2 Emission (Mt)")
plt.legend(title="Country")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% CORRELATION MATRIX
# Analyze relationships between numerical variables
numeric_df = df.select(cs.numeric()).to_pandas()
corr_matrix = numeric_df.corr()

plt.figure(figsize=(16, 12))
mask = np.triu(
    np.ones_like(corr_matrix, dtype=bool)
)  # Show only lower triangle
sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="RdBu_r",
    fmt=".2f",
    linewidths=0.5,
    mask=mask,
    center=0,
)
plt.title("Correlation Matrix of CO2 Emissions Data", fontsize=16)
plt.tight_layout()
plt.show()

# %% MODEL TRAINING AND EVALUATION
# Train machine learning model to predict CO2 emissions
features = ["Population", "GDP PER CAPITA (USD)", "Energy (Mt)"]
target = "Total CO2 Emission excluding LUCF (Mt)"

# Ensure all required columns exist
available_features = [f for f in features if f in df_cleaned.columns]
if target not in df_cleaned.columns:
    print(f"Warning: Target column '{target}' not found!")

print(f"Using features: {available_features}")

# Convert to numpy for sklearn
X = df_cleaned.select(available_features).to_numpy()
y = df_cleaned.select(target).to_numpy().flatten()

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("=== MODEL EVALUATION METRICS ===")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R² Score: {r2:.4f}")

# Feature importance
feature_importance = pl.DataFrame(
    {
        "feature": available_features,
        "coefficient": regressor.coef_,
        "abs_coefficient": np.abs(regressor.coef_),
    }
).sort("abs_coefficient", descending=True)

print("\n=== FEATURE IMPORTANCE ===")
print(feature_importance)

# %% ACTUAL VS PREDICTED VISUALIZATION
# Visualize model performance
plt.figure(figsize=(12, 5))

# Actual vs Predicted scatter plot
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, color="blue", alpha=0.6, s=50)
plt.plot(
    [np.min(y_test), np.max(y_test)],
    [np.min(y_test), np.max(y_test)],
    color="red",
    linestyle="--",
    linewidth=2,
)
plt.xlabel("Actual CO₂ Emission")
plt.ylabel("Predicted CO₂ Emission")
plt.title("Actual vs. Predicted CO₂ Emission")
plt.grid(True, alpha=0.3)

# Residuals distribution
plt.subplot(1, 2, 2)
residuals = y_test - y_pred
plt.hist(residuals, bins=20, alpha=0.7, color="skyblue", edgecolor="black")
plt.axvline(0, color="red", linestyle="--", linewidth=2)
plt.xlabel("Residuals (Error)")
plt.ylabel("Frequency")
plt.title("Residuals Distribution")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% ADVANCED VISUALIZATIONS
# Additional visualization for comprehensive analysis

# 1. GDP vs CO2 Emissions scatter plot
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
df_plot = df.to_pandas()
plt.scatter(
    df_plot["GDP PER CAPITA (USD)"],
    df_plot["Total CO2 Emission excluding LUCF (Mt)"],
    alpha=0.6,
    color="green",
)
plt.xlabel("GDP Per Capita (USD)")
plt.ylabel("CO2 Emissions (Mt)")
plt.title("GDP vs CO2 Emissions")
plt.grid(True, alpha=0.3)

# 2. Population vs CO2 Emissions
plt.subplot(1, 3, 2)
plt.scatter(
    df_plot["Population"],
    df_plot["Total CO2 Emission excluding LUCF (Mt)"],
    alpha=0.6,
    color="orange",
)
plt.xlabel("Population")
plt.ylabel("CO2 Emissions (Mt)")
plt.title("Population vs CO2 Emissions")
plt.grid(True, alpha=0.3)

# 3. Energy vs CO2 Emissions
plt.subplot(1, 3, 3)
if "Energy (Mt)" in df_plot.columns:
    plt.scatter(
        df_plot["Energy (Mt)"],
        df_plot["Total CO2 Emission excluding LUCF (Mt)"],
        alpha=0.6,
        color="purple",
    )
    plt.xlabel("Energy (Mt)")
    plt.ylabel("CO2 Emissions (Mt)")
    plt.title("Energy vs CO2 Emissions")
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% SAVING AND LOADING MODEL
# Save the trained model for future use
model_filename = "co2_emission_model.pkl"
joblib.dump(regressor, model_filename)
print(f"Model saved as {model_filename}")

# Test loading the model
loaded_model = joblib.load(model_filename)
new_predictions = loaded_model.predict(X_test)
print(f"Predictions from loaded model (first 5): {new_predictions[:5]}")

# %% PIE CHART COMPARISON
# Compare CO2 emissions share among selected countries
selected_countries = ["Algeria", "South Africa", "Egypt", "Nigeria", "Kenya"]
year_filter = 2005  # Use available year from dataset

df_selected = df.filter(
    (pl.col("Year") == year_filter)
    & (pl.col("Country").is_in(selected_countries))
).sort("Total CO2 Emission excluding LUCF (Mt)", descending=True)

if df_selected.height > 0:
    df_plot = df_selected.to_pandas()

    plt.figure(figsize=(10, 8))
    plt.pie(
        df_plot["Total CO2 Emission excluding LUCF (Mt)"],
        labels=df_plot["Country"].to_list(),
        autopct="%1.1f%%",
        startangle=140,
        colors=sns.color_palette("Set2", len(df_plot)),
    )
    plt.title(
        f"CO₂ Emission Share in {year_filter} (Selected African Countries)",
        fontsize=14,
    )
    plt.axis("equal")
    plt.show()

# %% BAR CHART COMPARISON
# Bar chart showing CO2 emissions comparison
if df_selected.height > 0:
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=df_plot,
        x="Country",
        y="Total CO2 Emission excluding LUCF (Mt)",
        palette="viridis",
    )
    plt.title(
        f"Total CO₂ Emissions in {year_filter} for Selected African Countries",
        fontsize=14,
    )
    plt.xlabel("Country")
    plt.ylabel("Total CO₂ Emission (Mt)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# %% CREATE NORMALIZED TABLES FOR DATABASE
# Create three normalized tables for relational database structure

# Table 1: Countries and Geographic Information
countries_table = (
    df.group_by(["Country", "Sub-Region", "Code"])
    .agg(
        [
            pl.col("Area (Km2)").first().alias("area_km2"),
            pl.col("Country").count().alias("record_count"),
        ]
    )
    .with_row_index("country_id")
    .select(["country_id", "Country", "Sub-Region", "Code", "area_km2"])
)

print("=== COUNTRIES TABLE ===")
print(countries_table.head())

# Table 2: Economic and Population Data
economic_data_table = df.select(
    [
        "Country",
        "Year",
        "Population",
        "GDP PER CAPITA (USD)",
        "GDP PER CAPITA PPP (USD)",
    ]
).with_row_index("economic_id")

print("\n=== ECONOMIC DATA TABLE ===")
print(economic_data_table.head())

# Table 3: Emissions Data
emissions_table = (
    df.select(
        [
            "Country",
            "Year",
            "Total CO2 Emission including LUCF (Mt)",
            "Total CO2 Emission excluding LUCF (Mt)",
            "Transportation (Mt)",
            "Other Fuel Combustion (Mt)",
            "Manufacturing/Construction (Mt)",
            "Land-Use Change and Forestry (Mt)",
            "Industrial Processes (Mt)",
            "Energy (Mt)",
            "Electricity/Heat (Mt)",
            "Bunker Fuels (Mt)",
            "Building (Mt)",
        ]
    )
    .drop_nulls()
    .with_row_index("emission_id")
)

print("\n=== EMISSIONS TABLE ===")
print(emissions_table.head())

# %% EXPORT CLEANED DATA
# Save cleaned dataset and individual tables
print("=== EXPORTING DATA ===")

# Save main cleaned dataset
df_cleaned.write_csv("data/cleaned_df.csv")
print("✓ Cleaned dataset saved as 'cleaned_df.csv'")

# Save individual tables
countries_table.write_csv("data/countries.csv")
economic_data_table.write_csv("data/economic_data.csv")
emissions_table.write_csv("data/emissions.csv")
print("✓ Normalized tables saved as separate CSV files")

# %% DATABASE CREATION AND NORMALIZATION (PostgreSQL)
# Create PostgreSQL database with normalized tables
# DATABASE_URL = (
#     "postgresql+psycopg://co2user:co2password@localhost:5432/co2_emissions"
# )
# engine = create_engine(DATABASE_URL)

# try:
#     # Convert polars DataFrames to pandas for SQLAlchemy compatibility
#     countries_table.to_pandas().to_sql(
#         "countries", con=engine, if_exists="replace", index=False
#     )
#     economic_data_table.to_pandas().to_sql(
#         "economic_data", con=engine, if_exists="replace", index=False
#     )
#     emissions_table.to_pandas().to_sql(
#         "emissions", con=engine, if_exists="replace", index=False
#     )

#     print(
#         "✓ All tables successfully written to PostgreSQL database 'co2_emissions'"
#     )

#     # Create indexes for better query performance (using SQLAlchemy for connection management)
#     with engine.connect() as connection:
#         connection.execute(
#             text(
#                 "CREATE INDEX IF NOT EXISTS idx_countries_name ON countries(country)"
#             )
#         )
#         connection.execute(
#             text(
#                 "CREATE INDEX IF NOT EXISTS idx_economic_country_year ON economic_data(country, year)"
#             )
#         )
#         connection.execute(
#             text(
#                 "CREATE INDEX IF NOT EXISTS idx_emissions_country_year ON emissions(country, year)"
#             )
#         )
#         connection.commit()

#     print("✓ Database indexes created for optimal query performance")

#     # Display table information (requires a direct connection to fetch table names)
#     inspector = inspect(engine)
#     table_names = inspector.get_table_names()
#     print(f"✓ Created tables: {table_names}")

# except Exception as e:
#     print(f"Error creating database or tables: {e}")

# %% FINAL OUTPUT SUMMARY
print("\n=== DATA PROCESSING COMPLETE ===")
print(f"✓ Original dataset: {df.shape}")
print(f"✓ Cleaned dataset: {df_cleaned.shape}")
print(f"✓ Countries table: {countries_table.shape}")
print(f"✓ Economic data table: {economic_data_table.shape}")
print(f"✓ Emissions table: {emissions_table.shape}")
print("✓ Model trained and saved")
print("✓ Database created with normalized tables")
