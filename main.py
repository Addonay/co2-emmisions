# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% IMPORTS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# %% IMPORTING THE DATASET
df = pd.read_csv("dataset.csv")
df.head(10)

# %% INSPECTING AND UNDERSTANDING THE DATASET
df.info()
print("\nMissing Values:\n", df.isnull().sum())
df.describe()

# %% SORTING OUT MISSING VALUES
df = df.dropna(thresh=len(df) * 0.5, axis=1)
df.fillna(df.median(numeric_only=True), inplace=True)
df_cleaned = df.copy()

# %% STANDARDIZING THE DATA
numerical_cols = df.select_dtypes(include=[np.number]).columns
scaler = MinMaxScaler()
df_scaled = df.copy()
df_scaled[numerical_cols] = scaler.fit_transform(df[numerical_cols])
df_scaled.head()

# %% CHECK FOR OUTLIERS
features_to_check = ["Total CO2 Emission excluding LUCF (Mt)", "GDP PER CAPITA (USD)", "Population"]

plt.figure(figsize=(10, 6))
df[features_to_check].boxplot()
plt.title("Boxplot of Key Features (Checking for Outliers)")
plt.xticks(rotation=15)
plt.show()

# %% RE-STANDARDIZING THE DATA
numerical_cols = df.select_dtypes(include=[np.number]).columns
scaler = MinMaxScaler()
df_scaled = df.copy()
df_scaled[numerical_cols] = scaler.fit_transform(df[numerical_cols])
df_scaled.head()

# %% REMOVING OUTLIERS
def remove_outliers(df, columns):
    Q1 = df[columns].quantile(0.25)
    Q3 = df[columns].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_filtered = df[~((df[columns] < lower_bound) | (df[columns] > upper_bound)).any(axis=1)]
    return df_filtered

df_cleaned = remove_outliers(df_scaled, features_to_check)
print(f"Original dataset size: {df.shape[0]}")
print(f"Dataset size after removing outliers: {df_cleaned.shape[0]}")

# %% DISPLAYING CLEANED DATASET
print(df_cleaned)

# %% EXPORTING CLEANED DATA
# df_cleaned.to_excel("cleaned_emissions.xlsx", index=False, engine='openpyxl')
df_cleaned.to_csv("cleaned_dataset.csv", index=False)

conn = sqlite3.connect("cleaned_emissions.db")
df_cleaned.to_sql("cleaned_emissions", con=conn, if_exists="replace", index=False)
conn.close()
print("DataFrame successfully written to SQLite database!")

# %% DATA MODELLING - CO2 EMISSIONS OVER TIME
df['Year'] = pd.to_numeric(df['Year'])

plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='Year', y='Total CO2 Emission excluding LUCF (Mt)', hue='Country')
plt.title('CO2 Emissions Over Time')
plt.xlabel('Year')
plt.ylabel('Total CO2 Emission (Mt)')
plt.legend(loc='upper left', bbox_to_anchor=(1,1))
plt.show()

# %% COMPARING CO2 EMISSIONS BETWEEN TWO COUNTRIES
countries = ["Lesotho", "Madagascar"]
df_selected = df[df["Country"].isin(countries)]

plt.figure(figsize=(10, 6))
sns.lineplot(data=df_selected, x="Year", y="Total CO2 Emission excluding LUCF (Mt)", hue="Country", marker="o")
plt.title("CO2 Emissions Trend for Lesotho & Madagascar")
plt.xlabel("Year")
plt.ylabel("Total CO2 Emission (Mt)")
plt.legend(title="Country")
plt.grid(True)
plt.show()

# %% CORRELATION MATRIX
corr_matrix = df.corr(numeric_only=True)

plt.figure(figsize=(20,10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix of CO2 Emissions Data')
plt.show()

# %% MODEL TRAINING AND EVALUATION
features = ["Population", "GDP PER CAPITA (USD)", "Energy (Mt)"]
target = "Total CO2 Emission excluding LUCF (Mt)"

X = df_cleaned[features]
y = df_cleaned[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# %% ACTUAL VS PREDICTED VISUALIZATION
plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot(y_test, y_test, color='purple', linestyle='dashed', linewidth=2)
plt.xlabel("Actual CO₂ Emission")
plt.ylabel("Predicted CO₂ Emission")
plt.title("Actual vs. Predicted CO₂ Emission")
plt.show()

residuals = y_test - y_pred
plt.figure(figsize=(10,5))
sns.histplot(residuals, bins=20, kde=True, color="blue")
plt.axvline(0, color="purple", linestyle="dashed", linewidth=2)
plt.xlabel("Residuals (Error)")
plt.ylabel("Frequency")
plt.title("Residuals Distribution")
plt.show()

# %% SAVING AND LOADING MODEL
model_filename = "co2_emission_model.pkl"
joblib.dump(regressor, model_filename)
print(f"Model saved as {model_filename}")

loaded_model = joblib.load(model_filename)
new_predictions = loaded_model.predict(X_test)
print("Predictions from loaded model:", new_predictions[:5])

# %% PIE CHART COMPARISON
selected_countries = ['Kenya', 'Lesotho', 'South Africa', 'Egypt', 'Nigeria']
year_filter = 2020

df_selected = df[(df['Year'] == year_filter) & (df['Country'].isin(selected_countries))]
df_selected_sorted = df_selected.sort_values(by='Total CO2 Emission excluding LUCF (Mt)', ascending=False)

plt.figure(figsize=(11, 10))
plt.pie(df_selected_sorted['Total CO2 Emission excluding LUCF (Mt)'],
        labels=df_selected_sorted['Country'],
        autopct='%1.1f%%',
        startangle=140,
        colors=sns.color_palette("Set2", len(df_selected_sorted)))
plt.title(f"CO₂ Emission Share in {year_filter} (Selected African Countries)")
plt.axis('equal')
plt.show()

# %% BAR CHART COMPARISON
selected_countries = ['Kenya', 'South Sudan', 'Lesotho', 'Egypt', 'South Africa']
year_filter = 2020

df_selected = df[(df['Year'] == year_filter) & (df['Country'].isin(selected_countries))]
df_selected_sorted = df_selected.sort_values(by='Total CO2 Emission excluding LUCF (Mt)', ascending=False)

plt.figure(figsize=(10, 5))
sns.barplot(data=df_selected_sorted,
            x='Country',
            y='Total CO2 Emission excluding LUCF (Mt)',
            hue='Country',
            palette='rocket',
            dodge=False)
plt.title(f'Total CO₂ Emissions in {year_filter} for Selected African Countries')
plt.xlabel('Country')
plt.ylabel('Total CO₂ Emission (Mt)')
plt.tight_layout()
plt.show()
