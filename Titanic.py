
# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Display settings
pd.set_option('display.max_columns', None)
sns.set_style("whitegrid")

# 2. Load Dataset (download 'train.csv' from Kaggle Titanic competition)
df = pd.read_csv("train.csv")

# 3. Basic Data Info
print("\n--- Dataset Info ---")
print(df.info())
print("\n--- First 5 Rows ---")
print(df.head())
print("\n--- Statistical Summary ---")
print(df.describe(include='all'))

# 4. Missing Values
print("\n--- Missing Values ---")
print(df.isnull().sum())

# Visualize Missing Data
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
plt.title("Missing Values Heatmap")
plt.show()

# 5. Value Counts for Categorical Columns
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    print(f"\nValue Counts for {col}:")
    print(df[col].value_counts())

# 6. Univariate Analysis
plt.figure(figsize=(6,4))
sns.countplot(x='Survived', data=df, palette='pastel')
plt.title("Survival Count")
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(df['Age'], kde=True, bins=30, color='skyblue')
plt.title("Age Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(x='Pclass', y='Age', data=df, palette='Set2')
plt.title("Age Distribution by Passenger Class")
plt.show()

# 7. Bivariate Analysis
plt.figure(figsize=(6,4))
sns.countplot(x='Pclass', hue='Survived', data=df, palette='husl')
plt.title("Survival Count by Passenger Class")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x='Sex', hue='Survived', data=df, palette='coolwarm')
plt.title("Survival Count by Gender")
plt.show()

# 8. Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()


sns.pairplot(df.dropna(), hue='Survived', diag_kind='kde', palette='husl')
plt.show()

print("\n--- Observations ---")
print("""
1. Females had a higher survival rate compared to males.
2. Passengers in 1st class had a much higher survival rate.
3. Younger passengers tended to survive more often.
4. There are missing values in Age, Cabin, and Embarked columns.
5. Fare is positively correlated with survival.
""")
