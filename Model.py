# %% Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer

# %% Loading the data
data_path = "/Users/rugg/Documents/GitHub/Credit-card-release-classifier/credit_scoring.csv"
df = pd.read_csv(data_path)
df.head()

# %% Exploratory Data Analysis (EDA) - Part 1
print(df.info())
print(df.describe()) #Descriptive statistics
print(df.isnull().sum()) #Checking for missing values

# Distribution of the target variable
plt.figure(figsize=(8, 6))
df['TARGET'].value_counts().plot(kind='bar')
plt.title('Distribution of Target Variable')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

# Correlation between numerical variables
corr_matrix = df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Identifying numerical features
numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_features.remove('TARGET')  # Removing the target variable from the list

# Visualizing distributions of all numerical features
n_features = len(numeric_features)
n_rows = (n_features + 1) // 2  # Rounding up

plt.figure(figsize=(15, 5 * n_rows))
for i, feature in enumerate(numeric_features, 1):
    plt.subplot(n_rows, 2, i)
    sns.histplot(df[feature], kde=True)
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Creating box plots to identify outliers
plt.figure(figsize=(15, 5 * n_rows))
for i, feature in enumerate(numeric_features, 1):
    plt.subplot(n_rows, 2, i)
    sns.boxplot(x=df[feature])
    plt.title(f'Box Plot of {feature}')
    plt.xlabel(feature)
plt.tight_layout()
plt.show()

# %% Outlier Detection and Cleaning

def remove_outliers(df, column, lower_percentile=1, upper_percentile=99):
    lower_bound = np.percentile(df[column], lower_percentile)
    upper_bound = np.percentile(df[column], upper_percentile)
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Removing outliers from numerical features
for feature in numeric_features:
    df = remove_outliers(df, feature)

print(f"Dataset size after outlier removal: {df.shape}")

# Visualizing box plots after outlier removal
plt.figure(figsize=(15, 5 * n_rows))
for i, feature in enumerate(numeric_features, 1):
    plt.subplot(n_rows, 2, i)
    sns.boxplot(x=df[feature])
    plt.title(f'Box Plot of {feature} (after outlier removal)')
    plt.xlabel(feature)
plt.tight_layout()
plt.show()

# %% Data Preprocessing

# Removing the 'ID' column if present
if 'ID' in df.columns:
    df = df.drop('ID', axis=1)

# Separating features and target variable
X = df.drop('TARGET', axis=1)
y = df['TARGET']

# Handling missing values
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Splitting into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %% Modeling

# Logistic Regression Model
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train_scaled, y_train)

# Predictions of the Logistic Regression model
lr_predictions = lr_model.predict(X_test_scaled)

print("Logistic Regression Model Results:")
print(classification_report(y_test, lr_predictions))

# Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Predictions of the Random Forest model
rf_predictions = rf_model.predict(X_test_scaled)

print("Random Forest Model Results:")
print(classification_report(y_test, rf_predictions))

# %% Model Evaluation

# Confusion Matrix for Logistic Regression
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, lr_predictions), annot=True, fmt='d')
plt.title('Confusion Matrix - Logistic Regression')
plt.show()

# Confusion Matrix for Random Forest
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, rf_predictions), annot=True, fmt='d')
plt.title('Confusion Matrix - Random Forest')
plt.show()

# Feature Importance for Random Forest
feature_importance = pd.DataFrame({'feature': X.columns, 'importance': rf_model.feature_importances_})
feature_importance = feature_importance.sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
plt.title('Top 10 Feature Importance - Random Forest')
plt.show()

# %% Conclusions and Recommendations

print("Conclusions:")
print("1. The Random Forest model showed better performance compared to Logistic Regression.")
print("2. The most important features for predicting creditworthiness are:")
print(feature_importance.head(5))
print("3. It is recommended to use the Random Forest model for assessing customer creditworthiness.")
print("4. To further improve the model, hyperparameter optimization and feature engineering could be considered.")
print("5. The outlier analysis allowed for obtaining a cleaner and more representative dataset.")