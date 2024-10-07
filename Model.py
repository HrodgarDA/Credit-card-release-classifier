# %% 0. IMPORTAZIONE E CARICAMENTO DEI DATI
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# Caricamento dei dati
credit_path = '/Users/rugg/Documents/GitHub/Credit-card-release-classifier/credit_record.csv'
application_path = '/Users/rugg/Documents/GitHub/Credit-card-release-classifier/application_record.csv'
credit_df = pd.read_csv(credit_path)
application_df = pd.read_csv(application_path)

# %% 1. TARGET VARIABLE CREATION
def categorize_status(status):
    return 'non-risky' if status in ['X', 'C', '0'] else 'risky'

credit_df['TARGET'] = credit_df['STATUS'].apply(categorize_status)
target_df = credit_df.groupby('ID')['TARGET'].agg(lambda x: 'risky' if 'risky' in x.values else 'non-risky').reset_index()

dataframe = pd.merge(application_df, target_df, on='ID', how='inner')
dataframe['TARGET'] = dataframe['TARGET'].map({'risky': 1, 'non-risky': 0})

# %% 2. PULIZIA E PREPROCESSAMENTO DEI DATI
dataframe = dataframe.drop(['ID', 'FLAG_MOBIL', 'FLAG_EMAIL'], axis=1)
dataframe['AGE'] = -dataframe['DAYS_BIRTH'] / 365
dataframe['YEARS_EMPLOYED'] = -dataframe['DAYS_EMPLOYED'] / 365
dataframe['INCOME_K'] = dataframe['AMT_INCOME_TOTAL'] / 1000
dataframe = dataframe.drop(['DAYS_BIRTH', 'DAYS_EMPLOYED', 'AMT_INCOME_TOTAL'], axis=1)

# Encoding delle variabili categoriche
binary_cols = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']
for col in binary_cols:
    dataframe[col] = dataframe[col].map({'Y': 1, 'N': 0, 'F': 0, 'M': 1})

education_order = {'Lower secondary': 0, 'Secondary / secondary special': 1, 'Incomplete higher': 2, 'Higher education': 3, 'Academic degree': 4}
dataframe['NAME_EDUCATION_TYPE'] = dataframe['NAME_EDUCATION_TYPE'].map(education_order)

dataframe = pd.get_dummies(dataframe, columns=['NAME_INCOME_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE'])

# %% 3. BILANCIAMENTO DEL DATASET
X = dataframe.drop('TARGET', axis=1)
y = dataframe['TARGET']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# %% 4. NORMALIZZAZIONE
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)

# %% 5. MODELLO DI RIFERIMENTO (LOGISTIC REGRESSION)
lr = LogisticRegression(random_state=42)
lr.fit(X_train_scaled, y_train_balanced)
y_pred_lr = lr.predict(X_test_scaled)

print("Logistic Regression Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.2f}")
print(f"Precision: {precision_score(y_test, y_pred_lr):.2f}")
print(f"Recall: {recall_score(y_test, y_pred_lr):.2f}")
print(f"F1 Score: {f1_score(y_test, y_pred_lr):.2f}")

# %% 6. RANDOM FOREST
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train_balanced)
y_pred_rf = rf.predict(X_test_scaled)

print("\nRandom Forest Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.2f}")
print(f"Precision: {precision_score(y_test, y_pred_rf):.2f}")
print(f"Recall: {recall_score(y_test, y_pred_rf):.2f}")
print(f"F1 Score: {f1_score(y_test, y_pred_rf):.2f}")

# %% 7. FEATURE IMPORTANCE
feature_importance = pd.DataFrame({'feature': X.columns, 'importance': rf.feature_importances_})
feature_importance = feature_importance.sort_values('importance', ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Top 10 Most Important Features')
plt.tight_layout()
plt.show()