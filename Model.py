# Progetto: Previsione dell'affidabilità creditizia per il rilascio della carta di credito

#%% Importazione delle librerie e caricamento dei dati
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

# Caricamento dei dati
data_path = '/Users/rugg/Documents/GitHub/Credit-card-release-classifier/credit_scoring.csv'
df = pd.read_csv(data_path)
print(df.shape)
df.head()

#%% Esplorazione e pulizia dei dati
# Informazioni sul dataset
print(df.info())

# Statistiche descrittive
print(df.describe())

# Controllo dei valori nulli
print(df.isnull().sum())

# Rimozione della colonna 'ID' se presente
if 'ID' in df.columns:
    df = df.drop('ID', axis=1)

# Conversione delle colonne categoriche
categorical_columns = df.select_dtypes(include=['object']).columns
for col in categorical_columns:
    df[col] = pd.Categorical(df[col]).codes

print(df.head())

#%% Analisi esplorativa dei dati (EDA)
# Distribuzione della variabile target
plt.figure(figsize=(8, 6))
df['TARGET'].value_counts().plot(kind='bar')
plt.title('Distribuzione della variabile Target')
plt.xlabel('Classe')
plt.ylabel('Conteggio')
plt.show()

# Matrice di correlazione
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Matrice di correlazione')
plt.show()

# Boxplot per alcune variabili numeriche
numerical_columns = df.select_dtypes(include=[np.number]).columns[:5]  # Selezioniamo le prime 5 colonne numeriche
plt.figure(figsize=(12, 6))
df.boxplot(column=numerical_columns)
plt.title('Boxplot delle variabili numeriche')
plt.xticks(rotation=45)
plt.show()

#%% Preparazione dei dati per il modello
# Divisione in features (X) e target (y)
X = df.drop('TARGET', axis=1)
y = df['TARGET']

# Divisione in set di training e test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizzazione delle feature
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Dimensioni del set di training:", X_train_scaled.shape)
print("Dimensioni del set di test:", X_test_scaled.shape)

#%% Creazione e valutazione dei modelli
def evaluate_model(y_true, y_pred, model_name):
    print(f"Risultati per {model_name}:")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred):.4f}")
    print(f"F1-score: {f1_score(y_true, y_pred):.4f}")
    print("Matrice di confusione:")
    print(confusion_matrix(y_true, y_pred))
    print("\n")

# Regressione Logistica
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
evaluate_model(y_test, lr_pred, "Regressione Logistica")

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)
evaluate_model(y_test, rf_pred, "Random Forest")

#%% Interpretazione dei risultati e feature importance
# Feature importance per Random Forest
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
plt.title('Top 10 Feature Importance - Random Forest')
plt.show()

print(feature_importance.head(10))