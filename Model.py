# %% Caricamento dei dati

#libraries
from IPython.display import display
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel

# Caricamento del dataset
data_path = '/Users/rugg/Documents/GitHub/Credit-card-release-classifier/credit_scoring.csv'
df = pd.read_csv(data_path)
df = df.drop(['ID'], axis=1)
display(df.head())
print(f'DATAFRAME INFO:')
display(df.info())

# %% Exploratory Data Analysis (EDA)

display(f'NULL VALUES COUNT BY COLUMN: \n{df.isnull().sum()}') # Check for missing values
print('____________________________________________________')
display(df.describe()) # Display summary statistics
df.dropna(inplace=True)
print('____________________________________________________')
display(f'DATAFRAME SIZE IS: {df.size}')

# Grouping features by data type
numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = df.select_dtypes(include=['object']).columns.tolist()

# Identify binary categorical features
binary_features = [col for col in categorical_features if df[col].nunique() == 2]
multi_categorical_features = [col for col in categorical_features if df[col].nunique() > 2]
print(f'Binary Features are: {binary_features}, \nMulti categorical Features are: {multi_categorical_features}')

# %% FEATURE ENGINEERING
df['DAYS_BIRTH'] = round(abs(df['DAYS_BIRTH']),0)
df['DAYS_EMPLOYED'] = round(abs(df['DAYS_EMPLOYED']),0)
df['AGE_YEARS'] = round(df['DAYS_BIRTH'] / 365,0)
df['YEARS_EMPLOYED'] = round(df['DAYS_EMPLOYED'] / 365,0)

df = df.drop(['DAYS_BIRTH', 'DAYS_EMPLOYED'], axis=1) # Drop original columns
df.head()

# Visualize the distribution of the target variable
plt.figure(figsize=(8, 6))
df['TARGET'].value_counts().plot(kind='bar')
plt.title('Distribution of Target Variable')
plt.xlabel('Target')
plt.ylabel('Count')
plt.show()

# Features distribution visualization
def plot_bar_chart(feature, df):
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(x=feature, data=df)
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    for i in ax.containers:
        ax.bar_label(i, label_type='edge')
    plt.tight_layout()
    plt.show()
    
#Columns visualization
for column in df[multi_categorical_features]:
    plot_bar_chart(column, df)
    
for column in df[binary_features]:
    plot_bar_chart(column, df)
    
# %% #ENCODING
df_encoded = df.copy()

le = LabelEncoder()
for feature in binary_features:
    df_encoded[feature] = le.fit_transform(df_encoded[feature])

#Hierical categorical variables encoding
ordinal_mapping = { 'Separated': 0, 'Single / not married': 1, 'Married': 2, 'Civil marriage': 2, 'Widow': 3}
# Applica l'encoding usando .map()
df_encoded['NAME_FAMILY_STATUS'] = df_encoded['NAME_FAMILY_STATUS'].map(ordinal_mapping)
print(df_encoded['NAME_FAMILY_STATUS'].value_counts())

categorical_columns = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
                       'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
                       'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE']

# Perform one-hot encoding
df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True, prefix=categorical_columns)
df_encoded.head(10)