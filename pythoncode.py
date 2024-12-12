# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



# %%
df = pd.read_csv('E:\\INFOSYS SPRINGBOARD AI\\RESOURCES\\pregdata.csv', encoding='ISO-8859-1')


# %%
print(df.info())
print(df.describe())

# %%
print(df.head())

# %%
print(df.columns)

# %%
plt.scatter(df['Age'], df['BMI(kg/m 2)'])
plt.xlabel('Age')
plt.ylabel('BMI (kg/m^2)')
plt.title('Age vs BMI Scatter Plot')
plt.show()

# %%
import numpy as np

numeric_cols = df.select_dtypes(include=[np.number]).columns

plt.figure(figsize=(15, 10))

for i, col in enumerate(numeric_cols):
    if col != 'Patient ID':  
        plt.subplot(len(numeric_cols)//3 + 1, 3, i)
        sns.boxplot(df[col])
        plt.title(f'Boxplot for {col}')

plt.tight_layout()
plt.show()


# %%
numeric_cols = df.select_dtypes(include=[np.number]).columns

plt.figure(figsize=(15, 12))


for i, col in enumerate(numeric_cols, -1):
    if col != 'Age'and col !='Patient ID': 
        plt.subplot(3, 3, i)
        plt.scatter(df['Age'], df[col])
        plt.title(f'Age vs {col}')
        plt.xlabel('Age')
        plt.ylabel(col)
        plt.tight_layout()

# %%

# Strip trailing spaces from column names
df.columns = df.columns.str.strip()

# Step 1: Remove rows where Age > 80 and Body Temperature(F) < 80
df_cleaned = df[(df['Age'] <= 80) & (df['Body Temperature(F)'] >= 80)]

# Step 2: Select numeric columns (excluding Patient ID)
numeric_cols = [col for col in df_cleaned.columns if df_cleaned[col].dtype in [np.int64, np.float64] and col != 'Patient ID']

# Step 3: Calculate Z-scores and remove rows with outliers (Z-score > 3 or < -3)
for col in numeric_cols:
    z_scores = (df_cleaned[col] - df_cleaned[col].mean()) / df_cleaned[col].std()
    df_cleaned = df_cleaned[(z_scores <= 3) & (z_scores >= -3)]  # Filter out rows with outliers

# Step 4: Apply Z-score normalization to the remaining data
for col in numeric_cols:
    df_cleaned[col] = (df_cleaned[col] - df_cleaned[col].mean()) / df_cleaned[col].std()

# Get the number of rows after cleaning and outlier removal
total_rows = df_cleaned.shape[0]
print(f'Total number of rows after cleaning and outlier removal: {total_rows}')

# Optional: Visualize the cleaned data with boxplots (after outlier removal and normalization)
plt.figure(figsize=(15, 10))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot((len(numeric_cols) + 2) // 3, 3, i)
    sns.boxplot(data=df_cleaned[col])
    plt.title(f'Boxplot after Outlier Removal & Normalization: {col}')
plt.tight_layout()
plt.show()


# %%
df_cleaned = df[(df['Age'] <= 80) & (df['Body Temperature(F)'] >= 80)]


# %%

total_rows = df_cleaned.shape[0]
print(f'Total number of rows after cleaning and outlier removal: {total_rows}')

# %%
numeric_cols = [col for col in df_cleaned.columns if df_cleaned[col].dtype in [np.int64, np.float64] and col != 'Patient ID']



# %%
for col in numeric_cols:
    z_scores = (df_cleaned[col] - df_cleaned[col].mean()) / df_cleaned[col].std()
    df_cleaned = df_cleaned[(z_scores <= 3) & (z_scores >= -3)]  

# %%
for col in numeric_cols:
    df_cleaned[col] = (df_cleaned[col] - df_cleaned[col].mean()) / df_cleaned[col].std()

# %%
total_rows = df_cleaned.shape[0]
print(f'Total number of rows after cleaning and outlier removal: {total_rows}')


# %%
plt.figure(figsize=(15, 10))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot((len(numeric_cols) + 2) // 3, 3, i)
    sns.boxplot(data=df_cleaned[col])
    plt.title(f'Boxplot for {col}')

plt.tight_layout()
plt.show()


# %%



# %%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

# Split data into features (X) and target (y)
X = df_cleaned[numeric_cols].drop('Outcome', axis=1, errors='ignore')  # Replace 'Outcome' with your actual target column name
y = df_cleaned['Outcome']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))


# %%
from sklearn.neighbors import KNeighborsClassifier

# Train the KNN model
knn_model = KNeighborsClassifier(n_neighbors=5)  # You can tune n_neighbors as needed
knn_model.fit(X_train, y_train)

# Predict on test set
y_pred_knn = knn_model.predict(X_test)

# Evaluate the model
print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))


# %%
from sklearn.svm import SVC

# Train the SVM model
svm_model = SVC(kernel='linear', random_state=42)  # You can try other kernels like 'rbf', 'poly', etc.
svm_model.fit(X_train, y_train)

# Predict on test set
y_pred_svm = svm_model.predict(X_test)

# Evaluate the model
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))


# %%
from sklearn.ensemble import RandomForestClassifier

# Train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)  # You can tune n_estimators as needed
rf_model.fit(X_train, y_train)

# Predict on test set
y_pred_rf = rf_model.predict(X_test)

# Evaluate the model
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))


# %%
# Select the first 10 rows for prediction
sample_X = X_test.iloc[:10]  # Features
sample_y = y_test.iloc[:10]  # Actual labels
print('knn predictions \n')
# Predict using the trained KNN model
sample_pred = knn_model.predict(sample_X)

# Reset the index of sample_X to show the correct row positions (from 0 to 9)
sample_X_reset = sample_X.reset_index(drop=True)

# Create a DataFrame for the results, showing the reset index and predicted values for the first 10 rows
prediction_results = pd.DataFrame({
    'Index': sample_X_reset.index,
    'Predicted': sample_pred
})

# Display the table with index and predicted values for the first 10 rows
print(prediction_results.to_string(index=False))


# %%
# Select the first 10 rows from X_test
sample_X = X_test.iloc[:10]  # Features
print('logistic regression \n')

# Predict using the trained Logistic Regression model for the first 10 rows
sample_pred_lr = model.predict(sample_X)

# Reset the index of sample_X to show the correct row positions (from 0 to 9)
sample_X_reset = sample_X.reset_index(drop=True)

# Create a DataFrame for the results, showing the reset index and predicted values for the first 10 rows
prediction_results_lr = pd.DataFrame({
    'Index': sample_X_reset.index,
    'Predicted': sample_pred_lr
})

# Display the table with index and predicted values for the first 10 rows
print(prediction_results_lr.to_string(index=False))


# %%
# Select the first 10 rows from X_test
sample_X = X_test.iloc[:10]  # Features
print("svm predictions\n")
# Predict using the trained SVM model for the first 10 rows
sample_pred_svm = svm_model.predict(sample_X)

# Reset the index of sample_X to show the correct row positions (from 0 to 9)
sample_X_reset = sample_X.reset_index(drop=True)

# Create a DataFrame for the results, showing the reset index and predicted values for the first 10 rows
prediction_results_svm = pd.DataFrame({
    'Index': sample_X_reset.index,
    'Predicted': sample_pred_svm
})

# Display the table with index and predicted values for the first 10 rows
print(prediction_results_svm.to_string(index=False))


# %%
# Select the first 10 rows from X_test
sample_X = X_test.iloc[:10]  # Features
print("random forest predictions\n")
# Predict using the trained Random Forest model for the first 10 rows
sample_pred_rf = rf_model.predict(sample_X)

# Reset the index of sample_X to show the correct row positions (from 0 to 9)
sample_X_reset = sample_X.reset_index(drop=True)

# Create a DataFrame for the results, showing the reset index and predicted values for the first 10 rows
prediction_results_rf = pd.DataFrame({
    'Index': sample_X_reset.index,
    'Predicted': sample_pred_rf
})

# Display the table with index and predicted values for the first 10 rows
print(prediction_results_rf.to_string(index=False))


# %%
import joblib

# Save the trained model
joblib.dump(rf_model, 'random_forest_model.pkl')



