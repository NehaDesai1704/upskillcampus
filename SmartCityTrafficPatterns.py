import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load datasets
train_file = 'train_aWnotuB.csv'
test_file = 'test_BdBKkAj.csv'

df_train = pd.read_csv(train_file)
df_test = pd.read_csv(test_file)

# Data Preprocessing
def preprocess_data(df):
    if 'DateTime' in df.columns:
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        df['hour'] = df['DateTime'].dt.hour
        df['day'] = df['DateTime'].dt.day
        df['weekday'] = df['DateTime'].dt.weekday
        df['month'] = df['DateTime'].dt.month
        df['is_weekend'] = (df['weekday'] >= 5).astype(int)
        df.drop(columns=['DateTime'], inplace=True)
    else:
        print("Warning: 'DateTime' column is missing!")
    return df

df_train = preprocess_data(df_train)
df_test = preprocess_data(df_test)

# Splitting Data
X = df_train.drop(columns=['Vehicles'])
y = df_train['Vehicles']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training (Linear Regression)
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predictions
y_pred = lr_model.predict(X_val)

# Evaluation
mae = mean_absolute_error(y_val, y_pred)
mse = mean_squared_error(y_val, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_val, y_pred)

print(f'MAE: {mae}')
print(f'RMSE: {rmse}')
print(f'R² Score: {r2}')

# Visualization of Error Distribution
plt.figure(figsize=(8,5))
sns.histplot(y_val - y_pred, bins=30, kde=True, color='blue')
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.title('Distribution of Prediction Errors')
plt.show()