# building-energy-anomaly-detection
MSc Data Science Dissertation – ML Anomaly Detection (Random Forest R² = 0.9977)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt

# Load and preprocess data
df = pd.read_excel("ENB2012_data.xlsx")
new_column_names = ['Relative Compactness','Surface Area','Wall Area','Roof Area','Overall Height',
                    'Orientation','Glazing Area','Glazing Area Distribution','Heating Load','Cooling Load']
df.columns = new_column_names

# Feature engineering & scaling
features = ['Relative Compactness','Surface Area','Wall Area','Roof Area','Overall Height',
            'Orientation','Glazing Area','Glazing Area Distribution']
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

X = df.drop(['Heating Load', 'Cooling Load'], axis=1)
y = df['Heating Load']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Model (your best performer)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = sqrt(mse)
r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2:.4f} | RMSE: {rmse:.4f}")
