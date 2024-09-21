# CELL 1

import pandas as pd
import numpy as np

# Sample data
data = {
    'timestamp': pd.date_range(start='2023-01-01', periods=100, freq='D'),
    'soil_moisture': np.random.uniform(20, 80, size=100),  # Percentage
    'temperature': np.random.uniform(10, 35, size=100),  # Celsius
    'humidity': np.random.uniform(30, 90, size=100),  # Percentage
    'equipment_hours': np.random.poisson(5, size=100),  # Hours used
    'vibration_level': np.random.uniform(0, 10, size=100),  # Arbitrary scale
    'crop_health': np.random.uniform(0, 100, size=100),  # Health score
    'maintenance_needed': np.random.choice([0, 1], size=100)  # 0: No, 1: Yes
}

df = pd.DataFrame(data)

# Display the DataFrame
print(df.head())

# CELL 2

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Prepare features and labels
X = df.drop(columns=['timestamp', 'maintenance_needed'])
y = df['maintenance_needed']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))
