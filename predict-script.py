import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# Read dataset
dataset = pd.read_csv('test.csv')
drop_columns = ['id', 'InstrumentalScore', 'AcousticQuality','VocalContent','LivePerformanceLikelihood', 'AudioLoudness']
X = dataset.drop(labels=drop_columns, axis=1)

# Standardize dataset
scaler = StandardScaler()
X = scaler.fit_transform(X)

# load model
model = joblib.load('model.joblib')
y_pred = model.predict(X)

prediction = pd.DataFrame()
prediction['BeatsPerMinute'] = y_pred
prediction['id'] = dataset['id']

prediction.set_index('id', inplace=True)
prediction.to_csv('prediction.csv')