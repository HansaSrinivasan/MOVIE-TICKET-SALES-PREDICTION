'''import'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import numpy as np

import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA



'''ts'''

data = pd.read_csv(r"C:/Users/19295/OneDrive/Documents/4th sem/predictive analysis lab/cinema - copy.csv")

 # Handling missing values
missing_values_count = data.isnull().sum()
    
if missing_values_count.sum() == 0:
        print("No missing values found")
else:
        print("Missing values found. Handling missing values...")

        # Replace missing values with the mean of the column
        data.fillna(data.mean(numeric_only=True),inplace=True)

        # Drop rows with missing values
        #data = data.dropna()
        
        # Replace missing values with a specific value
        # data = data.fillna(0)
        
        # Replace missing values using forward fill (previous value)
        # data = data.ffill()
        
        # Replace missing values using backward fill (next value)
        # data = data.bfill()
        
        # Replace missing values using interpolation
        # data = data.interpolate()
        
        replaced_values_count = missing_values_count.sum() - data.isnull().sum().sum()
        
        if replaced_values_count == 0:
            print("No data was replaced.")
        else:
            print("Number of replaced values:", replaced_values_count)



data['DATE'] = pd.to_datetime(data['DATE'])
data['Month'] = data['DATE'].dt.month

monthly_sales = data.groupby('Month')['Total_Sales'].sum().reset_index()
print(monthly_sales)


# Extract the relevant columns
print(data.columns)
data = data.drop('DATE' , axis =1)
# Normalize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Perform PCA
pca = PCA()
pca.fit(data_scaled)

feature_names = data.columns
principal_component_names = pca.get_feature_names_out(feature_names)
principal_components = pca.components_
explained_variances = pca.explained_variance_ratio_

# Determine the number of principal components to retain
num_components = [i for i, num in enumerate(explained_variances) if num > 0.04]

reduced_column = [data.columns[i] for i in range(len(data.columns)) if i in num_components]
data_reduced = data[reduced_column]
print(data_reduced.head())
#Drop columns
data = data_reduced


# Fit ARIMA model
training = monthly_sales['Total_Sales']
model = ARIMA(training, order=(1, 0, 0))
model_fit = model.fit()
forecast = model_fit.forecast(steps=5)
print(forecast)

# Plot the actual and forecasted values
plt.figure(figsize=(10, 6))
plt.plot(monthly_sales['Month'],monthly_sales['Total_Sales'] , label='Actual')
plt.plot([11,12,13,14 , 15],forecast , label='Forecast')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.legend()
plt.show()




''"classification"""

# Drop any rows with missing values
data.dropna(inplace=True)

print(data.columns)
target = data['Film Code']
features = data[['Total_Sales' , 'Tickets Sold' , 'Tickets Out' , 'Show Time' , 'Occupancy Percentage']]

le = LabelEncoder()
for column in features:
    if features[column].dtype == 'object':
        features[column] = le.fit_transform(features[column])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Create a random forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:")
print(report)




"""PCA"""









