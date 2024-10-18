# app.py
from flask import Flask, request, jsonify
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

# Initialize the Flask application
app = Flask(__name__)

# Load the model and data preprocessing components
with open('C://Users//hp//Desktop//MyArchive//Code//Virtual_Diabetalogist//src//server_src//Diabetes_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load and prepare the dataset for label encoding and scaling
df = pd.read_csv("C:/Users/hp/Desktop/MyArchive/Code/Virtual_Diabetalogist/dataset/diabetes_prediction_dataset.csv")
label_encoder_gender = LabelEncoder()
label_encoder_smoking = LabelEncoder()

# Encode the training data
df['gender'] = label_encoder_gender.fit_transform(df['gender'])
df['smoking_history'] = label_encoder_smoking.fit_transform(df['smoking_history'])

# Prepare the StandardScaler and PCA
scaler = StandardScaler()
pca = PCA()

# Fit the scaler and PCA
X_train = df[['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']]
scaler.fit(X_train)
X_scaled = scaler.transform(X_train)
pca.fit(X_scaled)

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from the request
    data = request.get_json()
    
    # Prepare the input DataFrame
    custom_df = pd.DataFrame([data])
    
    # Encode the 'gender' and 'smoking_history' columns
    custom_df['gender'] = label_encoder_gender.transform([custom_df['gender'][0]])
    custom_df['smoking_history'] = label_encoder_smoking.transform([custom_df['smoking_history'][0]])

    # Standardize the custom data
    custom_X = scaler.transform(custom_df[['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']])
    
    # Apply PCA transformation
    custom_X_pca = pca.transform(custom_X)

    # Make predictions using the trained model
    custom_predictions = model.predict(custom_X_pca)

    # Prepare the response
    response = {
        'predictions': [
            'not predicted to have diabetes' if pred == 0 else 'predicted to have diabetes' for pred in custom_predictions
        ]
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
