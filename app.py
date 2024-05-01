from flask import Flask, jsonify, request, send_file, url_for, render_template
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import lime
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# Load data and create model
df = pd.read_csv('diabetes.csv')
predictors = df.drop("Outcome", axis=1)
target = df["Outcome"]

X_train, X_test, Y_train, Y_test = train_test_split(predictors, target, test_size=0.20, random_state=0)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train_scaled, Y_train)

@app.route('/api/predict', methods=['GET'])
def predict():
    # Retrieve and log query parameters
    feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    new_data = [float(request.args.get(name, default=0)) for name in feature_names]
    print("Received data:", new_data)

    # Transform new data with the same scaler used for training
    new_data_scaled = scaler.transform([new_data])
    print("Scaled data:", new_data_scaled)

    # Make prediction
    prediction = model.predict(new_data_scaled)[0]
    prediction_text = 'Diabetic' if prediction == 1 else 'Non-Diabetic'
    print("Prediction:", prediction_text)

    # Generate LIME explanation
    column_names = predictors.columns.tolist()
    explainer = LimeTabularExplainer(X_train_scaled, feature_names=column_names,
                                     class_names=['Non-Diabetic', 'Diabetic'], mode='classification')
    exp = explainer.explain_instance(new_data_scaled[0], model.predict_proba, num_features=len(predictors.columns), top_labels=1)
    fig, axes = plt.subplots(figsize=(12, 8))
    exp.as_pyplot_figure(label=exp.available_labels()[0])
    plt.title('Predictive Factors for Diabetes')
    plt.tight_layout()
    image_path = 'static/diabetes_lime_output.png'
    plt.savefig(image_path, dpi=300)
    plt.close(fig)

    # Return HTML response with prediction and LIME image
    html_content = f"<html><body><h1>Prediction: {prediction_text}</h1><img src='{url_for('static', filename='diabetes_lime_output.png')}' alt='LIME Explanation'></body></html>"
    return html_content

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
