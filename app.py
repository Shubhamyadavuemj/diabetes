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

# Ensure there is a folder named 'static' in the same directory as your script
if not os.path.exists('static'):
    os.makedirs('static')

# Load data and create model
import requests

# Correct URL for direct download
url = 'https://drive.google.com/uc?export=download&id=17WsunLJ_3u3NZ7zQBVRKugiAiTJMhs-Y'

# It may be necessary to stream large files to avoid loading them entirely into memory
r = requests.get(url, allow_redirects=True)
open('diabetes.csv', 'wb').write(r.content)

df = pd.read_csv('diabetes.csv')
predictors = df.drop("Outcome", axis=1)  # Assuming 'Outcome' is the target column
target = df["Outcome"]

predictors = df.drop("Outcome", axis=1)  # Assuming 'Outcome' is the target column
target = df["Outcome"]

X_train, X_test, Y_train, Y_test = train_test_split(predictors, target, test_size=0.20, random_state=0)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train_scaled, Y_train)

@app.route('/api/predict', methods=['GET'])
def predict():
    new_data = [request.args.get(f'feature{i+1}', default=0, type=float) for i in range(len(predictors.columns))]
    new_data_scaled = scaler.transform([new_data])
    prediction = model.predict(new_data_scaled)[0]
    prediction_text = 'Diabetic' if prediction == 1 else 'Non-Diabetic'

    # LIME Explanation
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

    # Create HTML content to display the prediction and the image
    html_content = f"<html><body><h1>Prediction: {prediction_text}</h1>"
    html_content += f"<img src='{url_for('static', filename='diabetes_lime_output.png')}' alt='LIME Explanation'>"
    html_content += "</body></html>"

    return html_content

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
