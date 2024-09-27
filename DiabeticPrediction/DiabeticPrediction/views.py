from django.shortcuts import render
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from django.conf import settings

# Global variables for model and dataset
model = None

def load_dataset():
    """Helper function to load the dataset."""
    file_path = settings.DATASET_PATH  # Load dataset path from .env
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print(f"Dataset not found at {file_path}")
        return None

def train_model(data):
    """Train a logistic regression model and return the model."""
    X = data.drop('Outcome', axis=1)
    Y = data['Outcome']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)  # Use max_iter to avoid convergence issues
    model.fit(X_train, Y_train)
    return model

# Home view function
def home(request):
    return render(request, 'home.html')

# Result view function
def result(request):
    global model  # Use the globally declared model

    # Load dataset
    data = load_dataset()
    if data is None:
        return render(request, 'home.html', {"pred_result": "Error: Dataset not found."})

    # Train model only once if it's not already trained
    if model is None:
        model = train_model(data)

    # Get user input from request
    pregnancies = float(request.GET.get('pregnancies', 0))
    glucose = float(request.GET.get('glucose', 0))
    bloodPressure = float(request.GET.get('bloodPressure', 0))
    skinThickness = float(request.GET.get('skinThickness', 0))
    insulin = float(request.GET.get('insulin', 0))
    bmi = float(request.GET.get('bmi', 0))
    diabetesPedigreeFunction = float(request.GET.get('diabetesPedigreeFunction', 0))
    age = float(request.GET.get('age', 0))

    # Predict the outcome
    pred = model.predict([[pregnancies, glucose, bloodPressure, skinThickness, insulin, bmi, diabetesPedigreeFunction, age]])

    # Determine the result
    calc_result = "Positive" if pred == [1] else "Negative"

    # Render the result on the home page
    return render(request, 'home.html', {"pred_result": calc_result})
