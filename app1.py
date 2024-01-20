from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
mymodel = joblib.load('Croprecc.pkl')

fertilizer_recommendations = {
    'apple': 'Balanced NPK (Nitrogen, Phosphorus, Potassium) fertilizer with added micronutrients.',
    'banana': 'High-potassium fertilizer with additional phosphorus and magnesium.',
    'blackgram': 'NPK (Nitrogen, Phosphorus, Potassium) fertilizer with additional phosphorus.',
    'chickpea': 'Balanced NPK (Nitrogen, Phosphorus, Potassium) fertilizer with added sulfur and zinc.',
    'coconut': 'Balanced NPK (Nitrogen, Phosphorus, Potassium) fertilizer with added magnesium.',
    'coffee': 'Balanced NPK (Nitrogen, Phosphorus, Potassium) fertilizer with added magnesium and micronutrients.',
    'cotton': 'High-nitrogen fertilizer with additional phosphorus and potassium.',
    'grapes': 'Balanced NPK (Nitrogen, Phosphorus, Potassium) fertilizer with added magnesium and micronutrients.',
    'jute': 'Balanced NPK (Nitrogen, Phosphorus, Potassium) fertilizer with added sulfur.',
    'kidneybeans': 'Balanced NPK (Nitrogen, Phosphorus, Potassium) fertilizer with added sulfur and zinc.',
    'lentil': 'Balanced NPK (Nitrogen, Phosphorus, Potassium) fertilizer with added sulfur and zinc.',
    'maize': 'Balanced NPK (Nitrogen, Phosphorus, Potassium) fertilizer with additional phosphorus.',
    'mango': 'Balanced NPK (Nitrogen, Phosphorus, Potassium) fertilizer with added magnesium and micronutrients.',
    'mothbeans': 'Balanced NPK (Nitrogen, Phosphorus, Potassium) fertilizer with added sulfur and zinc.',
    'mungbean': 'Balanced NPK (Nitrogen, Phosphorus, Potassium) fertilizer with added sulfur and zinc.',
    'muskmelon': 'Balanced NPK (Nitrogen, Phosphorus, Potassium) fertilizer with added magnesium and micronutrients.',
    'orange': 'Balanced NPK (Nitrogen, Phosphorus, Potassium) fertilizer with added magnesium and micronutrients.',
    'papaya': 'Balanced NPK (Nitrogen, Phosphorus, Potassium) fertilizer with added magnesium and micronutrients.',
    'pigeonpeas': 'Balanced NPK (Nitrogen, Phosphorus, Potassium) fertilizer with added sulfur and zinc.',
    'pomegranate': 'Balanced NPK (Nitrogen, Phosphorus, Potassium) fertilizer with added magnesium and micronutrients.',
    'rice': 'Balanced NPK (Nitrogen, Phosphorus, Potassium) fertilizer with added sulfur and zinc.',
    'watermelon': 'Balanced NPK (Nitrogen, Phosphorus, Potassium) fertilizer with added magnesium and micronutrients.',
}

crop_images = {
    'apple': 'apple.png',
    'banana': 'banana.png',
    'blackgram': 'blackgram.png',
    'chickpea': 'chickpea.png',
    'coconut': 'coconut.png',
    'coffee': 'coffee.png',
    'cotton': 'cotton.png',
    'grapes': 'grapes.png',
    'jute': 'jute.png',
    'kidneybeans': 'kidneybeans.png',
    'lentil': 'lentil.png',
    'maize': 'maize.png',
    'mango': 'mango.png',
    'mothbeans': 'mothbeans.png',
    'mungbean': 'mungbean.png',
    'muskmelon': 'muskmelon.png',
    'orange': 'orange.png',
    'papaya': 'papaya.png',
    'pigeonpeas': 'pigeonpeas.png',
    'pomegranate': 'pomegranate.png',
    'rice': 'rice.png',
    'watermelon': 'watermelon.png',
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve form data from the request
    n = int(request.form.get('n'))
    p = int(request.form.get('p'))
    k = int(request.form.get('k'))
    temp = float(request.form.get('temp'))
    humidity = float(request.form.get('humidity'))
    ph = float(request.form.get('ph'))
    rainfall = float(request.form.get('rainfall'))

    # Make predictions using your model
    prediction = mymodel.predict([[n, p, k, temp, humidity, ph, rainfall]])[0]

    # Further processing based on the prediction
    recommended_fertilizer = fertilizer_recommendations.get(prediction, 'No recommendation available')
    crop_image = crop_images.get(prediction, 'default_crop_image.png')

    return render_template('index.html', prediction=prediction, fertilizer=recommended_fertilizer, crop_image=crop_image)

if __name__ == '__main__':
    app.run(debug=True)
