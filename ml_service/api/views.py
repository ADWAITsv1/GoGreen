import joblib
from django.http import JsonResponse
import requests
import pandas as pd
import numpy as np
import datetime

OPENWEATHER_API_KEY = 'c9478deddab23f12cca794d50c8e7897'
TREFLE_API_KEY = 'Cdas7h3h5AfX3K6e3FTSDZhn8L7YrNvLO2QHYIO8V70'  # 友達のキー

# モデルを読み込む
model_data = joblib.load('api/plant_recommendation_model.pkl')
model = model_data['model']
plant_types = model_data['plant_types']
accuracy = model_data['accuracy']

def fetch_weather(location):
    print(f"Fetching weather for {location}...")
    url = f'http://api.openweathermap.org/data/2.5/weather?q={location}&appid={OPENWEATHER_API_KEY}&units=metric'
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Error {response.status_code}: {response.text}")
        return None
    data = response.json()
    return {
        'temperature': float(data['main']['temp']),
        'humidity': float(data['main']['humidity'])
    }

def fetch_environmental_data(location):
    print(f"Fetching environmental data for {location}...")
    weather_data = fetch_weather(location)
    if not weather_data:
        print(f"Could not fetch weather data for {location}")
        return None
    weather_data['location'] = location
    latitude_data = {
        'London': 51.5, 'New York': 40.7, 'Tokyo': 35.7, 'Sydney': -33.9,
        'Paris': 48.9, 'Berlin': 52.5, 'Los Angeles': 34.0, 'Chicago': 41.9,
        'Toronto': 43.7, 'Beijing': 39.9, 'Delhi': 28.6, 'Mumbai': 19.1, 'Jakarta': -6.2
    }
    latitude = latitude_data.get(location, 40.0)
    current_month = datetime.datetime.now().month
    sunlight_factor = 1.0 + 0.5 * np.sin((current_month - 6 if latitude > 0 else current_month) * np.pi / 6)
    weather_data['sunlight_hours'] = float(round(8.0 * sunlight_factor, 1))
    air_quality_index = int(np.random.randint(30, 150))
    weather_data['air_quality_index'] = air_quality_index
    weather_data['air_quality'] = 'Good' if air_quality_index < 50 else 'Moderate' if air_quality_index < 100 else 'Poor'
    print(f"Environmental data fetched: {weather_data}")
    return weather_data

def fetch_trefle_plants(search_term, limit=3):
    """Fetch plants from Trefle API"""
    print(f"Fetching Trefle plants for {search_term}...")
    base_url = "https://trefle.io/api/v1/plants"
    headers = {"Authorization": f"Bearer {TREFLE_API_KEY}"}
    params = {"limit": limit, "q": search_term}
    response = requests.get(base_url, headers=headers, params=params)
    if response.status_code != 200:
        print(f"Error fetching plants: {response.status_code}")
        return []
    data = response.json()
    plants = data.get('data', [])
    suggestions = [
        {"id": plant.get('id'), "name": plant.get('common_name') or plant.get('scientific_name'), "image": plant.get('image_url')}
        for plant in plants
    ]
    print(f"Trefle suggestions: {suggestions}")
    return suggestions

def get_trefle_suggestions(plant_type):
    """Get plant suggestions from Trefle API for a given plant type"""
    search_terms = {
        'Succulent': 'succulent',
        'Fern': 'fern',
        'Herb': 'herb',
        'Flowering': 'flower ornamental',
        'Tropical': 'tropical',
        'Air Purifying': 'air purifying'
    }
    search_term = search_terms.get(plant_type, plant_type)
    return fetch_trefle_plants(search_term)

def predict(request):
    try:
        location = request.GET.get('location', '')
        print(f"Received location: {location}")
        if not location:
            return JsonResponse({'error': '場所を入力してください'}, status=400)

        env_data = fetch_environmental_data(location)
        if not env_data:
            return JsonResponse({'error': f'{location}の天気データが取れませんでした'}, status=400)

        indoor_conditions = {
            'indoor_placement': 'window_sill',
            'water_freq_days': 3,
            'soil_type': 'loamy'
        }

        conditions = {**env_data, **indoor_conditions}
        input_df = pd.DataFrame([conditions])
        print(f"Input to model: {input_df.to_dict()}")

        probabilities = model.predict_proba(input_df)[0]
        top_indices = probabilities.argsort()[-3:][::-1]
        plant_recommendations = [
            {"plant_type": plant_types[i], "confidence": float(probabilities[i] * 100)}
            for i in top_indices
        ]
        print(f"Model predictions: {plant_recommendations}")

        # Trefle APIでトップの植物の提案を取得
        top_plant = plant_recommendations[0]["plant_type"]
        plant_suggestions = get_trefle_suggestions(top_plant)

        result = {
            "environmental_data": env_data,
            "indoor_conditions": indoor_conditions,
            "plant_recommendations": plant_recommendations,
            "plant_suggestions": plant_suggestions,
            "model_accuracy": round(float(accuracy * 100), 2)
        }
        return JsonResponse(result)
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return JsonResponse({'error': str(e)}, status=400)