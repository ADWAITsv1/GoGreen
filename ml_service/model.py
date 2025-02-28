#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AI Green: Plant Recommendation System
Simple machine learning model for plant recommendations integrating Trefle API
"""

import pandas as pd
import numpy as np
import requests
import json
import datetime
import joblib
import os
import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Suppress warnings
warnings.filterwarnings('ignore')

# API Keys
OPENWEATHER_API_KEY = 'c9478deddab23f12cca794d50c8e7897'
TREFLE_API_KEY = 'Cdas7h3h5AfX3K6e3FTSDZhn8L7YrNvLO2QHYIO8V70'

# Data Fetching Functions
def fetch_weather(location):
    """Fetch weather data from OpenWeather API"""
    url = f'http://api.openweathermap.org/data/2.5/weather?q={location}&appid={OPENWEATHER_API_KEY}&units=metric'
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"Error fetching weather data: {response.status_code}")
        return pd.DataFrame()
        
    data = response.json()
    weather = {
        'location': location,
        'temperature': float(data['main']['temp']),
        'humidity': float(data['main']['humidity'])
    }
    return pd.DataFrame([weather])

def fetch_environmental_data(location):
    """Fetch environmental data for a given location"""
    print(f"Fetching environmental data for {location}...")
    
    # Get weather data from OpenWeather API
    weather_df = fetch_weather(location)
    
    if weather_df.empty:
        print(f"Could not fetch weather data for {location}")
        return None
    
    # Extract data from the DataFrame
    weather_data = {
        'location': location,
        'temperature': float(weather_df['temperature'].iloc[0]),
        'humidity': float(weather_df['humidity'].iloc[0])
    }
    
    # Add estimated sunlight hours based on location and season
    latitude_data = {
        'London': 51.5, 'New York': 40.7, 'Tokyo': 35.7, 'Sydney': -33.9,
        'Paris': 48.9, 'Berlin': 52.5, 'Los Angeles': 34.0, 'Chicago': 41.9,
        'Toronto': 43.7, 'Beijing': 39.9, 'Delhi': 28.6, 'Mumbai': 19.1,
        'Jakarta': -6.2
    }
    
    # Default latitude if location not found
    latitude = latitude_data.get(location, 40.0)
    
    # Estimate sunlight hours based on latitude and current month
    current_month = datetime.datetime.now().month
    
    # Northern/Southern hemisphere seasonal adjustment
    if latitude > 0:  # Northern hemisphere
        sunlight_factor = 1.0 + 0.5 * np.sin((current_month - 6) * np.pi / 6)
    else:  # Southern hemisphere
        sunlight_factor = 1.0 + 0.5 * np.sin((current_month) * np.pi / 6)
    
    # Base sunlight hours (between 4 and 12)
    base_hours = 8.0
    weather_data['sunlight_hours'] = float(round(base_hours * sunlight_factor, 1))
    
    # Estimate air quality (mock data for demo)
    air_quality_index = int(np.random.randint(30, 150))
    weather_data['air_quality_index'] = air_quality_index
    
    if air_quality_index < 50:
        air_quality = 'Good'
    elif air_quality_index < 100:
        air_quality = 'Moderate'
    else:
        air_quality = 'Poor'
    
    weather_data['air_quality'] = air_quality
    
    print(f"Environmental data fetched for {location}")
    return weather_data

def fetch_trefle_plants(search_term=None, limit=100):
    """Fetch plants from Trefle API"""
    try:
        base_url = "https://trefle.io/api/v1/plants"
        headers = {"Authorization": f"Bearer {TREFLE_API_KEY}"}
        
        params = {"limit": limit}
        if search_term:
            params["q"] = search_term
            
        response = requests.get(base_url, headers=headers, params=params)
        
        if response.status_code != 200:
            print(f"Error fetching plants: {response.status_code}")
            return []
            
        data = response.json()
        return data.get('data', [])
    except Exception as e:
        print(f"Error fetching from Trefle API: {e}")
        return []

def prepare_training_data(sample_size=500):
    """Create synthetic training data for the model"""
    print("Preparing training data...")
    
    # Feature data
    data = {
        'temperature': np.random.uniform(5, 35, sample_size).astype(float),
        'humidity': np.random.uniform(30, 90, sample_size).astype(float),
        'sunlight_hours': np.random.uniform(0, 12, sample_size).astype(float),
        'water_freq_days': np.random.choice([1, 2, 3, 7, 14], sample_size).astype(int),
        'indoor_placement': np.random.choice(['window_sill', 'near_window', 'away_from_window', 'balcony'], sample_size),
        'soil_type': np.random.choice(['sandy', 'loamy', 'clay', 'peaty', 'chalky'], sample_size),
        'air_quality': np.random.choice(['Good', 'Moderate', 'Poor'], sample_size)
    }
    
    # Plant types with optimal growing conditions
    plant_conditions = {
        'Succulent': {
            'temp_range': (10, 35),
            'humidity_range': (20, 50),
            'sunlight_range': (4, 12),
            'water_freq': [7, 14],
            'indoor_placement': ['window_sill', 'near_window'],
            'soil_type': ['sandy', 'loamy'],
            'air_quality': ['Good', 'Moderate', 'Poor']
        },
        'Fern': {
            'temp_range': (15, 25),
            'humidity_range': (60, 90),
            'sunlight_range': (0, 4),
            'water_freq': [1, 2, 3],
            'indoor_placement': ['near_window', 'away_from_window'],
            'soil_type': ['peaty', 'loamy'],
            'air_quality': ['Good', 'Moderate']
        },
        'Herb': {
            'temp_range': (15, 30),
            'humidity_range': (40, 70),
            'sunlight_range': (4, 8),
            'water_freq': [2, 3],
            'indoor_placement': ['window_sill', 'balcony'],
            'soil_type': ['loamy', 'sandy'],
            'air_quality': ['Good', 'Moderate']
        },
        'Flowering': {
            'temp_range': (15, 30),
            'humidity_range': (40, 80),
            'sunlight_range': (3, 10),
            'water_freq': [2, 3, 7],
            'indoor_placement': ['window_sill', 'near_window', 'balcony'],
            'soil_type': ['loamy', 'peaty'],
            'air_quality': ['Good', 'Moderate']
        },
        'Tropical': {
            'temp_range': (20, 35),
            'humidity_range': (60, 90),
            'sunlight_range': (2, 8),
            'water_freq': [1, 2, 3],
            'indoor_placement': ['near_window', 'away_from_window'],
            'soil_type': ['peaty', 'loamy'],
            'air_quality': ['Good']
        },
        'Air Purifying': {
            'temp_range': (15, 30),
            'humidity_range': (40, 70),
            'sunlight_range': (2, 6),
            'water_freq': [3, 7],
            'indoor_placement': ['near_window', 'away_from_window'],
            'soil_type': ['loamy', 'peaty'],
            'air_quality': ['Moderate', 'Poor']
        }
    }
    
    # Determine the best plant type for each set of conditions
    plant_types = []
    for i in range(sample_size):
        temp = data['temperature'][i]
        humidity = data['humidity'][i]
        sun = data['sunlight_hours'][i]
        water = data['water_freq_days'][i]
        placement = data['indoor_placement'][i]
        soil = data['soil_type'][i]
        air = data['air_quality'][i]
        
        scores = {}
        
        for plant, conditions in plant_conditions.items():
            score = 0
            
            # Temperature match
            if conditions['temp_range'][0] <= temp <= conditions['temp_range'][1]:
                score += 1
            
            # Humidity match
            if conditions['humidity_range'][0] <= humidity <= conditions['humidity_range'][1]:
                score += 1
            
            # Sunlight match
            if conditions['sunlight_range'][0] <= sun <= conditions['sunlight_range'][1]:
                score += 1
            
            # Water frequency match
            if water in conditions['water_freq']:
                score += 1
            
            # Placement match
            if placement in conditions['indoor_placement']:
                score += 1
            
            # Soil type match
            if soil in conditions['soil_type']:
                score += 1
                
            # Air quality match
            if air in conditions['air_quality']:
                score += 1
            
            scores[plant] = score
        
        # Select the plant type with highest score
        best_plant = max(scores, key=scores.get)
        plant_types.append(best_plant)
    
    data['plant_type'] = plant_types
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    print(f"Created training dataset with {len(df)} examples")
    
    return df

class PlantRecommendationModel:
    def __init__(self):
        """Initialize the plant recommendation model"""
        self.model = None
        self.features = None
        self.plant_types = None
        self.model_file = 'plant_recommendation_model.pkl'
        self.accuracy = 0.0
    
    def train(self, data):
        """Train the model using provided data"""
        # Split features and target
        X = data.drop('plant_type', axis=1)
        y = data['plant_type']
        
        # Store feature names for later reference
        self.features = list(X.columns)
        self.plant_types = list(y.unique())
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Define preprocessing for categorical features
        categorical_features = ['indoor_placement', 'soil_type', 'air_quality']
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        
        # Define preprocessing for numerical features
        numerical_features = ['temperature', 'humidity', 'sunlight_hours', 'water_freq_days']
        numerical_transformer = StandardScaler()
        
        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        # Create and train the pipeline with a RandomForest classifier
        self.model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        
        print("Training plant recommendation model...")
        self.model.fit(X_train, y_train)
        
        # Evaluate the model
        self.accuracy = self.model.score(X_test, y_test)
        print(f"Model accuracy: {self.accuracy:.2%}")
        
        return self.accuracy
    
    def save_model(self):
        """Save the trained model to disk"""
        if self.model is None:
            print("No trained model to save")
            return False
        
        print("Saving model to disk...")
        # Save model and metadata in a single file
        model_data = {
            'model': self.model,
            'features': self.features,
            'plant_types': self.plant_types,
            'accuracy': self.accuracy
        }
        joblib.dump(model_data, self.model_file)
        print(f"Model saved to {self.model_file}")
        return True
    
    def load_model(self):
        """Load a previously trained model from disk"""
        try:
            print("Loading model from disk...")
            model_data = joblib.load(self.model_file)
            
            # Extract model components
            self.model = model_data['model']
            self.features = model_data['features']
            self.plant_types = model_data['plant_types'] 
            self.accuracy = model_data.get('accuracy', 0.95)  # Default if not found
            
            print("Model loaded successfully")
            return True
        except FileNotFoundError:
            print("No saved model found. Please train a model first.")
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict(self, conditions):
        """
        Predict the best plant types for given conditions
        
        Args:
            conditions (dict): Dictionary containing environmental conditions
                               
        Returns:
            list: Top 3 recommended plant types with confidence scores
        """
        if self.model is None:
            print("No model loaded. Please train or load a model first.")
            return None
        
        # Create a copy of conditions to avoid modifying the original
        input_conditions = conditions.copy()
        
        # Ensure numeric values are of the correct type
        for key in ['temperature', 'humidity', 'sunlight_hours']:
            if key in input_conditions:
                input_conditions[key] = float(input_conditions[key])
        
        if 'water_freq_days' in input_conditions:
            input_conditions['water_freq_days'] = int(input_conditions['water_freq_days'])
        
        # Ensure all required features are present
        for feature in self.features:
            if feature not in input_conditions:
                if feature == 'water_freq_days':
                    # Assign a default value based on other conditions
                    if input_conditions.get('indoor_placement') == 'window_sill':
                        input_conditions[feature] = 3
                    else:
                        input_conditions[feature] = 7
                elif feature == 'soil_type':
                    # Default to loamy which is versatile
                    input_conditions[feature] = 'loamy'
                elif feature == 'air_quality':
                    # Default to Moderate if not specified
                    input_conditions[feature] = 'Moderate'
        
        # Create a DataFrame from the conditions
        input_df = pd.DataFrame([input_conditions])
        
        # Get probability predictions for each class
        probabilities = self.model.predict_proba(input_df)[0]
        
        # Get top 3 plant types with their probabilities
        top_indices = probabilities.argsort()[-3:][::-1]
        recommendations = [
            {"plant_type": self.plant_types[i], "confidence": float(probabilities[i] * 100)} 
            for i in top_indices
        ]
        
        return recommendations
    
    def get_trefle_suggestions(self, plant_type, limit=3):
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
        plants = fetch_trefle_plants(search_term=search_term, limit=limit)
        
        suggestions = []
        for plant in plants:
            suggestions.append({
                'id': plant.get('id'),
                'name': plant.get('common_name') or plant.get('scientific_name'),
                'image': plant.get('image_url')
            })
            
        return suggestions
    
    def analyze_and_recommend(self, location, indoor_conditions=None):
        """
        Generate plant recommendations based on location and conditions
        
        Args:
            location (str): City name or location
            indoor_conditions (dict, optional): Indoor growing environment
                                     
        Returns:
            dict: Recommendations with plant types and care instructions
        """
        # Initialize indoor_conditions if None
        if indoor_conditions is None:
            indoor_conditions = {}
        
        # Fetch environmental data for the location
        env_data = fetch_environmental_data(location)
        
        if not env_data:
            return {"error": f"Could not fetch environmental data for {location}"}
        
        # Combine environmental and indoor conditions
        conditions = {**env_data}
        if indoor_conditions:
            conditions.update(indoor_conditions)
        
        # Ensure we have a model
        if not self.model:
            # Try to load, or train if needed
            if not self.load_model():
                training_data = prepare_training_data(sample_size=1000)
                self.train(training_data)
                self.save_model()
        
        # Get plant recommendations
        plant_recommendations = self.predict(conditions)
        
        if not plant_recommendations:
            return {"error": "Could not generate plant recommendations"}
        
        # Get Trefle suggestions for top plant type
        top_plant = plant_recommendations[0]["plant_type"]
        plant_suggestions = self.get_trefle_suggestions(top_plant)
        
        # Prepare result
        result = {
            "environmental_data": env_data,
            "indoor_conditions": indoor_conditions,
            "plant_recommendations": plant_recommendations,
            "plant_suggestions": plant_suggestions,
            "model_accuracy": round(float(self.accuracy * 100), 2)
        }
        
        return result

def main():
    """Main function to train and save the model"""
    # Initialize model
    model = PlantRecommendationModel()
    
    # Check if model already exists
    if not os.path.exists(model.model_file):
        print("No existing model found. Creating new model...")
        # Generate training data
        training_data = prepare_training_data(sample_size=1000)
        
        # Train model with prepared data
        model.train(training_data)
        
        # Save the model to disk
        model.save_model()
    else:
        # Load existing model
        model.load_model()
    
    print("Model is ready to use with Django!")
    
    # Test with a location
    test_location = "London"
    test_conditions = {
        "indoor_placement": "window_sill",
        "water_freq_days": 3,
        "soil_type": "loamy"
    }
    
    # Generate recommendations
    recommendations = model.analyze_and_recommend(test_location, test_conditions)
    print(f"\nSample recommendations for {test_location}:")
    print(json.dumps(recommendations, indent=2))

if __name__ == "__main__":
    main()