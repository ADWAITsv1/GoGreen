# data_fetch.py
import requests
import pandas as pd
import os
import sys
import json

# Import configuration safely
try:
    from config import OPENWEATHER_API_KEY, TREFLE_API_KEY
except ImportError:
    # Fallback if there's a circular import
    OPENWEATHER_API_KEY = 'c9478deddab23f12cca794d50c8e7897'
    TREFLE_API_KEY = 'Cdas7h3h5AfX3K6e3FTSDZhn8L7YrNvLO2QHYIO8V70'

def fetch_weather(location):
    """ Fetch weather data from OpenWeather API """
    url = f'http://api.openweathermap.org/data/2.5/weather?q={location}&appid={OPENWEATHER_API_KEY}&units=metric'
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"Error fetching weather data: {response.status_code}")
        print(f"Response: {response.text}")
        return pd.DataFrame()
        
    data = response.json()
    weather = {
        'location': location,
        'temperature': data['main']['temp'],
        'humidity': data['main']['humidity']
    }
    return pd.DataFrame([weather])

def fetch_plants(plant_type):
    """ 
    Fetch plant data from alternative source since Trefle.io API is returning 500 errors
    This function will use a mock dataset instead
    """
    print(f"Attempting to fetch plant data for: {plant_type}")
    
    # Since Trefle.io API is giving 500 errors, we'll create a fallback mock dataset
    # This simulates what we would get from a working API
    mock_data = [
        {
            'common_name': 'Garden Rose',
            'scientific_name': 'Rosa chinensis',
            'family': 'Rosaceae',
            'water_needs': 'Medium',
            'sunlight': 'Full sun'
        },
        {
            'common_name': 'Climbing Rose',
            'scientific_name': 'Rosa setigera',
            'family': 'Rosaceae',
            'water_needs': 'Medium',
            'sunlight': 'Full sun to partial shade'
        },
        {
            'common_name': 'Rugosa Rose',
            'scientific_name': 'Rosa rugosa',
            'family': 'Rosaceae',
            'water_needs': 'Low to medium',
            'sunlight': 'Full sun'
        }
    ]
    
    # Try the original API first, but with improved error handling
    try:
        # Trying with both header and query parameter approaches
        # Method 1: Using header authentication
        headers = {
            'Authorization': f'Bearer {TREFLE_API_KEY}'
        }
        
        url = f'https://trefle.io/api/v1/plants/search?q={plant_type}'
        response = requests.get(url, headers=headers, timeout=10)
        
        print("Method 1 - Status Code:", response.status_code)
        
        # Method 2: Using query parameter (alternative method)
        if response.status_code != 200:
            url_alt = f'https://trefle.io/api/v1/plants/search?token={TREFLE_API_KEY}&q={plant_type}'
            response = requests.get(url_alt, timeout=10)
            print("Method 2 - Status Code:", response.status_code)
        
        # If any method succeeds, process the API response
        if response.status_code == 200:
            try:
                data = response.json()
                
                # Check if 'data' key exists and has content
                if 'data' in data and data['data']:
                    plants = []
                    for plant in data['data']:
                        plant_info = {
                            'common_name': plant.get('common_name', 'Unknown'),
                            'scientific_name': plant.get('scientific_name', 'Unknown')
                        }
                        plants.append(plant_info)
                    
                    return pd.DataFrame(plants)
            except json.JSONDecodeError:
                print("Error: Could not parse JSON response")
                print(f"Response text: {response.text[:100]}...")  # Print first 100 chars
    
    except Exception as e:
        print(f"Error connecting to Trefle API: {str(e)}")
    
    # If API call fails, use mock data instead
    print("Using mock plant data as fallback")
    return pd.DataFrame(mock_data)

# Example usage
if __name__ == "__main__":
    # Test weather API
    weather_df = fetch_weather("London")
    print("Weather data:")
    print(weather_df)
    
    # Test plant API
    plants_df = fetch_plants("rose")
    print("\nPlant data:")
    print(plants_df)