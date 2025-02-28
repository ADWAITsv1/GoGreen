# AI Green: Plant Recommendation System

AI Green is an intelligent plant recommendation system that suggests suitable plants based on your location and indoor growing conditions.

## Overview

This project combines machine learning with real-time environmental data to recommend the most suitable plants for a given location. The system analyzes various factors including:

- Local temperature and humidity (via OpenWeather API)
- Estimated sunlight hours based on location and season
- Indoor placement preferences (window sill, near window, etc.)
- Watering frequency preferences
- Soil type preferences
- Air quality considerations

## Key Features

- **Location-based recommendations**: Get plant suggestions tailored to your local climate
- **Machine learning model**: Uses Random Forest classifier to match environmental conditions to optimal plant types
- **Real plant data**: Integrates with Trefle API to provide real plant suggestions
- **Environmental analysis**: Provides detailed analysis of your location's growing conditions
- **Care instructions**: Includes watering, light, soil, and fertilizing guidelines for recommended plants

## Technical Architecture

The project consists of:

1. **Django Backend**: Handles API requests and serves the web interface
2. **Machine Learning Model**: Pre-trained on plant growing conditions and environmental factors
3. **External API Integration**: 
   - OpenWeather API for real-time climate data
   - Trefle API for botanical information
4. **Simple Frontend**: Clean, user-friendly interface for entering location and preferences

## How It Works

1. User enters their location and optional indoor growing preferences
2. The system fetches real-time environmental data for that location
3. The trained ML model predicts which plant types would thrive in those conditions
4. The system retrieves specific plant suggestions from botanical data
5. Results are presented to the user along with care instructions

## Plant Categories

The system can recommend plants from several categories:
- Succulents
- Ferns
- Herbs
- Flowering plants
- Tropical plants
- Air purifying plants

## Setup and Installation

### Prerequisites
- Python 3.10+
- Django 5.1+
- scikit-learn
- pandas
- numpy
- requests

### Installation
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the development server: `python manage.py runserver`
4. Access the web interface at http://127.0.0.1:8000/

## API Usage

The system provides a RESTful API endpoint at `/api/recommend/` that accepts POST requests with the following parameters:

- `location` (required): City name (e.g., "London")
- `indoor_placement` (optional): Where the plant will be placed ("window_sill", "near_window", "away_from_window", "balcony")
- `water_freq_days` (optional): How often you can water plants (in days)
- `soil_type` (optional): Preferred soil type ("loamy", "sandy", "clay", "peaty", "chalky")

## Technologies Used

- **Backend**:
  - Django 5.1: Web framework for the API and server-side logic
  - Python 3.10: Core programming language
  - scikit-learn: Machine learning library for plant recommendation model
  - joblib: For model serialization and persistence

- **Data Processing**:
  - pandas: Data manipulation and analysis
  - numpy: Numerical computations
  - requests: HTTP library for API interactions

- **External APIs**:
  - OpenWeather API: For real-time climate data
  - Trefle API: For botanical information and plant data

- **Frontend**:
  - HTML/CSS: Simple, clean user interface
  - Vanilla JavaScript: For API interactions

## Machine Learning Details

The recommendation system uses a Random Forest Classifier model trained on plant-environment compatibility data. The model:

- Has an accuracy of approximately 73%
- Considers 7 environmental factors
- Maps conditions to 6 distinct plant categories
- Provides confidence scores for recommendations

### Feature Importance

The model prioritizes these features (from most to least important):
1. Temperature
2. Humidity
3. Sunlight hours
4. Indoor placement
5. Soil type
6. Water frequency
7. Air quality

## Project Structure

```
AI_Green/
├── api/                     # Django app for the REST API
│   ├── migrations/
│   ├── __init__.py
│   ├── admin.py
│   ├── apps.py
│   ├── models.py
│   ├── tests.py
│   ├── urls.py
│   └── views.py
├── css/                     # CSS styles
│   └── style.css
├── frontend/                # Frontend files
│   └── public/
│       ├── index.html
│       └── ...
├── ml_service/             # Django project settings
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── model.py                # Main ML model implementation
├── plant_recommendation_model.pkl  # Serialized ML model
├── index.html              # Main page
├── manage.py
├── requirements.txt
└── README.md
```

## Running the Project

1. Ensure you have all dependencies installed:
   ```
   pip install -r requirements.txt
   ```

2. Start the Django development server:
   ```
   python manage.py runserver
   ```

3. Open a web browser and navigate to http://127.0.0.1:8000/

4. Enter your location and any preferences, then click "predict" to get plant recommendations

## Example API Response

```json
{
  "environmental_data": {
    "location": "London",
    "temperature": 3.59,
    "humidity": 81.0,
    "sunlight_hours": 4.5,
    "air_quality_index": 69,
    "air_quality": "Moderate"
  },
  "indoor_conditions": {
    "indoor_placement": "window_sill",
    "water_freq_days": 3,
    "soil_type": "loamy"
  },
  "plant_recommendations": [
    {
      "plant_type": "Succulent",
      "confidence": 31.0
    },
    {
      "plant_type": "Tropical",
      "confidence": 30.0
    },
    {
      "plant_type": "Flowering",
      "confidence": 21.0
    }
  ],
  "plant_suggestions": [
    {
      "id": 77116,
      "name": "Evergreen oak",
      "image": "https://d2seqvvyy3b8p2.cloudfront.net/40ab8e7cdddbe3e78a581b84efa4e893.jpg"
    }
  ],
  "model_accuracy": 73.0
}
```

## Future Improvements

- Enhanced user profiles to save preferences
- More detailed care instructions with seasonal adjustments
- Integration with plant disease identification
- Mobile app with geolocation support
- Community features to share growing experiences

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Credits

- OpenWeather API for environmental data
- Trefle API for botanical information
- Plant care guidelines from botanical research
