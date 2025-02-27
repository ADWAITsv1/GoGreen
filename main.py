# main.py
import json
import argparse
from model import PlantRecommendationModel

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='AI Green Plant Recommendation System')
    parser.add_argument('--location', default='London', help='Location for weather data')
    parser.add_argument('--placement', default='window_sill', 
                       choices=['window_sill', 'near_window', 'away_from_window', 'balcony'],
                       help='Indoor placement of plants')
    parser.add_argument('--water', type=int, default=3, 
                       help='Preferred watering frequency in days')
    parser.add_argument('--soil', default='loamy',
                       choices=['sandy', 'loamy', 'clay', 'peaty', 'chalky'],
                       help='Available soil type')
    
    args = parser.parse_args()
    
    # Set up indoor conditions from arguments
    indoor_conditions = {
        'indoor_placement': args.placement,
        'water_freq_days': args.water,
        'soil_type': args.soil
    }
    
    # Initialize the model
    model = PlantRecommendationModel()
    
    # Generate recommendations
    recommendations = model.analyze_and_recommend(args.location, indoor_conditions)
    
    # Output as JSON
    json_output = model.output_json(recommendations)
    print(json_output)
    
    # Save to file
    with open('recommendation_results.json', 'w') as f:
        f.write(json_output)
    
    print(f"\nResults also saved to recommendation_results.json")

if __name__ == "__main__":
    main()