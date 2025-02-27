# interactive_demo.py
import json
import os
from model import PlantRecommendationModel

def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """Print the application header"""
    print("\n" + "=" * 60)
    print("                   AI GREEN PLANT ASSISTANT")
    print("     Modern Agriculture × AI - Personalized Plant Recommendations")
    print("=" * 60 + "\n")

def get_user_input():
    """Get user input for location and growing conditions"""
    print("Please provide information about your growing environment:\n")
    
    # Get location
    location = input("Enter your location (city name): ").strip()
    if not location:
        location = "London"
        print(f"Using default location: {location}")
    
    # Get indoor placement
    print("\nWhere will you place your plants?")
    print("1. Window sill (direct sunlight)")
    print("2. Near window (bright indirect light)")
    print("3. Away from window (low light)")
    print("4. Balcony (outdoor conditions)")
    
    placement_choice = input("Choose placement (1-4): ").strip()
    placement_mapping = {
        '1': 'window_sill',
        '2': 'near_window',
        '3': 'away_from_window',
        '4': 'balcony'
    }
    
    placement = placement_mapping.get(placement_choice, 'near_window')
    
    # Get watering preference
    print("\nHow often do you prefer to water plants?")
    print("1. Daily")
    print("2. Every 2-3 days")
    print("3. Weekly")
    print("4. Every 2 weeks")
    
    water_choice = input("Choose watering frequency (1-4): ").strip()
    water_mapping = {
        '1': 1,
        '2': 3,
        '3': 7,
        '4': 14
    }
    
    water_freq = water_mapping.get(water_choice, 3)
    
    # Get soil type
    print("\nWhat type of soil do you have available?")
    print("1. Sandy (well-draining)")
    print("2. Loamy (balanced)")
    print("3. Clay (water-retaining)")
    print("4. Peaty (acidic)")
    print("5. Chalky (alkaline)")
    
    soil_choice = input("Choose soil type (1-5): ").strip()
    soil_mapping = {
        '1': 'sandy',
        '2': 'loamy',
        '3': 'clay',
        '4': 'peaty',
        '5': 'chalky'
    }
    
    soil_type = soil_mapping.get(soil_choice, 'loamy')
    
    # Return all user inputs
    return {
        'location': location,
        'indoor_conditions': {
            'indoor_placement': placement,
            'water_freq_days': water_freq,
            'soil_type': soil_type
        }
    }

def display_results(results):
    """Display the recommendation results in a user-friendly format"""
    print("\n" + "=" * 60)
    print("                   PLANT RECOMMENDATIONS")
    print("=" * 60 + "\n")
    
    # Environmental analysis
    env = results['environmental_analysis']
    print(f"ENVIRONMENTAL CONDITIONS FOR {env['location'].upper()}:")
    print(f"  Temperature: {env['temperature']}°C")
    print(f"  Humidity: {env['humidity']}%")
    print(f"  Estimated Sunlight: {env['sunlight_hours']} hours per day")
    print(f"  Air Quality: {env['air_quality']} (AQI: {env['air_quality_index']})")
    
    # Indoor conditions
    indoor = results['indoor_conditions']
    print("\nINDOOR GROWING CONDITIONS:")
    print(f"  Placement: {indoor['indoor_placement'].replace('_', ' ').title()}")
    print(f"  Watering Frequency: Every {indoor['water_freq_days']} days")
    print(f"  Soil Type: {indoor['soil_type'].title()}")
    
    # Model accuracy
    print(f"\nMODEL ACCURACY: {results['model_accuracy']}%")
    
    # Plant recommendations
    print("\nRECOMMENDED PLANTS:")
    for i, rec in enumerate(results['plant_recommendations'], 1):
        print(f"  {i}. {rec['plant_type']} - {rec['confidence']:.1f}% match")
    
    # Care instructions for top recommendation
    print("\nCARE INSTRUCTIONS FOR TOP RECOMMENDATION:")
    care = results['care_instructions']
    print(f"  Watering: {care['watering']}")
    print(f"  Light: {care['light']}")
    print(f"  Soil: {care['soil']}")
    print(f"  Fertilizing: {care['fertilizing']}")
    
    print("\nGROWING TIMELINE:")
    for time in care['timeline']:
        print(f"  Month {time['month']}: {time['action']}")
    
    print("\nResults saved to recommendation_results.json")
    print("=" * 60)


def main():
    """Main function to run the interactive demo"""
    clear_screen()
    print_header()
    
    # Initialize the model
    model = PlantRecommendationModel()
    
    # Get user input
    user_input = get_user_input()
    
    print("\nFetching environmental data and analyzing conditions...")
    print("Training AI model with your specifications...")
    
    # Generate recommendations
    recommendations = model.analyze_and_recommend(
        user_input['location'], 
        user_input['indoor_conditions']
    )
    
    # Save to file
    with open('recommendation_results.json', 'w') as f:
        json.dump(recommendations, f, indent=2)
    
    # Display results
    clear_screen()
    display_results(recommendations)
    
    print("\nThank you for using AI Green Plant Assistant!")
    print("Press Enter to exit...")
    input()


if __name__ == "__main__":
    main()