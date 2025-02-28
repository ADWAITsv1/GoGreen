import React, { useState } from 'react';
import './App.css';

function App() {
  const [location, setLocation] = useState('');
  const [result, setResult] = useState(null);

  const handlePredict = async () => {
    try {
      const response = await fetch(`http://localhost:8000/api/predict/?location=${location}`);
      const data = await response.json();
      if (data.error) {
        alert('エラー: ' + data.error);
      } else {
        setResult(data);
      }
    } catch (error) {
      alert('通信エラー: ' + error.message);
    }
  };

  return (
    <div className="container">
      <h1>GoGreen!</h1>
      <input
        id="location"
        placeholder="location"
        value={location}
        onChange={(e) => setLocation(e.target.value)}
      />
      <button id="predictBtn" onClick={handlePredict}>
        Predict
      </button>
      {result && (
        <div className="result">
          <h2>Environmental Data:</h2>
          <p>Temperature: {result.environmental_data.temperature}°C</p>
          <p>Humidity: {result.environmental_data.humidity}%</p>
          <p>Location: {result.environmental_data.location}</p>
          <p>Sunlight Hours: {result.environmental_data.sunlight_hours} hours</p>
          <p>Air Quality Index: {result.environmental_data.air_quality_index}</p>
          <p>Air Quality: {result.environmental_data.air_quality}</p>

          <h2>Indoor Conditions:</h2>
          <p>Placement: {result.indoor_conditions.indoor_placement}</p>
          <p>Water Frequency: Every {result.indoor_conditions.water_freq_days} days</p>
          <p>Soil Type: {result.indoor_conditions.soil_type}</p>

          <h2>Model Accuracy:</h2>
          <p>{result.model_accuracy}%</p>

          <h2>Plant Recommendations:</h2>
          <ul>
            {result.plant_recommendations.map((rec, index) => (
              <li key={index}>
                {index + 1}. {rec.plant_type} ({rec.confidence}%)
              </li>
            ))}
          </ul>

          <h2>Plant Suggestions:</h2>
          <ul>
            {result.plant_suggestions.map((suggestion, index) => (
              <li key={suggestion.id || index}>
                {index + 1}. {suggestion.name} {suggestion.image && <img src={suggestion.image} alt={suggestion.name} style={{ width: '50px', height: '50px' }} />}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default App;