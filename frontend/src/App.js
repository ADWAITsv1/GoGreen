import React, { useState } from 'react';
import './App.css';  // デザイン用のCSSを後で追加

function App() {
  const [location, setLocation] = useState('');
  const [plantsName, setPlantsName] = useState('');
  const [result, setResult] = useState('');

  const handlePredict = async () => {
    try {
      const response = await fetch(
        `http://localhost:8000/api/predict/?location=${location}&plants_name=${plantsName}`
      );
      const data = await response.json();
      if (data.error) {
        alert('エラー: ' + data.error);
      } else {
        setResult(data.result);
      }
    } catch (error) {
      alert('通信エラー: ' + error.message);
    }
  };

  return (
    <div className="container">
      <h1>植物予測</h1>
      <input
        id="location"
        value={location}
        onChange={(e) => setLocation(e.target.value)}
        placeholder="場所を入力"
      />
      <input
        id="plants_name"
        value={plantsName}
        onChange={(e) => setPlantsName(e.target.value)}
        placeholder="植物名を入力"
      />
      <button onClick={handlePredict}>予測する</button>
      <p>結果: <span id="result">{result}</span></p>
    </div>
  );
}

export default App;