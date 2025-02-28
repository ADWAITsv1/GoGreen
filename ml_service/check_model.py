import pickle

with open('/Users/inoseizuru/ml_service/plant_recommendation_model.pkl', 'rb') as f:
    data = pickle.load(f)
print("型:", type(data))
print("中身:", data)