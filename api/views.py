from django.shortcuts import render

# Create your views here.
import pickle
from django.http import JsonResponse

with open('api/model.pkl', 'rb') as f:
    model = pickle.load(f)

def predict(request):
    try:
        location = request.GET.get('location', '')  
        plants_name = request.GET.get('plants_name', '')
        input_data = [[location, plants_name]]  
        result = model.predict(input_data)
        return JsonResponse({'result': str(result[0])})  
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)