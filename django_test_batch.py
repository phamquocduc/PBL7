import os
import django
from django.test import Client

# Set up django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'skin_prediction_project.settings')
django.setup()

client = Client()

csv_path = "/Users/hhh/workspace/school/PBL7/batch_test_meta.csv"
image_path = "/Users/hhh/workspace/school/PBL7/test_lesion.jpg"

print("Starting batch prediction test...")
with open(csv_path, 'rb') as csv_file, open(image_path, 'rb') as img_file:
    # Post files to predict-batch/ API
    response = client.post('/predict-batch/', {
        'csv_file': csv_file,
        'images': [img_file]
    })
    
    print("Status code:", response.status_code)
    import json
    data = response.json()
    print("JSON Output:")
    print(json.dumps(data, indent=2))
    
    if data.get('success') is True:
        print("✅ Batch prediction test PASSED!")
    else:
        print("❌ Batch prediction test FAILED!")
