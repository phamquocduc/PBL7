import requests
import os

url = "http://localhost:8080/predict/"
image_path = "/Users/hhh/workspace/school/PBL7/test_lesion.jpg"

if not os.path.exists(image_path):
    print(f"❌ Error: Test image not found at {image_path}")
    exit(1)

print(f"Sending test image {image_path} to {url}...")

with open(image_path, 'rb') as f:
    files = {'image': f}
    data = {
        'age': 45,
        'sex': 'Male',
        'localization': 'Back'
    }
    
    try:
        response = requests.post(url, files=files, data=data)
        print("Status Code:", response.status_code)
        
        if response.status_code == 200:
            result = response.json()
            print("Response JSON:")
            import json
            print(json.dumps(result, indent=2))
            if result.get("success") is True:
                print("✅ Test PASSED successfully!")
            else:
                print("❌ Test FAILED (success was False)")
        else:
            print("❌ Test FAILED (status code not 200)")
            print(response.text)
            
    except Exception as e:
        print("❌ Request failed:", e)
