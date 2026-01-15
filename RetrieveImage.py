import requests
import os

def download_image(search_query, api_key, file_name="downloaded_image.jpg"):
    # 1. Search for the image
    url = "https://api.pexels.com/v1/search"
    headers = { "Authorization": api_key }
    params = { 
        "query": search_query, 
        "per_page": 1,
        "orientation": "landscape" 
    }

    print(f"Searching for: '{search_query}'...")
    response = requests.get(url, headers=headers, params=params)

    if response.status_code != 200:
        print(f"Error: API returned status {response.status_code}")
        return

    data = response.json()

    # 2. Extract the image URL
    if data['photos']:
        # Pexels offers different sizes: original, large2x, large, medium, small, etc.
        image_url = data['photos'][0]['src']['large2x']
        print(f"Image found! Downloading from: {image_url}")

        # 3. Download the actual image bytes
        img_data = requests.get(image_url).content
        
        with open(file_name, 'wb') as handler:
            handler.write(img_data)
            
        print(f"Saved to {os.path.abspath(file_name)}")
    else:
        print("No images found for that description.")

# --- Usage ---
API_KEY = "bIlNk10qnmExw0ookjjiv1w7lkNREsDPCRDrzfZBVXemQ9TDQp2NMq62" # Replace with your actual key
query = "Performance of students given different characteristics"

download_image(query, API_KEY)