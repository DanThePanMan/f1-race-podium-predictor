from datetime import datetime, timezone, timedelta
import requests
from dotenv import load_dotenv
import os
import pandas as pd
import json

all_listings = []
cursor = None
cutoff_time = datetime.now(timezone.utc) - timedelta(weeks=6)
# cutoff_time = datetime.now(timezone.utc) - timedelta(hours=1)

load_dotenv()
headers = {
    "Authorization": os.getenv("CSFLOAT_API_KEY")
}
base_url = "https://csfloat.com/api/v1/listings?limit=50&def_index=500,503,505,506,507,508,509,512,514,515,516,517,518,519,520,521,522,523,525,526&sort_by=most_recent"

while True:
    if cursor:
        url = f"{base_url}&cursor={cursor}"
    else:
        url = base_url
    
    data = requests.get(url, headers=headers)
    
    if data.status_code != 200:
        print(f"Error: {data.status_code}")
        break
    results = data.json()
    
    
    for item in results['data']:
        last_updated_str = item['reference']['last_updated']
        last_updated = datetime.fromisoformat(last_updated_str.replace('Z', '+00:00'))
        if last_updated < cutoff_time:
            print(f"cutoff time reached: {last_updated_str}")
            break
        all_listings.append(item)
    else:
        cursor = results.get('cursor')
        # Print the date of the last item in this batch
        if results['data']:
            last_item_date = results['data'][-1]['reference']['last_updated']
            print(f"Fetched {len(results['data'])} listings. Total: {len(all_listings)}. Last item date: {last_item_date}")
        continue 
    break


# puts it in a csv

if all_listings:
    flattened_data = []
    for item in all_listings:
        reference = item.get('reference', {})
        item_data = item.get('item', {})
        
        flat_item = {
            'reference_predicted_price': reference.get('predicted_price'), # this is the y
            'item_name': item_data.get('market_hash_name'),
            'item_wear': item_data.get('wear_name'),
            'item_is_stattrak': item_data.get('is_stattrak'),
            'item_rarity': item_data.get('rarity'),
            'item_quality': item_data.get('quality'),
            
            # Additional useful fields
            'item_float_value': item_data.get('float_value'),
            'item_paint_index': item_data.get('paint_index'),
        }
        flattened_data.append(flat_item)
    
    df = pd.DataFrame(flattened_data)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'csfloat_listings_{timestamp}.csv'
    
    df.to_csv(filename, index=False)
    print("done")
else:
    print("No listings to save")
