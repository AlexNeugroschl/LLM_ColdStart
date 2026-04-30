import os
import gzip
import json
import ast
import urllib.request
import pandas as pd

# ==========================================
# Configuration (Stanford SNAP Mirrors)
# ==========================================
DATASETS = {
    'amazon-office': {
        'inter_url': 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Office_Products_5.json.gz',
        'meta_url': 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Office_Products.json.gz'
    },
    'amazon-digital-music': {
        'inter_url': 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Digital_Music_5.json.gz',
        'meta_url': 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Digital_Music.json.gz'
    }
}

def parse_line(line):
    """Fallback parser for Stanford's strict JSON vs Python Dict formatting"""
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return ast.literal_eval(line.decode('utf-8'))

def download_and_convert(ds_name, urls):
    print(f"\n=== Processing {ds_name} ===")
    
    # 1. Dynamically find the project root (LLM_ColdStart)
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.dirname(SCRIPT_DIR)
    
    # 2. Build the absolute path to the dataset folder
    out_dir = os.path.join(ROOT_DIR, 'dataset', ds_name)
    os.makedirs(out_dir, exist_ok=True)
    
    inter_out = os.path.join(out_dir, f'{ds_name}.inter')
    item_out = os.path.join(out_dir, f'{ds_name}.item')

    # 1. Process Interactions
    print("Downloading and formatting interactions (This may take a minute)...")
    req = urllib.request.Request(urls['inter_url'], headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        with gzip.GzipFile(fileobj=response) as gz:
            inters = []
            for line in gz:
                data = parse_line(line)
                inters.append({
                    'user_id:token': data.get('reviewerID', ''),
                    'item_id:token': data.get('asin', ''),
                    'rating:float': data.get('overall', 0.0),
                    'timestamp:float': data.get('unixReviewTime', 0)
                })
            df_inter = pd.DataFrame(inters)
            df_inter.dropna(subset=['user_id:token', 'item_id:token'], inplace=True)
            df_inter.to_csv(inter_out, sep='\t', index=False)
            print(f"Saved {len(df_inter)} interactions to {inter_out}")

    # 2. Process Metadata (The MAX Semantic Text for your LLM)
    print("Downloading and formatting rich metadata...")
    req = urllib.request.Request(urls['meta_url'], headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        with gzip.GzipFile(fileobj=response) as gz:
            items = []
            for line in gz:
                data = parse_line(line)
                
                # 1. Clean Title
                title = str(data.get('title', '')).replace('\t', ' ').replace('\n', ' ').strip()
                
                # 2. Clean Category
                categories = data.get('categories', [])
                if isinstance(categories, list) and len(categories) > 0 and isinstance(categories[0], list):
                    categories = categories[0]
                cat_seq = ' '.join([c.replace(' ', '_') for c in categories]) if categories else 'Unknown'

                # 3. Extract Brand
                brand = str(data.get('brand', '')).replace('\t', ' ').replace('\n', ' ').strip()
                if not brand:
                    brand = 'Unknown_Brand'
                # RecBole tokens shouldn't have spaces
                brand = brand.replace(' ', '_')

                # 4. Extract and Combine Descriptions & Features
                desc_list = data.get('description', [])
                feature_list = data.get('feature', [])
                
                # Convert to lists if they are raw strings
                if isinstance(desc_list, str): desc_list = [desc_list]
                if isinstance(feature_list, str): feature_list = [feature_list]
                
                # Combine them into one massive semantic paragraph
                full_desc = ' '.join([str(d) for d in feature_list + desc_list])
                full_desc = full_desc.replace('\t', ' ').replace('\n', ' ').strip()
                if not full_desc:
                    full_desc = 'No_Description'

                items.append({
                    'item_id:token': data.get('asin', ''),
                    'title:token_seq': title if title else 'No_Title',
                    'categories:token_seq': cat_seq,
                    'brand:token': brand,
                    'description:token_seq': full_desc
                })
            df_item = pd.DataFrame(items)
            df_item.drop_duplicates(subset=['item_id:token'], inplace=True)
            df_item.to_csv(item_out, sep='\t', index=False)
            print(f"Saved {len(df_item)} rich semantic items to {item_out}")

if __name__ == "__main__":
    for name, urls in DATASETS.items():
        download_and_convert(name, urls)
    print("\n=== All datasets downloaded and formatted perfectly for RecBole! ===")