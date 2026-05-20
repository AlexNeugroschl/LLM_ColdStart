import os
import requests
import pandas as pd
import urllib3

# Mute SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

DS_NAME = 'ml-100k'

def download_and_convert():
    print(f"\n=== Processing {DS_NAME} ===")
    
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.dirname(SCRIPT_DIR)
    out_dir = os.path.join(ROOT_DIR, 'dataset', DS_NAME)
    os.makedirs(out_dir, exist_ok=True)
    
    # We are putting the raw text files in a subfolder just to keep it clean
    extracted_dir = os.path.join(out_dir, 'ml-100k')
    os.makedirs(extracted_dir, exist_ok=True)
    
    u_item = os.path.join(extracted_dir, 'u.item')
    u_data = os.path.join(extracted_dir, 'u.data')
    u_genre = os.path.join(extracted_dir, 'u.genre')

    # 1. Direct Text Download (NO ZIPS, NO PASSWORDS, 1-SHOT ATTEMPT)
    def fetch_file(filename, dest):
        urls = [
            f"https://huggingface.co/datasets/harisarang/ml-100k/resolve/main/raw/{filename}",
            f"https://raw.githubusercontent.com/s-miller/Content-based-recommender-system-using-Movielens-dataset/master/ml-100k/{filename}"
        ]
        headers = {'User-Agent': 'Mozilla/5.0'}
        for url in urls:
            print(f"  Fetching {filename} from {url.split('/')[2]}...")
            try:
                resp = requests.get(url, timeout=15, headers=headers, verify=False)
                resp.raise_for_status()
                with open(dest, 'wb') as f:
                    f.write(resp.content)
                print(f"  ✅ {filename} downloaded.")
                return True
            except Exception as e:
                print(f"  ❌ Failed: {type(e).__name__}")
        return False

    if not os.path.exists(u_item) or not os.path.exists(u_data):
        print("Raw files not found. Initiating direct text download...")
        
        # We strictly need data and items. Genre is optional but we try to grab it.
        if not (fetch_file('u.item', u_item) and fetch_file('u.data', u_data)):
            print("\n❌ FATAL: Could not download the required text files.")
            raise SystemExit(1)
        fetch_file('u.genre', u_genre)

    # 2. Process Metadata (.item)
    print("\nFormatting metadata...")
    genres = []
    if os.path.exists(u_genre):
        try:
            with open(u_genre, 'r', encoding='latin-1') as f:
                for line in f:
                    if '|' in line:
                        name = line.split('|')[0].strip()
                        if name: genres.append(name)
        except Exception:
            pass

    items = []
    with open(u_item, 'r', encoding='latin-1', errors='ignore') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) < 2: continue
            
            item_id = parts[0]
            raw_title = parts[1]
            year = raw_title[-5:-1] if raw_title.endswith(')') else 'Unknown'
            title = raw_title[:-7].strip() if raw_title.endswith(')') else raw_title

            genre_list = []
            if len(parts) >= 6:
                for idx, flag in enumerate(parts[5:]):
                    try:
                        if int(flag) == 1 and idx < len(genres):
                            genre_list.append(genres[idx])
                    except Exception:
                        continue

            genre_text = ' '.join(genre_list) if genre_list else 'Unknown'

            items.append({
                'item_id:token': item_id,
                'movie_title:token_seq': title,
                'release_year:token': year,
                'class:token_seq': genre_text
            })

    df_item = pd.DataFrame(items)
    item_out = os.path.join(out_dir, f'{DS_NAME}.item')
    df_item.to_csv(item_out, sep='\t', index=False)
    print(f"Saved {len(df_item)} movies to {item_out}")

    # 3. Process Interactions (.inter)
    print("Formatting interactions...")
    interactions = []
    with open(u_data, 'r', encoding='latin-1', errors='ignore') as f:
        for line in f:
            # ml-100k can sometimes have tabs, sometimes spaces
            parts = line.strip().split()
            if len(parts) != 4: continue
            
            interactions.append({
                'user_id:token': parts[0],
                'item_id:token': parts[1],
                'rating:float': parts[2],
                'timestamp:float': parts[3]
            })

    df_inter = pd.DataFrame(interactions)
    inter_out = os.path.join(out_dir, f'{DS_NAME}.inter')
    df_inter.to_csv(inter_out, sep='\t', index=False)
    print(f"Saved {len(df_inter)} interactions to {inter_out}")

if __name__ == "__main__":
    download_and_convert()