import os
import gzip
import json
import ast
import urllib.request
import pandas as pd

URLS = {
    'inter_url': 'http://cseweb.ucsd.edu/~wckang/steam_reviews.json.gz',
    'meta_url': 'http://cseweb.ucsd.edu/~wckang/steam_games.json.gz'
}
DS_NAME = 'steam'

def parse_line(line):
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return ast.literal_eval(line.decode('utf-8'))

def download_and_convert():
    print(f"\n=== Processing {DS_NAME} ===")
    
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.dirname(SCRIPT_DIR)
    out_dir = os.path.join(ROOT_DIR, 'dataset', DS_NAME)
    os.makedirs(out_dir, exist_ok=True)
    
    inter_out = os.path.join(out_dir, f'{DS_NAME}.inter')
    item_out = os.path.join(out_dir, f'{DS_NAME}.item')

    # 1. Process Interactions
    print("Downloading and formatting Steam reviews...")
    req = urllib.request.Request(URLS['inter_url'], headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        with gzip.GzipFile(fileobj=response) as gz:
            inters = []
            for line in gz:
                data = parse_line(line)
                inters.append({
                    'user_id:token': data.get('username', ''),
                    'item_id:token': data.get('product_id', ''),
                    'timestamp:float': 0 # UCSD steam reviews often lack standard unix time, dummy 0 is fine
                })
            df_inter = pd.DataFrame(inters)
            df_inter.dropna(subset=['user_id:token', 'item_id:token'], inplace=True)
            df_inter.to_csv(inter_out, sep='\t', index=False)
            print(f"Saved {len(df_inter)} interactions to {inter_out}")

    # 2. Process Metadata (Rich Semantics)
    print("Downloading and formatting Steam metadata...")
    req = urllib.request.Request(URLS['meta_url'], headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        with gzip.GzipFile(fileobj=response) as gz:
            items = []
            for line in gz:
                data = parse_line(line)
                item_id = str(data.get('id', ''))
                if not item_id:
                    continue
                
                app_name = str(data.get('app_name', '')).replace('\t', ' ').replace('\n', ' ').strip()
                
                genres = data.get('genres', [])
                genres_seq = ' '.join([g.replace(' ', '_') for g in genres]) if isinstance(genres, list) else ''
                
                tags = data.get('tags', [])
                tags_seq = ' '.join([t.replace(' ', '_') for t in tags]) if isinstance(tags, list) else ''

                items.append({
                    'item_id:token': item_id,
                    'app_name:token_seq': app_name if app_name else 'Unknown',
                    'genres:token_seq': genres_seq,
                    'tags:token_seq': tags_seq
                })
                
            df_item = pd.DataFrame(items)
            df_item.drop_duplicates(subset=['item_id:token'], inplace=True)
            df_item.to_csv(item_out, sep='\t', index=False)
            print(f"Saved {len(df_item)} rich semantic items to {item_out}")

if __name__ == "__main__":
    download_and_convert()