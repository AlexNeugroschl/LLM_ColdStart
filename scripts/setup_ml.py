import os
import urllib.request
import zipfile
import pandas as pd

DATASET_URL = 'https://files.grouplens.org/datasets/movielens/ml-1m.zip'
DS_NAME = 'ml-1m'

def download_and_convert():
    print(f"\n=== Processing {DS_NAME} ===")
    
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.dirname(SCRIPT_DIR)
    out_dir = os.path.join(ROOT_DIR, 'dataset', DS_NAME)
    os.makedirs(out_dir, exist_ok=True)
    
    zip_path = os.path.join(out_dir, 'ml-1m.zip')
    
    # 1. Download and Extract
    if not os.path.exists(zip_path):
        print("Downloading MovieLens 1M...")
        urllib.request.urlretrieve(DATASET_URL, zip_path)
    
    print("Extracting files...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(out_dir)

    # File paths from inside the zip
    extracted_dir = os.path.join(out_dir, 'ml-1m')
    movies_file = os.path.join(extracted_dir, 'movies.dat')
    ratings_file = os.path.join(extracted_dir, 'ratings.dat')

    # 2. Process Metadata (.item)
    print("Formatting metadata...")
    # Movies format: MovieID::Title::Genres
    movies = []
    with open(movies_file, 'r', encoding='latin-1') as f:
        for line in f:
            parts = line.strip().split('::')
            if len(parts) != 3: continue
            
            item_id = parts[0]
            raw_title = parts[1]
            genres = parts[2].replace('|', ' ') # Replace pipes with spaces for token_seq
            
            # Extract year from title e.g. "Toy Story (1995)"
            year = raw_title[-5:-1] if raw_title.endswith(')') else 'Unknown'
            title = raw_title[:-7].strip() if raw_title.endswith(')') else raw_title
            
            movies.append({
                'item_id:token': item_id,
                'movie_title:token_seq': title,
                'release_year:token': year,
                'class:token_seq': genres
            })
            
    df_item = pd.DataFrame(movies)
    item_out = os.path.join(out_dir, f'{DS_NAME}.item')
    df_item.to_csv(item_out, sep='\t', index=False)
    print(f"Saved {len(df_item)} movies to {item_out}")

    # 3. Process Interactions (.inter)
    print("Formatting interactions...")
    # Ratings format: UserID::MovieID::Rating::Timestamp
    ratings = []
    with open(ratings_file, 'r', encoding='latin-1') as f:
        for line in f:
            parts = line.strip().split('::')
            if len(parts) != 4: continue
            ratings.append({
                'user_id:token': parts[0],
                'item_id:token': parts[1],
                'rating:float': parts[2],
                'timestamp:float': parts[3]
            })
            
    df_inter = pd.DataFrame(ratings)
    inter_out = os.path.join(out_dir, f'{DS_NAME}.inter')
    df_inter.to_csv(inter_out, sep='\t', index=False)
    print(f"Saved {len(df_inter)} interactions to {inter_out}")

if __name__ == "__main__":
    download_and_convert()