import pandas as pd
import numpy as np
import os

def split_dataset(dataset_name, min_clicks=5, cold_ratio=0.10, valid_ratio=0.10):
    print(f"\n{'='*50}\nSplitting: {dataset_name}\n{'='*50}")
    
    # Dynamically find the project root
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.dirname(SCRIPT_DIR)
    DATA_DIR = os.path.join(ROOT_DIR, 'dataset', dataset_name)

    DATA_PATH = os.path.join(DATA_DIR, f'{dataset_name}.inter')
    ITEM_PATH = os.path.join(DATA_DIR, f'{dataset_name}.item')

    if not os.path.exists(DATA_PATH):
        print(f"⏭️ Skipping {dataset_name}: .inter file not found at {DATA_PATH}.")
        return

    print(f"Loading {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH, sep='\t', dtype=str)

    user_col = [col for col in df.columns if 'user_id' in col][0]
    item_col = [col for col in df.columns if 'item_id' in col or 'business_id' in col][0]

    # ==========================================
    # 1. The Density Filter (K-Core)
    # ==========================================
    print(f"Raw Interactions: {len(df)}")
    print("Applying K-Core density filter...")

    # Recursively filter until both users and items have >= MIN_CLICKS
    while True:
        start_len = len(df)
        
        # Filter Users
        user_counts = df[user_col].value_counts()
        df = df[df[user_col].isin(user_counts[user_counts >= min_clicks].index)]
        
        # Filter Items
        item_counts = df[item_col].value_counts()
        df = df[df[item_col].isin(item_counts[item_counts >= min_clicks].index)]
        
        if len(df) == start_len:
            break

    print(f"Dense Interactions (>= {min_clicks} clicks): {len(df)}")

    # ==========================================
    # 2. The Strict Holdout Split
    # ==========================================
    unique_items = df[item_col].unique()
    np.random.seed(42) # Lock for reproducible research
    np.random.shuffle(unique_items)

    num_cold = int(len(unique_items) * cold_ratio)
    cold_items = set(unique_items[:num_cold])
    warm_items = set(unique_items[num_cold:])

    print(f"\nTotal Dense Items: {len(unique_items)} | Warm: {len(warm_items)} | Simulated Cold: {len(cold_items)}")

    # 100% of clicks for the "Cold" items are hidden in the Test Set
    test_df = df[df[item_col].isin(cold_items)]

    # 100% of clicks for the "Warm" items are used for Training/Validation
    warm_df = df[df[item_col].isin(warm_items)]

    # Shuffle the warm interactions and split them for early-stopping
    warm_df = warm_df.sample(frac=1, random_state=42).reset_index(drop=True)
    num_valid = int(len(warm_df) * valid_ratio)

    valid_df = warm_df.iloc[:num_valid]
    train_df = warm_df.iloc[num_valid:]

    # ==========================================
    # 3. Save the Pre-Split Files
    # ==========================================
    train_df.to_csv(os.path.join(DATA_DIR, f'{dataset_name}.train.inter'), sep='\t', index=False)
    valid_df.to_csv(os.path.join(DATA_DIR, f'{dataset_name}.valid.inter'), sep='\t', index=False)
    test_df.to_csv(os.path.join(DATA_DIR, f'{dataset_name}.test.inter'), sep='\t', index=False)

    print("\n=== Data splitting complete! ===")
    print(f"Train interactions (Warm): {len(train_df)}")
    print(f"Valid interactions (Warm): {len(valid_df)}")
    print(f"Test interactions (Ground Truth for Cold): {len(test_df)}")

    # ==========================================
    # 4. Sync the Metadata (.item) File
    # ==========================================
    print("\nSyncing metadata file to match dense items...")
    item_df = pd.read_csv(ITEM_PATH, sep='\t', dtype=str)
    dense_item_df = item_df[item_df[item_col].isin(unique_items)]
    dense_item_df.to_csv(ITEM_PATH, sep='\t', index=False)
    print(f"✅ Reduced metadata from {len(item_df)} to {len(dense_item_df)} items.")

if __name__ == "__main__":
    DATASETS = ['amazon-office', 'ml-1m', 'steam']
    for ds in DATASETS:
        split_dataset(ds)