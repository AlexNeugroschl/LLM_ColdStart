import os
import pandas as pd
import numpy as np

def generate_item_split(dataset_name):
    print(f"\n============================================================")
    print(f"🔪 GENERATING TARGETED ITEM HOLDOUT: {dataset_name}")
    print(f"============================================================")
    
    base_dir = os.path.join('dataset', dataset_name)
    inter_path = os.path.join(base_dir, f'{dataset_name}.inter')
    
    if not os.path.exists(inter_path):
        print(f"⏭️  SKIPPING: {inter_path} not found.")
        return

    # Check if files already exist to save time
    if os.path.exists(os.path.join(base_dir, f'{dataset_name}.train.inter')):
        print("⏭️  SKIPPING: Targeted holdout files already exist.")
        return
        
    df = pd.read_csv(inter_path, sep='\t')
    
    # Identify the item_id column dynamically (RecBole formatting)
    item_col = [c for c in df.columns if 'item_id' in c][0]
    
    unique_items = df[item_col].unique()
    np.random.seed(42)  # Strict reproducibility
    np.random.shuffle(unique_items)
    
    # 80% Train, 10% Valid, 10% Test (Strict Item Split)
    train_cutoff = int(len(unique_items) * 0.8)
    valid_cutoff = int(len(unique_items) * 0.9)
    
    train_items = set(unique_items[:train_cutoff])
    valid_items = set(unique_items[train_cutoff:valid_cutoff])
    test_items = set(unique_items[valid_cutoff:])
    
    train_df = df[df[item_col].isin(train_items)]
    valid_df = df[df[item_col].isin(valid_items)]
    test_df = df[df[item_col].isin(test_items)]
    
    # Save the physically isolated files
    train_df.to_csv(os.path.join(base_dir, f'{dataset_name}.train.inter'), sep='\t', index=False)
    valid_df.to_csv(os.path.join(base_dir, f'{dataset_name}.valid.inter'), sep='\t', index=False)
    test_df.to_csv(os.path.join(base_dir, f'{dataset_name}.test.inter'), sep='\t', index=False)
    
    print(f"Total Unique Items: {len(unique_items)}")
    print(f"Train Interactions (Warm): {len(train_df)}")
    print(f"Test Interactions (Strict Cold): {len(test_df)}")
    print("✅ Successfully generated .train.inter, .valid.inter, and .test.inter\n")

if __name__ == "__main__":
    generate_item_split('amazon-office')
    generate_item_split('amazon-digital-music')