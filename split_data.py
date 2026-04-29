import pandas as pd
import numpy as np
import os

# ==========================================
# 1. Configuration
# ==========================================
DATASET_NAME = 'amazon-office' # Change this to 'amazon-office' or your dataset of choice
DATA_PATH = f'dataset/{DATASET_NAME}/{DATASET_NAME}.inter'

MIN_CLICKS = 5    # The K-Core Density Threshold
COLD_RATIO = 0.10  # 10% of items will be frozen as "Simulated Cold"
VALID_RATIO = 0.10 # 10% of Warm clicks used for early stopping

print(f"Loading {DATA_PATH}...")
df = pd.read_csv(DATA_PATH, sep='\t', dtype=str)

user_col = [col for col in df.columns if 'user_id' in col][0]
item_col = [col for col in df.columns if 'item_id' in col or 'business_id' in col][0]

# ==========================================
# 2. The Density Filter (K-Core)
# ==========================================
print(f"Raw Interactions: {len(df)}")
print("Applying K-Core density filter...")

# Recursively filter until both users and items have >= MIN_CLICKS
while True:
    start_len = len(df)
    
    # Filter Users
    user_counts = df[user_col].value_counts()
    df = df[df[user_col].isin(user_counts[user_counts >= MIN_CLICKS].index)]
    
    # Filter Items
    item_counts = df[item_col].value_counts()
    df = df[df[item_col].isin(item_counts[item_counts >= MIN_CLICKS].index)]
    
    if len(df) == start_len:
        break

print(f"Dense Interactions (>= {MIN_CLICKS} clicks): {len(df)}")

# ==========================================
# 3. The Strict Holdout Split
# ==========================================
unique_items = df[item_col].unique()
np.random.seed(42) # Lock for reproducible research
np.random.shuffle(unique_items)

num_cold = int(len(unique_items) * COLD_RATIO)
cold_items = set(unique_items[:num_cold])
warm_items = set(unique_items[num_cold:])

print(f"\nTotal Dense Items: {len(unique_items)} | Warm: {len(warm_items)} | Simulated Cold: {len(cold_items)}")

# 100% of clicks for the "Cold" items are hidden in the Test Set
test_df = df[df[item_col].isin(cold_items)]

# 100% of clicks for the "Warm" items are used for Training/Validation
warm_df = df[df[item_col].isin(warm_items)]

# Shuffle the warm interactions and split them for early-stopping
warm_df = warm_df.sample(frac=1, random_state=42).reset_index(drop=True)
num_valid = int(len(warm_df) * VALID_RATIO)

valid_df = warm_df.iloc[:num_valid]
train_df = warm_df.iloc[num_valid:]

# ==========================================
# 4. Save the Pre-Split Files
# ==========================================
train_df.to_csv(f'dataset/{DATASET_NAME}/{DATASET_NAME}.train.inter', sep='\t', index=False)
valid_df.to_csv(f'dataset/{DATASET_NAME}/{DATASET_NAME}.valid.inter', sep='\t', index=False)
test_df.to_csv(f'dataset/{DATASET_NAME}/{DATASET_NAME}.test.inter', sep='\t', index=False)

# ==========================================
# 5. Sync the Metadata (.item) File
# ==========================================
print("Syncing metadata file to match dense items...")
item_df = pd.read_csv(f'dataset/{DATASET_NAME}/{DATASET_NAME}.item', sep='\t', dtype=str)
dense_item_df = item_df[item_df[item_col].isin(unique_items)]
dense_item_df.to_csv(f'dataset/{DATASET_NAME}/{DATASET_NAME}.item', sep='\t', index=False)
print(f"Reduced metadata from {len(item_df)} to {len(dense_item_df)} items.")


print("\n=== Data splitting complete! ===")
print(f"Train interactions (Warm): {len(train_df)}")
print(f"Valid interactions (Warm): {len(valid_df)}")
print(f"Test interactions (Ground Truth for Cold): {len(test_df)}")