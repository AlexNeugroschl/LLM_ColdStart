import os
import numpy as np
import torch.distributed as dist

# 1. Patch NumPy for RecBole compatibility
if not hasattr(np, 'float_'):
    np.float_ = np.float64

# 2. Fix the Distributed Error
if not dist.is_initialized():
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='gloo', rank=0, world_size=1)

from recbole.config import Config
from recbole.data import create_dataset

dataset_columns = {
    'ml-100k': ['item_id', 'movie_title', 'release_year', 'class'],
    'amazon-books': ['item_id', 'title', 'categories', 'brand'],
    'yelp': ['item_id', 'item_name', 'categories', 'city']
}

for ds_name in ['ml-100k', 'amazon-books', 'yelp']:
    print(f"\n--- Loading {ds_name} with Metadata ---")
    
    config_dict = {
        'model': 'BPR',
        'dataset': ds_name,
        'user_inter_num_interval': "[0,inf)",
        'item_inter_num_interval': "[0,inf)",
        'filter_inter_by_user_or_item': False,
        
        # THIS IS THE KEY CHANGE:
        # We now tell RecBole exactly which metadata columns to pull into memory
        'load_col': {
            'inter': ['user_id', 'item_id', 'rating', 'timestamp'],
            'item': dataset_columns[ds_name]
        },
    }

    config = Config(model='BPR', dataset=ds_name, config_dict=config_dict)
    dataset = create_dataset(config)
    
    # Now dataset.item_feat will actually contain the text data
    print(f"Successfully loaded {ds_name}. Columns: {dataset.item_feat.columns.tolist()}")