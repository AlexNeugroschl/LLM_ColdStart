import os
import torch.distributed as dist
from recbole.config import Config
from recbole.data import create_dataset

# 1. Handle the Distributed Sync (Needed for Amazon/Yelp downloads)
if not dist.is_initialized():
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='gloo', rank=0, world_size=1)

# 2. Metadata Column Configurations
dataset_columns = {
    'ml-100k': ['item_id', 'movie_title', 'class'],
    'amazon-books': ['item_id', 'title', 'categories'],
    'yelp': ['item_id', 'item_name', 'categories'] 
}

for ds_name in ['ml-100k', 'amazon-books', 'yelp']:
    print(f"\n--- Loading {ds_name} ---")
    
    config_dict = {
        'model': 'BPR',
        'dataset': ds_name,
        'data_path': 'dataset/',
        
        # Cold-Start constraints: Keep everything, even zero-interaction items
        'user_inter_num_interval': "[0,inf)",
        'item_inter_num_interval': "[0,inf)",
        'filter_inter_by_user_or_item': False,
        
        # Load the interactions AND the text metadata
        'load_col': {
            'inter': ['user_id', 'item_id', 'rating', 'timestamp'],
            'item': dataset_columns[ds_name]
        },
    }

    try:
        config = Config(model='BPR', dataset=ds_name, config_dict=config_dict)
        dataset = create_dataset(config)
        
        print(f"SUCCESS: {ds_name} loaded.")
        print(f"Total Items: {len(dataset.item_feat)}")
        
        # Display the first row to prove the text loaded correctly
        first_item = dataset.item_feat.iloc[0]
        print(f"Sample Data:\n{first_item}")
        
    except Exception as e:
        print(f"FAILED {ds_name}: {e}")