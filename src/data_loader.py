import os
import torch.distributed as dist
from recbole.config import Config
from recbole.data import create_dataset, data_preparation

def load_data(ds_name='ml-100k'):
    if not dist.is_initialized():
        os.environ['MASTER_ADDR'] = 'localhost'; os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group(backend='gloo', rank=0, world_size=1)

    # 1. The Trap: Strip '-oracle' to find the real data folder
    actual_dataset = ds_name.replace('-oracle', '')

    dataset_columns = {
        'ml-100k': ['item_id', 'movie_title', 'class'],
        'yelp': ['item_id', 'item_name', 'categories'],
        'amazon-office': ['item_id', 'title', 'categories', 'brand', 'description'],
        'amazon-digital-music': ['item_id', 'title', 'categories', 'brand', 'description']
    }

    # 2. But still load the specific Oracle YAML file!
    yaml_file = f'{ds_name}.yaml'
    config_file_list = [yaml_file] if os.path.exists(yaml_file) else []

    config_dict = {
        'model': 'BPR',
        'dataset': actual_dataset, # Point RecBole to the real folder
        'data_path': 'dataset/',
        'user_inter_num_interval': "[0,inf)", 
        'item_inter_num_interval': "[0,inf)", 
        'filter_inter_by_user_or_item': False,
        'load_col': {
            'inter': ['user_id', 'item_id', 'rating', 'timestamp'],
            'item': dataset_columns.get(actual_dataset, ['item_id'])
        }
    }

    # Initialize RecBole Config
    config = Config(
        model='BPR', 
        dataset=actual_dataset, # And pass the real name here
        config_dict=config_dict, 
        config_file_list=config_file_list
    )
    
    if config_file_list:
        print(f"Loaded strict evaluation rules from: {yaml_file}")
    else:
        print(f"WARNING: {yaml_file} not found! Defaulting to RecBole internal settings.")

    dataset = create_dataset(config)
    
    # Split into train, valid, test dataloaders
    train_data, valid_data, test_data = data_preparation(config, dataset)

    return {
        'config': config,
        'dataset': dataset,
        'train_data': train_data,
        'valid_data': valid_data,
        'test_data': test_data,
        'model': None
    }