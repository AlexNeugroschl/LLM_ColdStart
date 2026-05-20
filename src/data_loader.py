import os
import torch.distributed as dist
from recbole.config import Config
from recbole.data import create_dataset, data_preparation

def load_data(ds_name='amazon-office', is_oracle=False, seed=2024):
    if not dist.is_initialized():
        os.environ['MASTER_ADDR'] = 'localhost'; os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group(backend='gloo', rank=0, world_size=1)

    dataset_columns = {
        'amazon-office': ['item_id', 'title', 'categories', 'brand', 'description'],
        'amazon-digital-music': ['item_id', 'title', 'categories', 'brand', 'description'],
        'ml-1m': ['item_id', 'movie_title', 'release_year', 'class'],
        'ml-100k': ['item_id', 'movie_title', 'release_year', 'class']
    }

    # 1. Define the dynamic split logic
    if is_oracle:
        print(f"--- LOADING ORACLE CONFIGURATION FOR {ds_name} ---")
        eval_args = {'split': {'RS': [0.8, 0.1, 0.1]}, 'group_by': 'user', 'order': 'RO', 'mode': 'full'}
        benchmark_filename = None  # Oracle splits the master file dynamically
    else:
        print(f"--- LOADING STRICT HOLDOUT CONFIGURATION FOR {ds_name} ---")
        eval_args = {'group_by': 'user', 'order': 'RO', 'mode': 'full'}
        benchmark_filename = ['train', 'valid', 'test']  # Forces RecBole to use your split_data.py files

    # 2. Build the dynamic config dictionary
    config_dict = {
        'seed': seed,
        'dataset': ds_name,
        'data_path': 'dataset/',
        'user_inter_num_interval': "[0,inf)", 
        'item_inter_num_interval': "[0,inf)", 
        'filter_inter_by_user_or_item': False,
        'benchmark_filename': benchmark_filename,  # <--- The Missing Link!
        'load_col': {
            'inter': ['user_id', 'item_id', 'rating', 'timestamp'],
            'item': dataset_columns.get(ds_name, ['item_id'])
        },
        'eval_args': eval_args
    }

    # 3. Load the Base YAML from the new configs folder
    yaml_file = 'configs/base_recbole.yaml'
    config_file_list = [yaml_file] if os.path.exists(yaml_file) else []

    config = Config(model='BPR', dataset=ds_name, config_dict=config_dict, config_file_list=config_file_list)
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    return {
        'config': config, 'dataset': dataset, 'train_data': train_data, 
        'valid_data': valid_data, 'test_data': test_data, 'model': None
    }