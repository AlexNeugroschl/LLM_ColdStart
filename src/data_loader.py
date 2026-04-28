import os
import torch.distributed as dist
from recbole.config import Config
from recbole.data import create_dataset, data_preparation

def load_data(ds_name='ml-100k'):
    if not dist.is_initialized():
        os.environ['MASTER_ADDR'] = 'localhost'; os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group(backend='gloo', rank=0, world_size=1)

    dataset_columns = {
        'ml-100k': ['item_id', 'movie_title', 'class'],
        'amazon-books': ['item_id', 'title', 'categories'],
        'yelp': ['item_id', 'item_name', 'categories'] 
    }

    config_dict = {
        'model': 'BPR',
        'dataset': ds_name,
        'data_path': 'dataset/',
        'user_inter_num_interval': "[0,inf)",
        'item_inter_num_interval': "[0,inf)",
        'filter_inter_by_user_or_item': False,
        'load_col': {
            'inter': ['user_id', 'item_id', 'rating', 'timestamp'],
            'item': dataset_columns.get(ds_name, ['item_id'])
        },
        'eval_args': {'split': {'RS': [0.8, 0.1, 0.1]}} # Standard 80/10/10 split
    }

    config = Config(model='BPR', dataset=ds_name, config_dict=config_dict)
    dataset = create_dataset(config)
    
    # Split into train, valid, test dataloaders
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # Return the Pipeline Payload
    return {
        'config': config,
        'dataset': dataset,
        'train_data': train_data,
        'valid_data': valid_data,
        'test_data': test_data,
        'model': None # Will be initialized in the trainer
    }