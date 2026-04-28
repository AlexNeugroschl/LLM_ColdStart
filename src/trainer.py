import copy
from recbole.model.general_recommender import BPR
from recbole.trainer import Trainer

def train_model(payload):
    # Deepcopy the payload so we don't overwrite previous pipeline states
    state = copy.copy(payload) 
    
    # Initialize the model
    model = BPR(state['config'], state['dataset']).to(state['config']['device'])
    
    # Initialize RecBole's trainer
    trainer = Trainer(state['config'], model)
    
    # Train the model (silently)
    print("\nTraining standard model...")
    trainer.fit(state['train_data'], state['valid_data'], show_progress=False)
    
    state['model'] = model
    return state