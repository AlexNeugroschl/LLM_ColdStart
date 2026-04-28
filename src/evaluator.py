from recbole.trainer import Trainer

def evaluate(payload):
    print("Evaluating model on test set...")
    
    # Initialize the evaluator with the current state of the model
    trainer = Trainer(payload['config'], payload['model'])
    
    # CRITICAL FIX: load_best_model=False forces RecBole to use the in-memory weights.
    # This ensures your future KNN embedding modifications are actually evaluated!
    test_result = trainer.evaluate(
        payload['test_data'], 
        load_best_model=False, 
        show_progress=True
    )
    
    return test_result