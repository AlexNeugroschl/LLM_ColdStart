from src.data_loader import load_data
from src.modifiers import apply_knn, apply_tfidf_knn
from src.trainer import train_model
from src.evaluator import evaluate

# Note: Once you write your KNN logic, you will import it here like:
# from src.modifiers import apply_knn

class ColdStartExperiment:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def run_single_pass(self, modifier_func=None):
        """
        Executes a standard 80/10/10 split for prototyping.
        If a modifier_func (like KNN) is provided, it applies it to the cold items 
        after baseline training but before evaluation.
        """
        run_type = modifier_func.__name__ if modifier_func else 'Baseline'
        print(f"\n=== Running Single Pass: {run_type} for {self.dataset_name} ===")
        
        # 1. Load Data State
        state = load_data(self.dataset_name)
        
        # 2. Train the Standard Collaborative Filtering Model
        state = train_model(state)
        
        # 3. Apply Custom Research Modifications (e.g., LLM Semantic Imputation)
        if modifier_func:
            print(f"Applying modifier: {run_type}...")
            state = modifier_func(state)
            
        # 4. Evaluate against the Test Set
        results = evaluate(state)
        
        return results

if __name__ == "__main__":
    experiment = ColdStartExperiment('amazon-office')
    
    # Contestant 1: The Blind Baseline
    baseline_metrics = experiment.run_single_pass()
    
    # Contestant 2: The TF-IDF Keyword Baseline
    tfidf_modifier = lambda state: apply_tfidf_knn(state, threshold=5, k_neighbors=5)
    tfidf_metrics = experiment.run_single_pass(modifier_func=tfidf_modifier)
    
    # Contestant 3: Your Ollama LLM
    knn_modifier = lambda state: apply_knn(state, threshold=5, k_neighbors=5, embed_model='mxbai-embed-large')
    knn_metrics = experiment.run_single_pass(modifier_func=knn_modifier)

    print("\n" + "="*60)
    print(" 🏆 FINAL RESEARCH SCORECARD 🏆")
    print("="*60)
    print(f"{'Metric':<10} | {'BPR (Zero)':<12} | {'TF-IDF (1990s)':<15} | {'Ollama LLM':<15}")
    print("-" * 60)
    
    for metric in baseline_metrics.keys():
        if metric in knn_metrics and metric in tfidf_metrics:
            base_val = baseline_metrics[metric]
            tf_val = tfidf_metrics[metric]
            knn_val = knn_metrics[metric]
            print(f"{metric:<10} | {base_val:<12.4f} | {tf_val:<15.4f} | {knn_val:<15.4f}")
    print("="*60)