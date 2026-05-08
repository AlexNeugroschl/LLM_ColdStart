import os
import torch
import numpy as np
from scipy import stats
from src.visualization import plot_knn_sweep

# --- RECBOLE IMPORTS ---
from recbole.quick_start import run_recbole, load_data_and_model
from recbole.utils import init_logger, get_trainer

# --- MODIFIER IMPORTS ---
from src.modifiers import apply_knn, apply_tfidf_knn, apply_contrastive_mapper

class ColdStartExperiment:
    def __init__(self, dataset_name, is_oracle=False, seed=2024):
        self.dataset_name = dataset_name
        self.is_oracle = is_oracle
        self.seed = seed
        self.saved_model_path = None 

    def prepare_base_model(self):
        """Trains the base model ONCE per seed and caches the weights to disk."""
        mode_text = "ORACLE (80/10/10)" if self.is_oracle else "STRICT HOLDOUT"
        print(f"\n{'='*50}")
        print(f" TRAINING BASE MODEL: {self.dataset_name} Seed: {self.seed} [{mode_text}] ")
        print(f"{'='*50}")
        
        config_dict = {
            'seed': self.seed,
            'epochs': 50, # Set to your standard baseline
            'eval_args': {'split': {'RS': [0.8, 0.1, 0.1]}, 'order': 'RO', 'mode': 'full'}
        }
        
        # Train and cache to the /saved directory
        result = run_recbole(model='BPR', dataset=self.dataset_name, config_dict=config_dict)
        self.saved_model_path = result['best_valid_model_file']
        print(f"✅ Base model trained and cached to disk at: {self.saved_model_path}")

    def run_evaluation(self, modifier_func=None, name="Baseline"):
        """Loads the sterile model from disk, applies the modifier, and evaluates."""
        if self.saved_model_path is None:
            self.prepare_base_model()

        print(f"\n--- Running Evaluation: {name} ---")
        
        # LOAD PRISTINE STATE FROM DISK 
        config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
            model_file=self.saved_model_path
        )
        
        current_state = {
            'config': config,
            'model': model,
            'dataset': dataset,
            'test_data': test_data
        }
        
        # APPLY IMPUTATION
        if modifier_func:
            current_state = modifier_func(current_state)
            
        # EVALUATE
        print("Evaluating model on test set...")
        model = current_state['model']
        test_data = current_state['test_data']
        
        trainer = get_trainer(config['MODEL_TYPE'], config['MODEL'])(config, model)
        test_result = trainer.evaluate(test_data, load_best_model=False, show_progress=False)
        
        return test_result


def run_statistical_suite(dataset_name='amazon-office', seeds=None):
    """The Master Suite: Runs TF-IDF, Mapper, 4-Axis Prompting, and Oracle."""
    if seeds is None:
        seeds = [42, 101, 777, 2024, 9999]
        
    print(f"\n{'='*80}")
    print(f" STARTING MASTER STATISTICAL SUITE: {dataset_name.upper()}")
    print(f" Seeds to evaluate: {seeds}")
    print(f"{'='*80}")

    # All architectures we are evaluating
    architectures = [
        "oracle", 
        "tf-idf", 
        "mapper_net", 
        "1-nn_raw", 
        "1-nn_prefixed", 
        "1-nn_vibe_only", 
        "1-nn_combination"
    ]
    
    results = {arch: {'hit': [], 'ndcg': []} for arch in architectures}

    for idx, seed in enumerate(seeds):
        print(f"\n\n{'#'*60}")
        print(f" RUNNING ACADEMIC SUITE {idx+1}/{len(seeds)} (SEED: {seed})")
        print(f"{'#'*60}")
        
        # ==========================================
        # 1. ORACLE CEILING
        # ==========================================
        oracle_exp = ColdStartExperiment(dataset_name, is_oracle=True, seed=seed)
        oracle_exp.prepare_base_model()
        orc_res = oracle_exp.run_evaluation(name="Oracle Baseline")
        results["oracle"]['hit'].append(orc_res['hit@10'])
        results["oracle"]['ndcg'].append(orc_res['ndcg@10'])

        # ==========================================
        # 2. STANDARD MODELS (Train base once)
        # ==========================================
        std_exp = ColdStartExperiment(dataset_name, is_oracle=False, seed=seed)
        std_exp.prepare_base_model()
        
        # A. TF-IDF + KNN (The Lexical Baseline)
        tf_mod = lambda state: apply_tfidf_knn(state, threshold=5, k_neighbors=5)
        tf_res = std_exp.run_evaluation(modifier_func=tf_mod, name="TF-IDF + KNN")
        results["tf-idf"]['hit'].append(tf_res['hit@10'])
        results["tf-idf"]['ndcg'].append(tf_res['ndcg@10'])
        
        # B. Mapper Network (The Deep Learning Baseline)
        map_mod = lambda state: apply_contrastive_mapper(state, threshold=5, embed_model='mxbai-embed-large', epochs=150)
        map_res = std_exp.run_evaluation(modifier_func=map_mod, name="Mapper Network")
        results["mapper_net"]['hit'].append(map_res['hit@10'])
        results["mapper_net"]['ndcg'].append(map_res['ndcg@10'])

        # C. The 4-Axis Prompt Engineering Suite (1-NN)
        prompt_strategies = ["raw", "prefixed", "vibe_only", "combination"]
        for strategy in prompt_strategies:
            knn_mod = lambda state: apply_knn(state, k_neighbors=1, strategy_name=strategy)
            res = std_exp.run_evaluation(modifier_func=knn_mod, name=f"1-NN ({strategy.upper()})")
            
            arch_key = f"1-nn_{strategy}"
            results[arch_key]['hit'].append(res['hit@10'])
            results[arch_key]['ndcg'].append(res['ndcg@10'])

    # ==========================================
    # STATISTICAL MATH & FINAL SCORECARD
    # ==========================================
    def calc_stats(arr):
        return np.mean(arr), np.std(arr)

    print("\n" + "="*110)
    print(f" 🏆 ULTIMATE {len(seeds)}-SEED STATISTICAL SCORECARD 🏆")
    print("="*110)
    print(f"{'Model Architecture':<30} | {'Hit@10 (Mean ± SD)':<25} | {'NDCG@10 (Mean ± SD)':<25} | {'Note'}")
    print("-" * 110)
    
    # Helper to print rows cleanly
    def print_row(key, name_override, note):
        h_m, h_s = calc_stats(results[key]['hit'])
        n_m, n_s = calc_stats(results[key]['ndcg'])
        print(f"{name_override:<30} | {h_m:.4f} ± {h_s:.4f}     | {n_m:.4f} ± {n_s:.4f}     | {note}")

    print_row("tf-idf", "TF-IDF + KNN (K=5)", "Lexical Baseline")
    print_row("mapper_net", "Contrastive Mapper Network", "Neural Baseline")
    print("-" * 110)
    print_row("1-nn_raw", "1-NN (Raw Metadata)", "Semantic SOTA Baseline")
    print_row("1-nn_prefixed", "1-NN (Instruction Prefix)", "Embedder Attention Shift")
    print_row("1-nn_vibe_only", "1-NN (LLM 'Vibe' Only)", "Generative Augmentation")
    print_row("1-nn_combination", "1-NN (Hybrid Text)", "Proposed Full Architecture")
    print("-" * 110)
    print_row("oracle", "Oracle Ceiling", "Theoretical Maximum")
    print("="*110)


if __name__ == "__main__":
    
    # ---------------------------------------------------------
    # MODE 1: Fast Dev Testing (Single Seed)
    # ---------------------------------------------------------
    # sweep_exp = ColdStartExperiment('amazon-office', is_oracle=False, seed=42)
    # sweep_exp.prepare_base_model()
    # k_test_values = [1, 2, 3, 5, 8, 12, 15, 20, 30]
    # hits, ndcgs = [], []
    # for k in k_test_values:
    #     knn_mod = lambda state: apply_knn(state, k_neighbors=k, strategy_name="raw")
    #     res = sweep_exp.run_evaluation(modifier_func=knn_mod, name=f"KNN (k={k})")
    #     hits.append(res['hit@10'])
    #     ndcgs.append(res['ndcg@10'])
    # plot_knn_sweep(k_test_values, hits, ndcgs, title="Ollama KNN Sweep")

    # ---------------------------------------------------------
    # MODE 2: The Final Publication Run
    # ---------------------------------------------------------
    run_statistical_suite('amazon-office')