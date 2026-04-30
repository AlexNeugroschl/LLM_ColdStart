import copy
import torch
import numpy as np
from scipy import stats
from src.data_loader import load_data
from src.trainer import train_model
from src.modifiers import apply_knn, apply_tfidf_knn, apply_contrastive_mapper
from recbole.utils import init_logger, get_model, get_trainer

class ColdStartExperiment:
    def __init__(self, dataset_name, is_oracle=False, seed=2024):
        self.dataset_name = dataset_name
        self.is_oracle = is_oracle
        self.seed = seed
        self.base_state = None # This will hold our single source of truth

    def prepare_base_model(self):
        """Trains the model ONCE and stores it in memory."""
        mode_text = "ORACLE (80/10/10)" if self.is_oracle else "STRICT HOLDOUT"
        print(f"\n{'='*50}")
        print(f" TRAINING BASE MODEL: {self.dataset_name} Seed: {self.seed} [{mode_text}] ")
        print(f"{'='*50}")
        
        state = load_data(self.dataset_name, is_oracle=self.is_oracle, seed=self.seed)
        self.base_state = train_model(state)
        print("Base model trained and frozen in memory!")

    def run_evaluation(self, modifier_func=None, name="Baseline"):
        """Clones the base model, applies the modifier, and evaluates."""
        if self.base_state is None:
            self.prepare_base_model()

        print(f"\n--- Running Evaluation: {name} ---")
        
        # 1. CREATE A PRISTINE CLONE OF THE STATE AND MODEL
        # We must deepcopy the model so imputation doesn't corrupt the base weights
        current_state = copy.copy(self.base_state)
        current_state['model'] = copy.deepcopy(self.base_state['model'])
        
        # 2. APPLY IMPUTATION (If any)
        if modifier_func:
            current_state = modifier_func(current_state)
            
        # 3. EVALUATE
        print("Evaluating model on test set...")
        model = current_state['model']
        test_data = current_state['test_data']
        config = current_state['config']
        
        # Initialize RecBole Trainer just for evaluation
        trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
        test_result = trainer.evaluate(test_data, load_best_model=False, show_progress=False)
        
        return test_result


def run_statistical_suite(dataset_name='amazon-office', seeds=None):
    """Runs a complete ablation study across multiple seeds for statistical validity."""
    if seeds is None:
        seeds = [42, 101, 777, 2024, 9999]
        
    print(f"\n{'='*80}")
    print(f" STARTING MULTI-SEED STATISTICAL SUITE: {dataset_name.upper()}")
    print(f" Seeds to evaluate: {seeds}")
    print(f"{'='*80}")

    tfidf_hits, tfidf_ndcgs = [], []
    llm_hits, llm_ndcgs = [], []
    oracle_hits, oracle_ndcgs = [], []

    for idx, seed in enumerate(seeds):
        print(f"\n\n{'#'*60}")
        print(f" RUNNING ACADEMIC SUITE {idx+1}/{len(seeds)} (SEED: {seed})")
        print(f"{'#'*60}")
        
        # 1. Oracle Run
        oracle_exp = ColdStartExperiment(dataset_name, is_oracle=True, seed=seed)
        oracle_exp.prepare_base_model()
        orc_res = oracle_exp.run_evaluation(name="Oracle Baseline")
        oracle_hits.append(orc_res['hit@10'])
        oracle_ndcgs.append(orc_res['ndcg@10'])

        # 2. Ablation Run
        standard_exp = ColdStartExperiment(dataset_name, is_oracle=False, seed=seed)
        standard_exp.prepare_base_model()
        
        tfidf_mod = lambda state: apply_tfidf_knn(state, threshold=5, k_neighbors=5)
        tf_res = standard_exp.run_evaluation(modifier_func=tfidf_mod, name="TF-IDF")
        tfidf_hits.append(tf_res['hit@10'])
        tfidf_ndcgs.append(tf_res['ndcg@10'])
        
        knn_mod = lambda state: apply_knn(state, threshold=5, k_neighbors=5, embed_model='mxbai-embed-large')
        llm_res = standard_exp.run_evaluation(modifier_func=knn_mod, name="Ollama LLM")
        llm_hits.append(llm_res['hit@10'])
        llm_ndcgs.append(llm_res['ndcg@10'])

        mapper_hits, mapper_ndcgs = [], []
        # 3. The New Projection Network
        mapper_mod = lambda state: apply_contrastive_mapper(state, threshold=5, embed_model='mxbai-embed-large', epochs=150)
        map_res = standard_exp.run_evaluation(modifier_func=mapper_mod, name="Ollama Mapper Network")
        mapper_hits.append(map_res['hit@10'])
        mapper_ndcgs.append(map_res['ndcg@10'])

    # ==========================================
    # STATISTICAL MATH & FINAL SCORECARD
    # ==========================================
    def calc_stats(arr):
        return np.mean(arr), np.std(arr)

    # 1. Calculate Means and STDs for Hit@10
    tf_hit_mean, tf_hit_std = calc_stats(tfidf_hits)
    llm_hit_mean, llm_hit_std = calc_stats(llm_hits)
    map_hit_mean, map_hit_std = calc_stats(mapper_hits)
    orc_hit_mean, orc_hit_std = calc_stats(oracle_hits)

    # 2. Calculate Means and STDs for NDCG@10
    tf_ndcg_mean, tf_ndcg_std = calc_stats(tfidf_ndcgs)
    llm_ndcg_mean, llm_ndcg_std = calc_stats(llm_ndcgs)
    map_ndcg_mean, map_ndcg_std = calc_stats(mapper_ndcgs)
    orc_ndcg_mean, orc_ndcg_std = calc_stats(oracle_ndcgs)

    # 3. T-Tests: Did the Neural Network beat simple KNN?
    _, p_hit = stats.ttest_rel(mapper_hits, llm_hits)
    _, p_ndcg = stats.ttest_rel(mapper_ndcgs, llm_ndcgs)
    
    def get_sig(p):
        return "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else "ns"))

    # 4. Print Final Dual-Metric Scorecard
    print("\n" + "="*105)
    print(f" 🏆 FINAL {len(seeds)}-SEED STATISTICAL SCORECARD 🏆")
    print("="*105)
    print(f"{'Model':<25} | {'Hit@10 (Mean ± SD)':<20} | {'NDCG@10 (Mean ± SD)':<20} | {'Note (vs KNN)'}")
    print("-" * 105)
    
    print(f"{'TF-IDF + KNN':<25} | {tf_hit_mean:.4f} ± {tf_hit_std:.4f} | {tf_ndcg_mean:.4f} ± {tf_ndcg_std:.4f} | ---")
    print(f"{'Ollama + KNN':<25} | {llm_hit_mean:.4f} ± {llm_hit_std:.4f} | {llm_ndcg_mean:.4f} ± {llm_ndcg_std:.4f} | Baseline SOTA")
    
    # Calculate percentage improvements
    hit_imp = ((map_hit_mean - llm_hit_mean) / llm_hit_mean) * 100
    ndcg_imp = ((map_ndcg_mean - llm_ndcg_mean) / llm_ndcg_mean) * 100
    
    # Format the dynamic improvement string
    imp_str = f"Hit: {hit_imp:+.1f}% ({get_sig(p_hit)}), NDCG: {ndcg_imp:+.1f}% ({get_sig(p_ndcg)})"
    print(f"{'Ollama + Mapper Network':<25} | {map_hit_mean:.4f} ± {map_hit_std:.4f} | {map_ndcg_mean:.4f} ± {map_ndcg_std:.4f} | {imp_str}")
    
    print("-" * 105)
    print(f"{'Oracle Ceiling':<25} | {orc_hit_mean:.4f} ± {orc_hit_std:.4f} | {orc_ndcg_mean:.4f} ± {orc_ndcg_std:.4f} | Theoretical Maximum")
    print("="*105)
    print("Significance keys: *** p<0.01, ** p<0.05, * p<0.1, ns: not significant")


if __name__ == "__main__":
    
    # ---------------------------------------------------------
    # MODE 1: Fast Dev Testing (Single Seed)
    # ---------------------------------------------------------
    # print("\n=== 🛠️ RUNNING QUICK DEV TEST 🛠️ ===")
    
    # # 1. Initialize experiment with a fixed seed
    # dev_exp = ColdStartExperiment('amazon-office', is_oracle=False, seed=42)
    # dev_exp.prepare_base_model()
    
    # # # 2. Run the baseline Ollama KNN
    # # knn_mod = lambda state: apply_knn(state, threshold=5, k_neighbors=5, embed_model='mxbai-embed-large')
    # # knn_res = dev_exp.run_evaluation(modifier_func=knn_mod, name="Dev: Ollama KNN")
    
    # # # 3. Run the NEW Projection Network
    # mapper_mod = lambda state: apply_contrastive_mapper(state, threshold=5, embed_model='mxbai-embed-large', epochs=150)
    # map_res = dev_exp.run_evaluation(modifier_func=mapper_mod, name="Dev: Ollama Mapper Network")

    # # 4. Print Mini-Scorecard
    # print("\n" + "="*50)
    # print(" 🛠️ QUICK DEV SCORECARD (Hit@10) 🛠️")
    # # print("="*50)
    # # print(f"Ollama + KNN:    {knn_res['hit@10']:.4f}")
    # print(f"Ollama + Mapper: {map_res['hit@10']:.4f}")
    # print("="*50)

    # ---------------------------------------------------------
    # MODE 2: The Final Publication Run
    # ---------------------------------------------------------
    # Uncomment the line below when you are ready for the final statistical test
    run_statistical_suite('amazon-office')