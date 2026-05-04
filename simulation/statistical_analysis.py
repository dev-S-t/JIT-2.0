import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
import random

# Add parent dir to path to import
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data_generation.generate_platelet_data import generate_demand_for_scenario, SCENARIOS, START_DATE, END_DATE
from models.demand_predictor import train_all_models
from simulation.inventory_sim import simulate_traditional, simulate_jit_only, simulate_jit_micro, calculate_metrics

def main():
    print("Running 30 iterations...")
    results = []
    
    for i in range(1, 31):
        print(f"Iteration {i}/30...")
        # Set seed
        np.random.seed(i)
        random.seed(i)
        
        # 1. Generate Data
        scenario = "Hamilton_Medium_Hospital"
        config = SCENARIOS[scenario]
        df = generate_demand_for_scenario(scenario, config, START_DATE, END_DATE)
        
        df = df.sort_values('date').reset_index(drop=True)
        for lag in [1, 2, 3, 7, 14]:
            df[f'lag_{lag}'] = df['units_demanded'].shift(lag)
        df['rolling_mean_7'] = df['units_demanded'].shift(1).rolling(7).mean()
        df['rolling_std_7'] = df['units_demanded'].shift(1).rolling(7).std()
        df['rolling_mean_14'] = df['units_demanded'].shift(1).rolling(14).mean()
        df = df.dropna().reset_index(drop=True)
        
        # 2. Train Models (to get new predictions)
        # Suppress printing
        class HiddenPrints:
            def __enter__(self):
                self._original_stdout = sys.stdout
                sys.stdout = open(os.devnull, 'w')
            def __exit__(self, exc_type, exc_val, exc_tb):
                sys.stdout.close()
                sys.stdout = self._original_stdout
                
        with HiddenPrints():
            model_results, predictions_dict, test_df, _, _, _ = train_all_models(df)
            
        demand_series = test_df['units_demanded'].values
        predictions = predictions_dict.get('SARIMA', predictions_dict['XGBoost']) # fallback
        if np.any(np.isnan(predictions)):
            predictions = predictions_dict['XGBoost'] # Use XGBoost if SARIMA fails
            
        dates = test_df['date'].values
        day_of_week = test_df['day_of_week'].values
        
        # 3. Simulate
        trad = simulate_traditional(demand_series, dates)
        jit = simulate_jit_only(demand_series, predictions, dates)
        micro = simulate_jit_micro(demand_series, predictions, dates, day_of_week)
        
        trad_metrics = calculate_metrics(trad, 'Traditional')
        jit_metrics = calculate_metrics(jit, 'JIT-Only')
        micro_metrics = calculate_metrics(micro, 'JIT+Micro')
        
        results.append({
            'iteration': i,
            'trad_waste': trad_metrics['wastage_rate'],
            'trad_short': trad_metrics['shortage_rate'],
            'jit_waste': jit_metrics['wastage_rate'],
            'jit_short': jit_metrics['shortage_rate'],
            'micro_waste': micro_metrics['wastage_rate'],
            'micro_short': micro_metrics['shortage_rate'],
            'micro_total_demand': micro_metrics['total_demand'],
            'micro_total_supply': micro_metrics['total_supply'],
            'micro_raw_wasted': micro_metrics['total_wasted'],
            'micro_raw_shortage': micro_metrics['total_shortage']
        })

    # Convert to DataFrame
    df_res = pd.DataFrame(results)
    
    # Calculate Mean, Std, 95% CI
    def get_stats(col):
        mean = df_res[col].mean()
        std = df_res[col].std()
        ci = stats.t.interval(0.95, len(df_res)-1, loc=mean, scale=std/np.sqrt(len(df_res)))
        return mean, std, ci
    
    print("\n--- STATISTICS ACROSS 30 RUNS ---")
    models = [('Traditional', 'trad'), ('JIT Only', 'jit'), ('JIT+Micro', 'micro')]
    for name, prefix in models:
        wm, ws, wci = get_stats(f'{prefix}_waste')
        sm, ss, sci = get_stats(f'{prefix}_short')
        print(f"{name}:")
        print(f"  Wastage Rate : {wm:.2f}% (Std: {ws:.2f}, 95% CI: [{wci[0]:.2f}, {wci[1]:.2f}])")
        print(f"  Shortage Rate: {sm:.2f}% (Std: {ss:.2f}, 95% CI: [{sci[0]:.2f}, {sci[1]:.2f}])")
        
    # ANOVA / paired t-test
    print("\n--- P-VALUES (Paired t-test vs JIT+Micro) ---")
    t_trad_waste, p_trad_waste = stats.ttest_rel(df_res['micro_waste'], df_res['trad_waste'])
    t_trad_short, p_trad_short = stats.ttest_rel(df_res['micro_short'], df_res['trad_short'])
    t_jit_waste, p_jit_waste = stats.ttest_rel(df_res['micro_waste'], df_res['jit_waste'])
    t_jit_short, p_jit_short = stats.ttest_rel(df_res['micro_short'], df_res['jit_short'])
    
    print(f"JIT+Micro vs Traditional Wastage p-value: {p_trad_waste:.4e}")
    print(f"JIT+Micro vs Traditional Shortage p-value: {p_trad_short:.4e}")
    print(f"JIT+Micro vs JIT-Only Wastage p-value: {p_jit_waste:.4e}")
    print(f"JIT+Micro vs JIT-Only Shortage p-value: {p_jit_short:.4e}")

    print("\n--- RAW COUNTS FOR 2.5% WASTAGE AND 0.9% SHORTAGE ---")
    # 2.5% wastage of what supply? 0.9% shortage of what demand?
    # We will use the average total demand and average total supply across 30 runs, or just the original run.
    avg_demand = df_res['micro_total_demand'].mean()
    avg_supply = df_res['micro_total_supply'].mean()
    raw_wastage = int(round(avg_supply * 0.025))
    raw_shortage = int(round(avg_demand * 0.009))
    print(f"Assuming Average Total Supply ({avg_supply:.1f}): 2.5% Wastage = {raw_wastage} units")
    print(f"Assuming Average Total Demand ({avg_demand:.1f}): 0.9% Shortage = {raw_shortage} units")

if __name__ == '__main__':
    main()
