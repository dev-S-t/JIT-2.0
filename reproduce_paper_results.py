import os
import subprocess
import sys

def run_step(script_path, description):
    print("="*80)
    print(f"STEP: {description}")
    print(f"Executing: python {script_path}")
    print("="*80)
    
    result = subprocess.run([sys.executable, script_path], cwd=os.path.dirname(script_path))
    
    if result.returncode != 0:
        print(f"\n❌ Error executing {script_path}")
        sys.exit(1)
    
    print("\n✅ Step completed successfully.\n")

if __name__ == "__main__":
    print("""
    🩸 BBMS Research Paper - Reproducibility Script 🩸
    ---------------------------------------------------
    This script sequentially runs the entire pipeline to generate
    the exact data models and metrics reported in the paper.
    """)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Step 1: Data Generation
    data_gen_script = os.path.join(base_dir, "data_generation", "generate_platelet_data.py")
    run_step(data_gen_script, "Generating realistic synthetic hospital dataset")
    
    # Step 2: Model Training & Prediction
    model_script = os.path.join(base_dir, "models", "demand_predictor.py")
    run_step(model_script, "Training SMA, SARIMA, and XGBoost models. Generating Table I error metrics.")
    
    # Step 3: Simulation Engine
    sim_script = os.path.join(base_dir, "simulation", "inventory_sim.py")
    run_step(sim_script, "Running inventory simulations (Traditional vs JIT+Micro). Generating Table II Wastage/Shortage metrics.")
    
    print("🎉 Pipeline Reproduction Complete!")
    print("All models have been serialized in 'models/trained_models/'")
    print("All daily simulation logs are available in 'outputs/'")
