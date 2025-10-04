# 4_evaluate_guardrail.py
"""
Step 4: Measure Success.

This script performs the final evaluation of the guardrail system:
1.  Loads all tuned artifacts (vector, classifier, thresholds) and the held-out
    test set from TruthfulQA.
2.  Defines the final `answer_guarded` function, which implements the two-path
    (Fast/Steer) routing logic based on prompt risk.
3.  Defines a `generate_baseline` function for comparison.
4.  Runs a resilient evaluation loop to generate and save responses from both the
    guarded and baseline systems for the entire test set.
5.  Runs a resilient judging loop to score all generated answers for factual correctness
    using an LLM judge.
6.  Performs a comprehensive final analysis, calculating metrics for accuracy, latency,
    and path distribution.
7.  Generates and saves a risk-coverage plot and a final summary table of the results.
"""
import os
import time
import pandas as pd
import numpy as np
import torch
import joblib
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

import config
import utils

# --- Global variable for loaded artifacts to avoid reloading ---
artifacts = {}

def load_all_artifacts():
    """Loads all necessary model and project artifacts into a global dict."""
    if artifacts:  # Avoid reloading if already loaded
        return

    print("Loading all necessary artifacts for evaluation...")
    model, tokenizer = utils.load_model_and_tokenizer()
    artifacts['model'] = model
    artifacts['tokenizer'] = tokenizer
    
    artifacts['v_halluc'] = torch.load(config.VECTOR_PATH).to(model.device)
    artifacts['risk_classifier'] = joblib.load(config.CLASSIFIER_PATH)
    artifacts['thresholds'] = {
        "tau_low": config.TAU_LOW,
        "tau_high": config.TAU_HIGH,
        "optimal_alpha": config.OPTIMAL_ALPHA
    }
    print(f"Loaded thresholds: {artifacts['thresholds']}")

def answer_guarded(prompt_text: str, max_new_tokens: int = 128):
    """Generates a response using the two-path guardrail system."""
    start_time = time.time()
    
    risk_score = utils.get_hallucination_risk(
        prompt_text, artifacts['model'], artifacts['tokenizer'],
        artifacts['v_halluc'], artifacts['risk_classifier']
    )
    
    full_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nYou are a helpful assistant.\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    inputs = artifacts['tokenizer'](full_prompt, return_tensors="pt").to(artifacts['model'].device)
    input_token_length = inputs.input_ids.shape[1]

    if risk_score < artifacts['thresholds']['tau_high']:
        path = "Fast Path (Untouched)"
        with torch.no_grad():
            outputs = artifacts['model'].generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    else:
        alpha = artifacts['thresholds']['optimal_alpha']
        path = f"Steer Path (α={alpha})"
        with utils.ActivationSteerer(artifacts['model'], artifacts['v_halluc'], config.TARGET_LAYER, coeff=alpha):
            with torch.no_grad():
                outputs = artifacts['model'].generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    answer = artifacts['tokenizer'].decode(outputs[0, input_token_length:], skip_special_tokens=True)
    latency = time.time() - start_time
    
    return {"answer": answer.strip(), "risk_score": risk_score, "path_taken": path, "latency_seconds": latency}

def generate_baseline(prompt_text: str, max_new_tokens: int = 128):
    """Generates a baseline response without the guardrail."""
    start_time = time.time()
    full_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nYou are a helpful assistant.\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    inputs = artifacts['tokenizer'](full_prompt, return_tensors="pt").to(artifacts['model'].device)
    with torch.no_grad():
        outputs = artifacts['model'].generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    
    input_token_length = inputs.input_ids.shape[1]
    answer = artifacts['tokenizer'].decode(outputs[0, input_token_length:], skip_special_tokens=True)
    latency = time.time() - start_time
    
    return {"answer": answer.strip(), "latency_seconds": latency}

def run_judging_process(input_df, output_path, api_key):
    """
    CORRECTED VERSION: Iterates through a DataFrame, gets a hallucination score, 
    and saves resiliently. Fixes the TypeError by calling the judge function
    with explicit keyword arguments.
    """
    print(f"\n--- Starting Corrected Judging Process for {os.path.basename(output_path)} ---")
    
    # Initialize CSV and load processed prompts
    output_headers = input_df.columns.tolist() + ['hallucination_score', 'is_correct']
    utils.initialize_csv(output_path, output_headers)
    processed_prompts = utils.load_processed_prompts(output_path, 'prompt')
    print(f"Found {len(processed_prompts)} already judged prompts. Resuming...")

    with open(output_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for index, row in tqdm(input_df.iterrows(), total=len(input_df), desc=f"Judging {os.path.basename(output_path)}"):
            if row['prompt'] in processed_prompts:
                continue
            
            score = -1
            try:
                for _ in range(3): # Retry logic
                    score = utils.get_hallucination_score_0_100(
                        api_key=api_key,
                        question=row['Question'],        # Use the 'Question' column
                        answer=row['answer'],          # Use the 'answer' column
                        best_answer=row['Best Answer'],  # Use the 'Best Answer' column
                        correct_answers=row['Correct Answers'],
                        incorrect_answers=row['Incorrect Answers']
                    )
                    if score != -1:
                        break
                
                is_correct = 1 if 0 <= score <= 50 else 0
                new_row = row.tolist() + [score, is_correct]
                writer.writerow(new_row)

            except Exception as e:
                print(f"An unexpected error occurred for prompt '{row['prompt']}': {e}")
                error_row = row.tolist() + [-1, 0]
                writer.writerow(error_row)

def main():
    """Main execution function for final evaluation."""
    print("--- Step 4: Final Evaluation ---")
    
    # 1. Setup
    secrets = utils.load_secrets()
    api_key = secrets.get('SCALEDOWN_API_KEY')
    if not api_key:
        print("ERROR: SCALEDOWN_API_KEY not found. Cannot proceed with judging. Exiting.")
        return

    load_all_artifacts()
    test_df = pd.read_csv(config.FINAL_TEST_SET_PATH)
    print(f"Loaded test set with {len(test_df)} prompts.")

    # 2. Resilient Evaluation Loop (Generation)
    print("\nStarting response generation for baseline and guarded models...")
    guarded_headers = ['prompt', 'answer', 'risk_score', 'path_taken', 'latency_seconds']
    baseline_headers = ['prompt', 'answer', 'latency_seconds']
    utils.initialize_csv(config.GUARDED_RESULTS_PATH, guarded_headers)
    utils.initialize_csv(config.BASELINE_RESULTS_PATH, baseline_headers)
    
    processed_guarded = utils.load_processed_prompts(config.GUARDED_RESULTS_PATH)
    processed_baseline = utils.load_processed_prompts(config.BASELINE_RESULTS_PATH)

    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Evaluating Prompts"):
        prompt = row['Question']
        # Guarded Run
        if prompt not in processed_guarded:
            try:
                result = answer_guarded(prompt)
                with open(config.GUARDED_RESULTS_PATH, 'a', newline='', encoding='utf-8') as f:
                    csv.writer(f).writerow([prompt] + list(result.values()))
            except Exception as e:
                print(f"Error on guarded prompt: {prompt}. Error: {e}")
        # Baseline Run
        if prompt not in processed_baseline:
            try:
                result = generate_baseline(prompt)
                with open(config.BASELINE_RESULTS_PATH, 'a', newline='', encoding='utf-8') as f:
                    csv.writer(f).writerow([prompt] + list(result.values()))
            except Exception as e:
                print(f"Error on baseline prompt: {prompt}. Error: {e}")

    print("Response generation complete.")

    # 3. Resilient Judging Loop
    guarded_df = pd.read_csv(config.GUARDED_RESULTS_PATH)
    baseline_df = pd.read_csv(config.BASELINE_RESULTS_PATH)
    
    guarded_merged_df = pd.merge(guarded_df, test_df, left_on='prompt', right_on='Question', how='left')
    baseline_merged_df = pd.merge(baseline_df, test_df, left_on='prompt', right_on='Question', how='left')
    
    run_judging_process(guarded_merged_df, config.GUARDED_JUDGED_RESULTS_PATH, api_key)
    run_judging_process(baseline_merged_df, config.BASELINE_JUDGED_RESULTS_PATH, api_key)
    print("Judging complete for both models.")

    # 4. Final Analysis and Reporting
    print("\n--- Final Performance Analysis ---")
    guarded_judged_df = pd.read_csv(config.GUARDED_JUDGED_RESULTS_PATH)
    baseline_judged_df = pd.read_csv(config.BASELINE_JUDGED_RESULTS_PATH)

    # Accuracy / Hallucination Analysis
    baseline_accuracy = baseline_judged_df['is_correct'].mean()
    guarded_accuracy = guarded_judged_df['is_correct'].mean()
    baseline_error_rate = 1 - baseline_accuracy
    guarded_error_rate = 1 - guarded_accuracy
    relative_error_reduction = (baseline_error_rate - guarded_error_rate) / baseline_error_rate if baseline_error_rate > 0 else 0

    # Latency Analysis
    baseline_latency = baseline_judged_df['latency_seconds'].mean()
    guarded_latency = guarded_judged_df['latency_seconds'].mean()
    latency_increase_percent = (guarded_latency - baseline_latency) / baseline_latency * 100

    # Path Distribution
    path_distribution = guarded_judged_df['path_taken'].value_counts(normalize=True) * 100

    # Create Summary Table
    summary_data = {
        "Metric": ["Accuracy", "Hallucination Rate", "Avg Latency (s)", "Relative Error Reduction", "Latency Increase"],
        "Baseline Model": [f"{baseline_accuracy:.2%}", f"{baseline_error_rate:.2%}", f"{baseline_latency:.2f}", "N/A", "N/A"],
        "Guarded Model": [f"{guarded_accuracy:.2%}", f"{guarded_error_rate:.2%}", f"{guarded_latency:.2f}", f"{relative_error_reduction:.2%}", f"{latency_increase_percent:+.2f}%"],
        "Target": ["Maximize", "Minimize (≥15% Red.)", "N/A", "Maximize", "<10%"]
    }
    summary_df = pd.DataFrame(summary_data)
    print(summary_df)
    summary_df.to_csv(config.FINAL_SUMMARY_TABLE_PATH, index=False)
    print(f"\nSummary table saved to {config.FINAL_SUMMARY_TABLE_PATH}")

    # Risk-Coverage Plot
    merged_analysis_df = pd.merge(
        guarded_judged_df[['prompt', 'risk_score', 'is_correct']],
        baseline_judged_df[['prompt', 'is_correct']],
        on='prompt', suffixes=('_guarded', '_baseline')
    )
    merged_analysis_df['risk_bin'] = pd.qcut(merged_analysis_df['risk_score'], q=10, labels=False, duplicates='drop')
    accuracy_per_bin = merged_analysis_df.groupby('risk_bin')[['is_correct_guarded', 'is_correct_baseline']].mean()
    
    plt.figure(figsize=(12, 7))
    sns.lineplot(data=accuracy_per_bin, markers=True, dashes=False)
    plt.title('Model Accuracy vs. Predicted Hallucination Risk')
    plt.xlabel('Predicted Risk Quantile (0 = Lowest Risk, 9 = Highest Risk)')
    plt.ylabel('Accuracy (1 - Hallucination Rate)')
    plt.grid(True, linestyle='--')
    plt.ylim(0, 1.05)
    plt.xticks(np.arange(len(accuracy_per_bin)))
    plt.legend(title='Model Version', labels=['Guarded Model', 'Baseline Model'])
    plt.savefig(config.RISK_COVERAGE_PLOT_PATH)
    print(f"Risk-coverage plot saved to {config.RISK_COVERAGE_PLOT_PATH}")

    print("\n--- Step 4 Complete ---")

if __name__ == "__main__":
    main()