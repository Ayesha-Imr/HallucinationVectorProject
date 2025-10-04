# 3_tune_guardrail_hyperparameters.py
"""
Step 3: Tune Guardrail Hyperparameters (Alpha and Tau).

This script replicates the logic from the third notebook (v3_...):
1.  Loads the necessary artifacts (vector, classifier) and the TruthfulQA dataset.
2.  Splits TruthfulQA into validation and test sets if not already done.
3.  Tunes the steering coefficient (alpha) by generating responses across a search
    space, judging them for hallucination and coherence, and saving results resiliently.
4.  Analyzes the tuning results to find the optimal alpha that minimizes hallucination
    while maintaining an acceptable coherence level.
5.  Tunes the risk thresholds (tau) by calculating risk scores for the entire
    validation set and finding the values corresponding to specified percentiles.
6.  Saves the final tuned thresholds (`risk_thresholds.joblib`) for use in the
    final evaluation step.
"""
import os
import time
import pandas as pd
import numpy as np
import torch
import joblib
from tqdm import tqdm
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import config
import utils

def generate_steered_answer(prompt, alpha_value, model, tokenizer, v_halluc):
    """Generates an answer from the model with a specific steering coefficient."""
    full_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nYou are a helpful assistant.\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_token_length = inputs.input_ids.shape[1]

    with utils.ActivationSteerer(model, v_halluc, layer_idx=config.TARGET_LAYER, coeff=alpha_value):
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    newly_generated_tokens = outputs[0, input_token_length:]
    answer = tokenizer.decode(newly_generated_tokens, skip_special_tokens=True)
    return answer

def main():
    """Main execution function for tuning hyperparameters."""
    print("--- Step 3: Tuning Guardrail Hyperparameters ---")

    # 1. Setup
    secrets = utils.load_secrets()
    api_key = secrets.get('SCALEDOWN_API_KEY')
    if not api_key:
        print("ERROR: SCALEDOWN_API_KEY not found. Cannot proceed. Exiting.")
        return

    model, tokenizer = utils.load_model_and_tokenizer()

    required_artifacts = [config.VECTOR_PATH, config.CLASSIFIER_PATH]
    for path in required_artifacts:
        if not os.path.exists(path):
            print(f"ERROR: Required artifact not found at {path}. Please run previous steps. Exiting.")
            return

    v_halluc = torch.load(config.VECTOR_PATH).to(model.device)
    risk_classifier = joblib.load(config.CLASSIFIER_PATH)
    print("All necessary artifacts loaded.")

    # 2. Data Preparation
    if not os.path.exists(config.VALIDATION_SET_PATH) or not os.path.exists(config.FINAL_TEST_SET_PATH):
        print("\nSplitting TruthfulQA dataset into validation and test sets...")
        full_dataset = load_dataset(config.TRUTHFULQA_DATASET_NAME, split="train")
        split_dataset = full_dataset.train_test_split(test_size=config.TRUTHFULQA_VALIDATION_SIZE, seed=42, shuffle=True)
        
        validation_df = split_dataset['test'].to_pandas()
        final_test_df = split_dataset['train'].to_pandas()
        
        validation_df.to_csv(config.VALIDATION_SET_PATH, index=False)
        final_test_df.to_csv(config.FINAL_TEST_SET_PATH, index=False)
        print(f"Validation set ({len(validation_df)} rows) saved to {config.VALIDATION_SET_PATH}")
        print(f"Final test set ({len(final_test_df)} rows) saved to {config.FINAL_TEST_SET_PATH}")
    else:
        print("\nFound existing validation and test sets.")

    validation_df = pd.read_csv(config.VALIDATION_SET_PATH)
    
    # Create a smaller, stratified tuning set for faster alpha tuning
    _, tuning_df = train_test_split(
        validation_df,
        train_size=config.TRUTHFULQA_TUNING_SAMPLE_SIZE,
        random_state=42,
        shuffle=True,
        stratify=validation_df['Category'] if 'Category' in validation_df.columns else None
    )
    print(f"Using a stratified sample of {len(tuning_df)} prompts for alpha tuning.")

    # 3. Tune Alpha (α) - Resilient Loop
    print("\nStarting alpha (α) tuning...")
    if os.path.exists(config.ALPHA_TUNING_RESULTS_PATH):
        results_df = pd.read_csv(config.ALPHA_TUNING_RESULTS_PATH)
        print(f"Resuming from existing results file with {len(results_df)} entries.")
    else:
        results_df = pd.DataFrame(columns=['alpha', 'question_index', 'Question', 'generated_answer', 'hallucination_score', 'coherence_score'])
        print("Starting new tuning run.")

    for alpha in tqdm(config.ALPHA_SEARCH_SPACE, desc="Overall Alpha Progress"):
        for index, row in tqdm(tuning_df.iterrows(), total=len(tuning_df), desc=f"Processing Alpha = {alpha}", leave=False):
            is_done = not results_df[(results_df['alpha'] == alpha) & (results_df['question_index'] == index)].empty
            if is_done:
                continue

            answer = generate_steered_answer(row['Question'], alpha, model, tokenizer, v_halluc)
            
            hall_score, coh_score = -1, -1
            for attempt in range(3): # Retry logic
                hall_score = utils.get_judge_score(row['Question'], answer, 'hallucination', api_key, **row.to_dict())
                coh_score = utils.get_judge_score(row['Question'], answer, 'coherence', api_key)
                if hall_score != -1 and coh_score != -1:
                    break
                time.sleep(5)
            
            new_row = pd.DataFrame([{'alpha': alpha, 'question_index': index, 'Question': row['Question'],
                                     'generated_answer': answer, 'hallucination_score': hall_score,
                                     'coherence_score': coh_score}])
            results_df = pd.concat([results_df, new_row], ignore_index=True)
            results_df.to_csv(config.ALPHA_TUNING_RESULTS_PATH, index=False)

    print("Alpha tuning data collection complete.")

    # 4. Programmatic Selection of Optimal Alpha
    print("\nAnalyzing alpha tuning results to select optimal alpha...")
    summary_df = results_df.groupby('alpha').agg(
        avg_hallucination_rate=('hallucination_score', lambda x: x[x!=-1].mean() / 100.0),
        avg_coherence=('coherence_score', lambda x: x[x!=-1].mean())
    ).reset_index()

    baseline_coherence = summary_df.loc[summary_df['alpha'] == 0.0, 'avg_coherence'].iloc[0]
    admissible_alphas_df = summary_df[summary_df['avg_coherence'] >= config.ACCEPTABLE_COHERENCE_THRESHOLD]

    if not admissible_alphas_df.empty:
        optimal_row = admissible_alphas_df.loc[admissible_alphas_df['avg_hallucination_rate'].idxmin()]
        OPTIMAL_ALPHA = optimal_row['alpha']
        print(f"--- Optimal Alpha Selected (Trade-off) ---")
        print(f"Optimal alpha selected: {OPTIMAL_ALPHA}")
        print(f"At this value, Avg Hallucination Rate = {optimal_row['avg_hallucination_rate']:.2%} and Avg Coherence = {optimal_row['avg_coherence']:.2f}")
    else:
        print("--- Fallback: No alpha met the coherence threshold ---")
        optimal_row = summary_df.loc[summary_df['avg_hallucination_rate'].idxmin()]
        OPTIMAL_ALPHA = optimal_row['alpha']
        print(f"Selecting alpha with minimum hallucination rate as our candidate: {OPTIMAL_ALPHA}")

    # Plotting
    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax1.plot(summary_df['alpha'], summary_df['avg_hallucination_rate'], color='tab:red', marker='o', label='Hallucination Rate')
    ax1.set_xlabel('Steering Coefficient (α)')
    ax1.set_ylabel('Average Hallucination Rate', color='tab:red')
    ax2 = ax1.twinx()
    ax2.plot(summary_df['alpha'], summary_df['avg_coherence'], color='tab:blue', marker='x', linestyle='--', label='Coherence Score')
    ax2.set_ylabel('Average Coherence Score', color='tab:blue')
    ax2.axhline(y=config.ACCEPTABLE_COHERENCE_THRESHOLD, color='gray', linestyle=':', linewidth=2, label=f'Coherence Threshold ({config.ACCEPTABLE_COHERENCE_THRESHOLD})')
    fig.suptitle('Effect of Steering Coefficient (α) on Hallucination and Coherence')
    fig.legend()
    plt.savefig(config.ALPHA_TUNING_PLOT_PATH)
    print(f"Alpha tuning plot saved to {config.ALPHA_TUNING_PLOT_PATH}")

    # 5. Tune Thresholds (τ)
    print("\nCalculating risk scores for validation set to determine thresholds...")
    risk_scores = []
    for prompt in tqdm(validation_df['Question'], desc="Calculating risk scores"):
        risk = utils.get_hallucination_risk(prompt, model, tokenizer, v_halluc, risk_classifier)
        risk_scores.append(risk)
    
    validation_df['risk_score'] = risk_scores
    
    TAU_LOW = np.percentile(validation_df['risk_score'], config.TAU_LOW_PERCENTILE)
    TAU_HIGH = np.percentile(validation_df['risk_score'], config.TAU_HIGH_PERCENTILE)
    
    threshold_values = {'tau_low': TAU_LOW, 'tau_high': TAU_HIGH, 'optimal_alpha': OPTIMAL_ALPHA}
    joblib.dump(threshold_values, config.RISK_THRESHOLDS_PATH)

    print("\n--- Final Tuned Hyperparameters ---")
    print(f"Optimal Steering Coefficient (α): {OPTIMAL_ALPHA}")
    print(f"Low Risk Threshold (τ_low) ({config.TAU_LOW_PERCENTILE}th percentile): {TAU_LOW:.4f}")
    print(f"High Risk Threshold (τ_high) ({config.TAU_HIGH_PERCENTILE}th percentile): {TAU_HIGH:.4f}")
    print(f"Thresholds saved to: {config.RISK_THRESHOLDS_PATH}")
    
    print("\n--- Step 3 Complete ---")

if __name__ == "__main__":
    main()